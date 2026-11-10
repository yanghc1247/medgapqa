"""Agent loop (plan §4). Model, simulator and retriever are injected (unit tests run on CPU with fakes).

model.chat(messages, prefill, max_new_tokens) -> (text, n_in, n_out, truncated)
  text is the full output including the prefill. truncated = limit reached, no stop token.

Returns the trajectory of one item in one state as a dict. Correctness (correct/wrong) is not computed here (src/medgapqa/grade.py).
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_check import parse_last_json  # noqa: E402
from prompts import DEFS, EXAMPLES, FORMAT, GUIDANCE, GUIDANCE_ACTION, HEADER, PREFILL, PROMPT_VERSION, retry_message  # noqa: E402
from retrieve import format_results  # noqa: E402

ACTIONS = ["ANSWER", "ASK", "SEARCH", "ABSTAIN"]
CONDITIONS = {
    "C0": ["ANSWER"],
    "C1": ["ANSWER", "ABSTAIN"],
    "C2": ["ANSWER", "ASK"],
    "C3": ["ANSWER", "SEARCH"],
    "C4": ["ANSWER", "ASK", "SEARCH", "ABSTAIN"],
}
STATES = {"S0": "s0", "S1": "recoverable", "S2": "unrecoverable"}
FIRST_TRAILER = "Your action:"
NEXT_TRAILER = "Your next action:"
FORCED_PREFILL = '{"action": "ANSWER", "content": "'
# protocol v2 (deviations D8, 2026-10-16): per condition T = sum of tool limits + 1, last turn ANSWER/ABSTAIN only,
# exceeding a tool limit gets one retry with a fixed sentence; forced answer disabled. v1 (to reproduce earlier analyses) is the default when cfg has no protocol.
LAST_TURN = "This is your last action. Choose {acts}."
OVER_LIMIT = "You have already used {a} the maximum number of times. Choose another action."
FINAL_ACTIONS = ("ANSWER", "ABSTAIN")
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "agent.json"


def load_config(path=CONFIG, allow_provisional=True):
    cfg = json.load(open(path))
    if cfg["provisional"] and not allow_provisional:
        raise SystemExit(f"provisional config values not confirmed: {sorted(cfg['provisional'])}")
    return {**cfg["confirmed"], **cfg["provisional"]}


def protocol(cfg):
    return cfg.get("protocol", "v1")


def tools_of(cond):
    return [a for a in ("ASK", "SEARCH") if a in CONDITIONS[cond]]


def turns_for(cond, cfg):
    """v1: T for every condition. v2: for conditions with tools, sum of tool limits + 1 (C2 3, C3 3, C4 5); C0 and C1 keep T."""
    tools = tools_of(cond)
    if protocol(cfg) == "v1" or not tools:
        return cfg["T"]
    lim = {"ASK": cfg["ask_max"], "SEARCH": cfg["search_max"]}
    return sum(lim[a] for a in tools) + 1


def last_turn_sentence(cond):
    return LAST_TURN.format(acts=" or ".join(a for a in FINAL_ACTIONS if a in CONDITIONS[cond]))


def system_prompt(cond, cfg):
    acts = [a for a in ACTIONS if a in CONDITIONS[cond]]
    lines = [HEADER, "", "Actions:"] + [f"- {DEFS[a]}" for a in acts]
    tools = [a for a in ("ASK", "SEARCH") if a in acts]
    if tools:
        variant = cfg.get("prompt_variant", "v4")
        g = GUIDANCE[variant]
        if variant == "p2":
            g = [x for k, x in enumerate(g) if k == 0 or GUIDANCE_ACTION[k] in acts]
        lines += [""] + [x.replace("{tools}", " or ".join(tools)) for x in g]
        if cfg.get("budget_disclosure"):
            lim = {"ASK": cfg["ask_max"], "SEARCH": cfg["search_max"]}
            lines.append(f"You can take at most {turns_for(cond, cfg)} actions in total. "
                         + " ".join(f"You can use {a} at most {lim[a]} times." for a in tools))
    lines += ["", FORMAT, "Examples of valid actions:"] + [EXAMPLES[a] for a in acts]
    return "\n".join(lines)


def case_text(question, options):
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"Question:\n{question}\n\nOptions:\n{opts}"


def _norm_opt(t):
    return re.sub(r"\s+", " ", t).strip().strip(".;:,!").strip().lower()


def normalize_answer(content, options):
    """Pre-recorded normalization rule (logs/decisions.md): that letter only when the output starts with a letter and the following text matches that letter's option."""
    m = re.fullmatch(r"\(?([A-Da-d])\s*(?:[.):\-]|\s)\s*(.+)", content.strip(), re.S)
    if not m:
        return None
    letter = m.group(1).upper()
    if letter in options and _norm_opt(m.group(2)) == _norm_opt(options[letter]):
        return letter
    return None


def validate(text, truncated, allowed, options=None, normalize=False):
    """(action dict or None, invalid reason). For ANSWER, format_clean_answer is recorded as well."""
    if truncated:
        return None, "truncated"
    found = parse_last_json(text)
    if not found:
        return None, "no_json"
    obj = found[0]
    if set(obj) != {"action", "content"} or obj.get("action") not in ACTIONS or not isinstance(obj.get("content"), str):
        return None, "bad_schema"
    if obj["action"] not in allowed:
        return None, "disallowed_action"
    content = obj["content"].strip()
    if obj["action"] == "ANSWER":
        if content.upper() in list("ABCD"):
            return {"action": "ANSWER", "content": content.upper(), "format_clean_answer": True}, None
        letter = normalize_answer(content, options or {}) if normalize else None
        if letter is None:
            return None, "bad_letter"
        return {"action": "ANSWER", "content": letter, "format_clean_answer": False}, None
    elif not content:
        return None, "empty_content"
    return {"action": obj["action"], "content": content}, None


def _sha(msgs):
    return hashlib.sha256(json.dumps(msgs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def _with_trailer(msgs, trailer):
    """Copy with the instruction appended to the end of the last user message (not appended to the recorded history)."""
    out = [dict(m) for m in msgs]
    out[-1]["content"] = out[-1]["content"] + "\n\n" + trailer
    return out


def run_episode(item, state, cond, model, cfg, simulator=None, retriever=None):
    """item: id, hash, question (original), pert_question, removed_fact, removed_type, options. answer_idx is not read."""
    allowed = CONDITIONS[cond]
    mode = STATES[state]
    visible = item["question"] if state == "S0" else item["pert_question"]
    withheld = None if state == "S0" else item["removed_fact"]
    history = [{"role": "system", "content": system_prompt(cond, cfg)},
               {"role": "user", "content": case_text(visible, item["options"])}]
    trailer = FIRST_TRAILER
    counts = {a: 0 for a in ACTIONS}
    limit = {"ASK": cfg["ask_max"], "SEARCH": cfg["search_max"]}
    turns, outcome, end_reason, answer = [], None, None, None
    tok_in = tok_out = 0

    v2 = protocol(cfg) == "v2"
    T = turns_for(cond, cfg)
    tools = tools_of(cond)

    def check(raw, trunc, turn):
        """Adds the v2 rules to validate: tool attempt on the last turn (last_turn_tool), over the limit (over_limit)."""
        act, why = validate(raw, trunc, allowed, item["options"], cfg["answer_normalization"])
        if act is None or not v2 or act["action"] not in limit:
            return act, why
        if tools and turn == T:
            return None, "last_turn_tool"
        if counts[act["action"]] >= limit[act["action"]]:
            return {**act, "over": True}, "over_limit"
        return act, why

    def retry_text(why, act):
        if why == "last_turn_tool":
            return last_turn_sentence(cond)
        if why == "over_limit":
            return OVER_LIMIT.format(a=act["action"])
        return retry_message(why)

    for turn in range(1, T + 1):
        if v2 and tools and turn == T:
            trailer = last_turn_sentence(cond) + "\n\n" + NEXT_TRAILER
        msgs = _with_trailer(history, trailer)
        raw, n_in, n_out, trunc = model.chat(msgs, PREFILL, cfg["max_new_tokens"])
        tok_in, tok_out = tok_in + n_in, tok_out + n_out
        act, why = check(raw, trunc, turn)
        rec = {"turn": turn, "input_sha": _sha(msgs), "raw": raw, "truncated": trunc, "invalid_reason": why,
               "tokens": {"in": n_in, "out": n_out}}
        if why is not None:
            retry_msgs = msgs + [{"role": "assistant", "content": raw}, {"role": "user", "content": retry_text(why, act)}]
            raw2, n_in2, n_out2, trunc2 = model.chat(retry_msgs, PREFILL, cfg["max_new_tokens"])
            tok_in, tok_out = tok_in + n_in2, tok_out + n_out2
            act, why2 = check(raw2, trunc2, turn)
            rec.update(retry_raw=raw2, retry_truncated=trunc2, retry_invalid_reason=why2,
                       retry_tokens={"in": n_in2, "out": n_out2})
            raw = raw2
            if why2 in ("over_limit", "last_turn_tool"):
                # still violating a tool limit or the last-turn rule after the retry: ends unresolved, distinct from parse_fail (no observation)
                rec["action"] = why2
                if act is not None:
                    rec["attempted_action"], rec["content"] = act["action"], act["content"]
                turns.append(rec)
                outcome = "unresolved"
                end_reason = ("last_turn_tool_after_retry" if why2 == "last_turn_tool"
                              else f"{act['action'].lower()}_limit_after_retry")
                break
            if why2 is not None:
                act = None
        if act is None:
            rec["action"] = "parse_fail"
            turns.append(rec)
            outcome, end_reason = "parse_fail", "parse_fail"
            break
        rec.update(action=act["action"], content=act["content"])
        if "format_clean_answer" in act:
            rec["format_clean_answer"] = act["format_clean_answer"]
        a = act["action"]
        if a in limit and counts[a] >= limit[a]:
            # v1 repetition limit exceeded: no observation is given and the episode ends unresolved (repeat_limit_policy)
            rec["over_limit"] = True
            turns.append(rec)
            outcome, end_reason = "unresolved", f"{a.lower()}_limit"
            break
        counts[a] += 1
        history.append({"role": "assistant", "content": raw})
        if a == "ANSWER":
            turns.append(rec)
            outcome, end_reason, answer = "answer", "answer", act["content"]
            break
        if a == "ABSTAIN":
            turns.append(rec)
            outcome, end_reason = "abstain", "abstain"
            break
        if a == "ASK":
            obs, sim = simulator.respond(mode, visible, withheld, act["content"])
            rec["sim"] = sim
            obs_text = f"Response: {obs}"
        else:
            results = retriever.search(act["content"])
            rec["search"] = [{k: r[k] for k in ("idx", "id", "score", "truncated")} for r in results]
            obs_text = format_results(act["content"], results)
        rec["observation"] = obs_text
        history.append({"role": "user", "content": obs_text})
        trailer = NEXT_TRAILER
        turns.append(rec)
    else:
        outcome, end_reason = "unresolved", "max_turns"

    forced = None
    if outcome == "unresolved" and not v2:
        # forced answer for v1 auxiliary evaluation (excluded from the main metric). Disabled in v2 (code kept to reproduce earlier analyses)
        msgs = _with_trailer(history, cfg["forced_answer_prompt"])
        raw, n_in, n_out, trunc = model.chat(msgs, FORCED_PREFILL, cfg["max_new_tokens"])
        letter = raw[len(FORCED_PREFILL):].strip()[:1].upper()
        forced = {"raw": raw, "answer": letter if letter in list("ABCD") else None, "truncated": trunc,
                  "tokens": {"in": n_in, "out": n_out}}

    first_ask = next((t["turn"] for t in turns if t.get("action") == "ASK"), None)
    return {
        "id": item["id"], "hash": item["hash"], "state": state, "cond": cond,
        "removed_type": item.get("removed_type"), "prompt_version": PROMPT_VERSION,
        "prompt_variant": cfg.get("prompt_variant", "v4"), "protocol": protocol(cfg), "T": T,
        "outcome": outcome, "end_reason": end_reason, "answer": answer,
        "unresolved_cause": end_reason if outcome == "unresolved" else None,
        "budget_exhausted": all(counts[a] >= limit[a] for a in tools) if tools else None,
        "forced_answer": forced, "n_turns": len(turns), "action_counts": counts, "first_action": turns[0]["action"],
        "first_ask_turn": first_ask, "tokens": {"in": tok_in, "out": tok_out}, "turns": turns,
    }
