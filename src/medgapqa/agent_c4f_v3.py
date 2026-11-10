"""D36 C4f v3 (new file; agent_c4f.py v2 and frozen files unchanged). Same prompt as v2 (src/medgapqa/prompts_c4f, sha256 0f22a2ad…5745), last-turn rule, violation re-request rule and forced answer.
One difference from v2 (decoding constraint): on tool turns (1..4) the action name is restricted to the two candidates ASK and SEARCH — after the prefix PREFILL ('{"action": "'),
the summed token log-probabilities of the two candidate strings CONT['ASK'] = 'ASK", "content": "' and CONT['SEARCH'] = 'SEARCH", "content": "' are computed on the same input and the larger one (ASK on ties) is chosen;
that string is appended to the prefix and only content is generated greedily (re-request generation works the same way). The last turn (5) is generated without constraint, as in v2.
"""
from agent_c4f import FORCE_TOOL
from prompts_c4f import c4f_system_prompt
from run_agent import template_messages
from agent import (CONDITIONS, FIRST_TRAILER, FINAL_ACTIONS, NEXT_TRAILER, PREFILL, PROMPT_VERSION, STATES, _sha, _with_trailer,
                   case_text, format_results, last_turn_sentence, protocol, retry_message, tools_of, turns_for, validate)
import agent as A

CONT = {"ASK": 'ASK", "content": "', "SEARCH": 'SEARCH", "content": "'}
COND = "C4"


def _parts(model):
    """EvalModel → (tokenizer, causal LM, render). render builds the same prompt string as each model's chat()."""
    if hasattr(model, "sc"):
        hf = model.sc.hf
        return hf.tok, hf.model, lambda msgs: hf.tok.apply_chat_template(template_messages(hf.name, msgs), add_generation_prompt=True, tokenize=False)
    lm = model.lm
    return lm.tok, lm.model, lm.render


def choose_tool(model, msgs):
    import torch
    tok, lm, render = _parts(model)
    prompt = render(msgs) + PREFILL
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    score = {}
    for a, c in CONT.items():
        f_ids = tok(prompt + c, add_special_tokens=False)["input_ids"]
        assert f_ids[:len(p_ids)] == p_ids, "tokenization boundary mismatch"
        x = torch.tensor([f_ids], device=lm.device)
        with torch.no_grad():
            lp = torch.log_softmax(lm(x).logits[0].float(), -1)
        score[a] = float(sum(lp[k - 1, f_ids[k]].item() for k in range(len(p_ids), len(f_ids))))
    return {"choice": "ASK" if score["ASK"] >= score["SEARCH"] else "SEARCH", "logprob": score}


def run_episode_c4f_v3(item, state, cond, model, cfg, simulator=None, retriever=None):
    assert cond == COND and protocol(cfg) == "v2"
    allowed = CONDITIONS[cond]
    mode = STATES[state]
    visible = item["question"] if state == "S0" else item["pert_question"]
    withheld = None if state == "S0" else item["removed_fact"]
    history = [{"role": "system", "content": c4f_system_prompt(cond, cfg)},
               {"role": "user", "content": case_text(visible, item["options"])}]
    trailer = FIRST_TRAILER
    counts = {a: 0 for a in A.ACTIONS}
    turns, outcome, end_reason, answer = [], None, None, None
    tok_in = tok_out = 0
    T = turns_for(cond, cfg)
    tools = tools_of(cond)
    violations = []
    choices = []

    def gen(msgs, turn):
        if turn < T:
            r = choose_tool(model, msgs)
            choices.append({"turn": turn, **r})
            return model.chat(msgs, PREFILL + CONT[r["choice"]], cfg["max_new_tokens"])
        return model.chat(msgs, PREFILL, cfg["max_new_tokens"])


    def check(raw, trunc, turn):
        act, why = validate(raw, trunc, allowed, item["options"], cfg["answer_normalization"])
        if act is None:
            return act, why
        if turn == T and act["action"] in tools:
            return None, "last_turn_tool"
        if turn < T and act["action"] in FINAL_ACTIONS:
            return act, "final_action_in_tool_turn"
        return act, why

    def retry_text(why):
        if why == "last_turn_tool":
            return last_turn_sentence(cond)
        if why == "final_action_in_tool_turn":
            return FORCE_TOOL
        return retry_message(why)

    for turn in range(1, T + 1):
        if turn == T:
            trailer = last_turn_sentence(cond) + "\n\n" + NEXT_TRAILER
        msgs = _with_trailer(history, trailer)
        raw, n_in, n_out, trunc = gen(msgs, turn)
        tok_in, tok_out = tok_in + n_in, tok_out + n_out
        act, why = check(raw, trunc, turn)
        rec = {"turn": turn, "input_sha": _sha(msgs), "raw": raw, "truncated": trunc, "invalid_reason": why,
               "tokens": {"in": n_in, "out": n_out}}
        if why == "final_action_in_tool_turn":
            violations.append({"turn": turn, "attempt": 1, "action": act["action"]})
        if why is not None:
            retry_msgs = msgs + [{"role": "assistant", "content": raw}, {"role": "user", "content": retry_text(why)}]
            raw2, n_in2, n_out2, trunc2 = gen(retry_msgs, turn)
            tok_in, tok_out = tok_in + n_in2, tok_out + n_out2
            act, why2 = check(raw2, trunc2, turn)
            rec.update(retry_raw=raw2, retry_truncated=trunc2, retry_invalid_reason=why2, retry_tokens={"in": n_in2, "out": n_out2})
            raw = raw2
            if why2 == "final_action_in_tool_turn":
                violations.append({"turn": turn, "attempt": 2, "action": act["action"]})
            if why2 in ("last_turn_tool", "final_action_in_tool_turn"):
                rec["action"] = why2
                if act is not None:
                    rec["attempted_action"], rec["content"] = act["action"], act["content"]
                turns.append(rec)
                outcome = "unresolved"
                end_reason = "last_turn_tool_after_retry" if why2 == "last_turn_tool" else "final_action_in_tool_turn_after_retry"
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

    first_ask = next((t["turn"] for t in turns if t.get("action") == "ASK"), None)
    return {
        "id": item["id"], "hash": item["hash"], "state": state, "cond": cond, "condition": "C4f",
        "removed_type": item.get("removed_type"), "prompt_version": PROMPT_VERSION,
        "prompt_variant": cfg.get("prompt_variant", "v4"), "protocol": protocol(cfg), "T": T, "N_tool_turns": T - 1,
        "outcome": outcome, "end_reason": end_reason, "answer": answer,
        "unresolved_cause": end_reason if outcome == "unresolved" else None,
        "budget_exhausted": None, "forced_answer": None, "n_turns": len(turns), "action_counts": counts,
        "first_action": turns[0]["action"], "first_ask_turn": first_ask, "tokens": {"in": tok_in, "out": tok_out}, "turns": turns,
        "c4f": {"violations": violations, "force_tool_sentence": FORCE_TOOL, "version": "v3", "tool_choices": choices},
    }
