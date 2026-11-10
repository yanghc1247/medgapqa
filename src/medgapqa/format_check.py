"""Comparison of action-format candidates (DEV, S0 original items, C4, first action only, one re-request). Accuracy is neither computed nor printed.

Candidate A: prefill the reply with '{"action": "' and output a single JSON object only.
Candidate B: short reasoning (at most 3 sentences) followed by a single JSON object at the end.
Action order: canonical (ANSWER, ASK, SEARCH, ABSTAIN) / reversed. Action definitions and examples are both reordered the same way.
Parser: extracts the last JSON object. format_clean = the whole output is exactly one JSON object.
Truncation (limit reached, no stop token) counts as invalid even if JSON is present and gets a re-request; if the re-request is also truncated, parse_fail.
For the selection rule see logs/decisions.md (recorded before seeing results).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_manifest  # noqa: E402
from parse_check import CACHE, MODELS, REVISIONS, user_msg  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ["ANSWER", "ASK", "SEARCH", "ABSTAIN"]
MAX_NEW = {"A": 256, "B": 384}  # A: 128 up to v2, 256 from v3 (2026-10-16)
PREFILL = '{"action": "'

DEFS = {
    "ANSWER": 'ANSWER: commit to one option. "content" is the option letter only. Ends the case.',
    "ASK": 'ASK: ask one question about this patient. "content" is the question.',
    "SEARCH": 'SEARCH: search a general medical knowledge corpus. "content" is the query.',
    "ABSTAIN": 'ABSTAIN: decline and hand the case to a human. "content" is a brief reason. Ends the case.',
}
# examples of the four actions: placeholders in the same format without medical content, of similar length (v2, 2026-10-16)
EXAMPLES = {
    "ANSWER": '{"action": "ANSWER", "content": "<one option letter>"}',
    "ASK": '{"action": "ASK", "content": "<one question to ask>"}',
    "SEARCH": '{"action": "SEARCH", "content": "<one search query>"}',
    "ABSTAIN": '{"action": "ABSTAIN", "content": "<one brief reason>"}',
}
HEADER = "You are a clinical decision agent working on a multiple-choice medical question. At each turn you choose exactly one action."
COST = "Gathering information costs time, so use ASK or SEARCH only when needed."
CONTENT_LIMIT = 'Keep "content" to at most 30 words.'  # v3 (2026-10-16, user decision)
FORMAT = {
    "A": "Output exactly one JSON object and nothing else. " + CONTENT_LIMIT,
    "B": "First think briefly (at most 3 sentences). Then, on the last line, output exactly one JSON object and nothing after it.",
}
RETRY = {
    "A": "Your previous output was not a valid action. Output exactly one JSON object with the keys \"action\" and \"content\".",
    "B": "Your previous output did not end with a valid action. Think briefly, then end with exactly one JSON object with the keys \"action\" and \"content\".",
}


def system_prompt(cand, order):
    acts = ACTIONS if order == "canonical" else ACTIONS[::-1]
    lines = [HEADER, "", "Actions:"] + [f"- {DEFS[a]}" for a in acts]
    lines += ["", COST, "", FORMAT[cand], "Examples of valid actions:"] + [EXAMPLES[a] for a in acts]
    return "\n".join(lines)


def parse_last_json(text):
    """Returns the last complete JSON object (dict)."""
    dec = json.JSONDecoder()
    last = None
    i = text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
            if isinstance(obj, dict):
                last = (obj, i, end)
            i = text.find("{", end)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
    return last


def judge(text):
    found = parse_last_json(text)
    if not found:
        return None, False
    obj, s, e = found
    valid = obj.get("action") in ACTIONS and isinstance(obj.get("content"), str) and set(obj) == {"action", "content"}
    if valid and obj["action"] == "ANSWER":
        valid = obj["content"].strip().upper() in list("ABCD")
    clean = valid and text.strip() == text[s:e].strip() and text.count("{") == 1
    return (obj if valid else None), clean


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--candidates", default="A,B")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    repo, rev = MODELS[args.model], REVISIONS[args.model]
    cache = CACHE.get(args.model, str(ROOT / "hf_cache" / "hub"))
    tok = AutoTokenizer.from_pretrained(repo, cache_dir=cache, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(repo, cache_dir=cache, revision=rev, dtype=torch.bfloat16,
                                                 device_map={"": "cuda:0"})
    model.eval()
    eos = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|eot_id|>")] if "<|eot_id|>" in tok.get_vocab() else [])

    def gen(msgs, prefill, max_new):
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + prefill
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=max_new,
                                 eos_token_id=eos, pad_token_id=tok.pad_token_id or eos[0])
        new = out[0, ids.shape[1]:]
        # truncation: generated up to the limit and the last token is not a stop token. Counted as invalid (toward parse_fail) for every model.
        truncated = len(new) >= max_new and int(new[-1]) not in eos
        return prefill + tok.decode(new, skip_special_tokens=True).strip(), truncated

    items = load_manifest("dev")[: args.n]
    out_dir = ROOT / "runs" / "format_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for cand in args.candidates.split(","):
        prefill = PREFILL if cand == "A" else ""
        max_new = MAX_NEW[cand]
        for order in ["canonical", "reversed"]:
            rows = []
            for it in items:
                msgs = [{"role": "system", "content": system_prompt(cand, order)}, {"role": "user", "content": user_msg(it)}]
                s = time.time()
                raw1, t1 = gen(msgs, prefill, max_new)
                a1, c1 = (None, False) if t1 else judge(raw1)
                raw2, a2, c2, t2 = None, None, False, False
                if a1 is None:
                    raw2, t2 = gen(msgs + [{"role": "assistant", "content": raw1}, {"role": "user", "content": RETRY[cand]}],
                                   prefill, max_new)
                    a2, c2 = (None, False) if t2 else judge(raw2)
                final = a1 or a2
                rows.append({"id": it["id"], "raw1": raw1, "ok1": a1 is not None, "clean1": c1, "raw2": raw2, "truncated1": t1, "truncated2": t2,
                             "ok_after_retry": final is not None, "format_clean": c1 if a1 else c2,
                             "action": final["action"] if final else "parse_fail", "content_words": len(final["content"].split()) if final else None, "seconds": round(time.time() - s, 2)})
            with open(out_dir / f"{args.model}{args.tag}_{cand}_{order}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            summary.append({"model": args.model, "tag": args.tag, "candidate": cand, "order": order, "n": len(rows),
                            "ok_first_try": sum(r["ok1"] for r in rows),
                            "parse_fail_after_retry": sum(not r["ok_after_retry"] for r in rows),
                            "format_clean": sum(r["format_clean"] for r in rows),
                            "truncated_first_try": sum(r["truncated1"] for r in rows),
                            "truncated_final": sum((r["truncated2"] if r["raw2"] is not None else r["truncated1"])
                                                   for r in rows),
                            "max_new_tokens": max_new,
                            "content_over_30_words": sum((r["content_words"] or 0) > 30 for r in rows),
                            "first_action_counts": {a: sum(r["action"] == a for r in rows) for a in ACTIONS + ["parse_fail"]},
                            "sec_mean": round(sum(r["seconds"] for r in rows) / len(rows), 2)})
            print(json.dumps(summary[-1]), flush=True)
    json.dump(summary, open(out_dir / f"{args.model}{args.tag}_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
