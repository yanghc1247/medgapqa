"""Checks with DEV items whether evaluation models follow the JSON action format (first action only, C4, S0 original items).

The prompt is a draft (DRAFT) before W2 was settled. Accuracy is neither computed nor printed.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "olmo2-7b": "allenai/OLMo-2-1124-7B-Instruct",
    "meerkat-8b": "dmis-lab/llama-3-meerkat-8b-v1.0",
    "medgemma-4b": "google/medgemma-4b-it",  # deviations D15
}
REVISIONS = {
    "qwen2.5-7b": "a09a35458c702b33eeacc393d103063234e8bc28",
    "olmo2-7b": "470b1fba1ae01581f270116362ee4aa1b97f4c84",
    "meerkat-8b": "ba8405f1d873b99004b2d121548fcafd33217eb7",
    "medgemma-4b": "290cda5eeccbee130f987c4ad74a59ae6f196408",
}
CACHE = {
    "meerkat-8b": "$HF_CACHE/hub",  # read-only
}
ACTIONS = ["ANSWER", "ASK", "SEARCH", "ABSTAIN"]

SYSTEM_DRAFT = """You are a clinical decision agent answering a multiple-choice medical question. At each turn you must output exactly one action as a single JSON object and nothing else:
{"action": "ANSWER|ASK|SEARCH|ABSTAIN", "content": "..."}

Actions:
- ANSWER: commit to an option. "content" is the option letter only (A, B, C, or D). Ends the case.
- ASK: ask the patient/clinician one question about this patient. "content" is the question.
- SEARCH: search a general medical knowledge corpus. "content" is the search query.
- ABSTAIN: decline to answer and hand the case to a human. "content" is a brief reason. Ends the case.

Gathering information costs time, so use ASK or SEARCH only when needed. Output only the JSON object."""

RETRY_MSG = 'Your previous output was not a valid action. Output exactly one JSON object: {"action": "ANSWER|ASK|SEARCH|ABSTAIN", "content": "..."}'


def parse_action(text):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if obj.get("action") not in ACTIONS or not isinstance(obj.get("content"), str):
        return None
    if obj["action"] == "ANSWER" and obj["content"].strip().upper() not in list("ABCD"):
        return None
    return obj


def user_msg(item):
    opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    return f"Question:\n{item['question']}\n\nOptions:\n{opts}\n\nYour action:"


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    repo = MODELS[args.model]
    cache = CACHE.get(args.model, str(ROOT / "hf_cache" / "hub"))
    rev = REVISIONS[args.model]
    tok = AutoTokenizer.from_pretrained(repo, cache_dir=cache, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(repo, cache_dir=cache, revision=rev, dtype=torch.bfloat16, device_map={"": "cuda:0"})
    model.eval()
    eos = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|eot_id|>")] if "<|eot_id|>" in tok.get_vocab() else [])

    def gen(msgs):
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)["input_ids"].to(model.device)
        with torch.no_grad():
            out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=256,
                                 eos_token_id=eos, pad_token_id=tok.pad_token_id or eos[0])
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    rows = []
    for it in load_manifest("dev")[: args.n]:
        msgs = [{"role": "system", "content": SYSTEM_DRAFT}, {"role": "user", "content": user_msg(it)}]
        s = time.time()
        raw1 = gen(msgs)
        a1 = parse_action(raw1)
        raw2 = a2 = None
        if a1 is None:
            raw2 = gen(msgs + [{"role": "assistant", "content": raw1}, {"role": "user", "content": RETRY_MSG}])
            a2 = parse_action(raw2)
        final = a1 or a2
        rows.append({"id": it["id"], "raw1": raw1, "ok1": a1 is not None, "raw2": raw2, "ok_after_retry": final is not None,
                     "action": final["action"] if final else "parse_fail", "seconds": round(time.time() - s, 2)})
    out = ROOT / "runs" / "parse_check"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.model}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    summ = {"model": args.model, "repo": repo, "n": n,
            "parse_ok_first_try": sum(r["ok1"] for r in rows),
            "parse_fail_after_retry": sum(not r["ok_after_retry"] for r in rows),
            "first_action_counts": {a: sum(r["action"] == a for r in rows) for a in ACTIONS + ["parse_fail"]}}
    print(json.dumps(summ, ensure_ascii=False))
    json.dump(summ, open(out / f"{args.model}.summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
