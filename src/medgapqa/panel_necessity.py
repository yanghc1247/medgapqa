"""D26 panel necessity score and candidate selection (new file). Evaluation-model results are not used. Gold labels are used only in the score (probability of the gold letter).

Panel of 4 (Mistral-7B-Instruct-v0.3, phi-4, granite-3.1-8b-instruct, Meerkat-70B NF4). For each model m, in C0 format (agent.system_prompt("C0"),
first user turn case_text + "\\n\\nYour action:", agent.FORCED_PREFILL), p_m(gold letter) from a softmax over the 4 A–D logits only, computed at S0 and on the candidate (S1K);
d_m = p_m(S0) − p_m(candidate), candidate score = mean of the 4 d_m. Selection: highest score among valid candidates. Adoption: score ≥ 0.10 and d_m > 0 for at least 3 panel models.
Thresholds are not changed after seeing results (D26 C).

CUDA_VISIBLE_DEVICES=4 python src/medgapqa/panel_necessity.py score --model mistral-7b --cands runs/v5/devk_candidates/candidates.jsonl --out runs/v5/panel
python src/medgapqa/panel_necessity.py select --cands ... --scores runs/v5/panel --out runs/v5/devk
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/medgapqa"))
from agent import FIRST_TRAILER, FORCED_PREFILL, case_text, load_config, system_prompt  # noqa: E402
from forced_answer import LETTERS, letter_ids, logits4_sha  # noqa: E402

CACHE = "$PROJECT_ROOT/hf_cache/hub"
NF4_70B = "$PROJECT_ROOT/models/meerkat-70b-nf4"
SPECS = {
    "mistral-7b": {"repo": "mistralai/Mistral-7B-Instruct-v0.3", "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71"},
    "phi-4": {"repo": "microsoft/phi-4", "revision": "2db69c1c3e91a05d2c64a3185acfbaf36f744e25"},
    "granite-8b": {"repo": "ibm-granite/granite-3.1-8b-instruct", "revision": "4009206d5fc95d2e65a7b7633e159d6e97e25d35"},
    "meerkat-70b": {"path": NF4_70B},
    "qwen2.5-32b": {"repo": "Qwen/Qwen2.5-32B-Instruct", "revision": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd", "bnb4": True},
}
PANEL = ["mistral-7b", "phi-4", "granite-8b", "meerkat-70b"]
THRESH, MIN_POS = 0.10, 3


class LetterModel:
    """Arbitrary HF model: A–D restricted logits (next token after prefill) and greedy generation. Templates that do not accept a system role get it merged before the first user turn with "\\n\\n"."""

    def __init__(self, name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch, self.name, s = torch, name, SPECS[name]
        src = s.get("path") or s["repo"]
        kw = {} if "path" in s else {"revision": s["revision"], "cache_dir": CACHE}
        self.tok = AutoTokenizer.from_pretrained(src, local_files_only=True, **kw)
        mk = dict(dtype=torch.bfloat16, device_map={"": "cuda:0"}, local_files_only=True, **kw)
        if s.get("bnb4"):
            mk["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
        self.model = AutoModelForCausalLM.from_pretrained(src, **mk).eval()
        self.merge_system = not self._template_ok()
        self.ids = letter_ids(self.tok, self.render([{"role": "system", "content": "s"}, {"role": "user", "content": "x"}]) + FORCED_PREFILL)
        v = self.tok.get_vocab()
        self.eos = sorted({self.tok.eos_token_id, *[v[t] for t in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>", "<|end|>") if t in v]} - {None})

    def _template_ok(self):
        try:
            out = self.tok.apply_chat_template([{"role": "system", "content": "SYSMARK"}, {"role": "user", "content": "x"}],
                                               add_generation_prompt=True, tokenize=False)
            return "SYSMARK" in out
        except Exception:
            return False

    def render(self, msgs):
        if self.merge_system and msgs and msgs[0]["role"] == "system":
            msgs = [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}] + msgs[2:]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def letters(self, msgs):
        prompt = self.render(msgs) + FORCED_PREFILL
        x = self.tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.model.device)
        with self.torch.no_grad():
            lg = self.model(x).logits[0, -1].float().cpu().numpy()
        four = lg[self.ids].astype(np.float64)
        p = np.exp(four - four.max())
        p /= p.sum()
        return {"p4": dict(zip(LETTERS, map(float, p))), "logits4": [float(v) for v in four], "n_in": int(x.shape[1]),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}

    def chat(self, msgs, prefill, max_new):
        """Same format as run_agent.HFModel._chat (text, n_in, n_out, truncated). greedy."""
        prompt = self.render(msgs) + prefill
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(ids, attention_mask=self.torch.ones_like(ids), do_sample=False, max_new_tokens=max_new,
                                      eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id or self.eos[0])
        new = out[0, ids.shape[1]:]
        truncated = len(new) >= max_new and int(new[-1]) not in self.eos
        return prefill + self.tok.decode(new, skip_special_tokens=True).strip(), int(ids.shape[1]), int(len(new)), truncated


def c0_messages(visible, options, cfg):
    return [{"role": "system", "content": system_prompt("C0", cfg)},
            {"role": "user", "content": case_text(visible, options) + "\n\n" + FIRST_TRAILER}]


def cmd_score(a):
    from budget import Budget
    import time
    cfg = load_config(allow_provisional=False)
    man = {x["id"]: x for x in map(json.loads, open(ROOT / "data" / "dev_manifest.jsonl"))}
    rows = [json.loads(x) for x in open(a.cands)]
    lm = LetterModel(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{a.model}.jsonl"
    done = {json.loads(x)["key"] for x in open(f)} if f.exists() else set()
    budget = Budget("dev")
    print(f"{a.model} loaded, letter ids {lm.ids}, merge_system {lm.merge_system}", flush=True)
    with open(f, "a") as fo:
        for r in rows:
            if r["status"] != "pass":
                continue
            it = man[r["id"]]
            texts = [("S0", it["question"])] + [(f"cand{c['cand']}", c["pert_question"]) for c in r["candidates"] if c["valid"]]
            for tag, text in texts:
                key = f"{r['id']}/{tag}"
                if key in done:
                    continue
                t0 = time.time()
                s = lm.letters(c0_messages(text, it["options"], cfg))
                budget.charge(time.time() - t0, 1.0, note=f"v5_panel/{a.model}")
                fo.write(json.dumps({"key": key, "id": r["id"], "tag": tag, "model": a.model, **s,
                                     "logits4_sha256": logits4_sha(s["logits4"])}) + "\n")
    print(f"wrote {f}", flush=True)


def cmd_select(a):
    man = {x["id"]: x for x in map(json.loads, open(ROOT / "data" / "dev_manifest.jsonl"))}
    rows = [json.loads(x) for x in open(a.cands)]
    S = {}
    for m in PANEL:
        for x in map(json.loads, open(Path(a.scores) / f"{m}.jsonl")):
            S[(m, x["id"], x["tag"])] = x["p4"]
    out_items, report = [], {"n_items_called": 0, "n_with_valid": 0, "selected": [], "adopted": 0}
    for r in rows:
        if r["status"] == "excluded" and not r["candidates"] and "rule" in (r.get("exclude_reason") or ""):
            continue
        report["n_items_called"] += 1
        if r["status"] != "pass":
            continue
        report["n_with_valid"] += 1
        it = man[r["id"]]
        g = it["answer_idx"]
        cs = []
        for c in r["candidates"]:
            if not c["valid"]:
                continue
            d = {m: S[(m, r["id"], "S0")][g] - S[(m, r["id"], f"cand{c['cand']}")][g] for m in PANEL}
            cs.append({"cand": c["cand"], "d": d, "score": float(np.mean(list(d.values()))), "n_pos": sum(v > 0 for v in d.values())})
        best = max(cs, key=lambda c: (c["score"], -c["cand"]))
        adopt = best["score"] >= THRESH and best["n_pos"] >= MIN_POS
        report["selected"].append({"id": r["id"], **best, "n_valid": len(cs), "adopted": adopt})
        if adopt:
            c = next(x for x in r["candidates"] if x["cand"] == best["cand"])
            out_items.append({"id": r["id"], "hash": it["hash"], "question": it["question"], "options": it["options"],
                              "answer_idx": g, "pert_question": c["pert_question"],
                              "removed_facts": [f["text"] for f in c["facts"]], "removed_types": [f["type"] for f in c["facts"]],
                              "removed_fact": " ".join(f["text"].strip() for f in c["facts"]),
                              "removed_type": c["facts"][0]["type"], "cand": best["cand"], "panel_score": best["score"],
                              "panel_d": best["d"]})
    report["adopted"] = len(out_items)
    sc = [s["score"] for s in report["selected"]]
    pos = [s["n_pos"] for s in report["selected"]]
    report["survival_by_threshold"] = {f"{t:.2f}": {f"n_pos>={k}": sum(s >= t and p >= k for s, p in zip(sc, pos)) for k in (1, 2, 3, 4)}
                                       for t in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)}
    report["score_quantiles"] = dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"),
                                         map(float, np.quantile(sc, [0, .1, .25, .5, .75, .9, 1])))) if sc else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "devk.jsonl", "w") as f:
        for x in out_items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    (out / "selection_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "selected"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "select"])
    ap.add_argument("--model")
    ap.add_argument("--cands", required=True)
    ap.add_argument("--scores")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_score(a) if a.cmd == "score" else cmd_select(a)


if __name__ == "__main__":
    main()
