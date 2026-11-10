"""D18 forced answer (for FCR). Appends a fixed sentence to trajectories that ended in abstain or unresolved and takes the maximum of the next-token logits restricted to A, B, C, D as the answer.

No generation (one forward pass), no re-abstention, deterministic. Gold labels are not read (scoring is in src/medgapqa/analysis_fcr.py). Frozen files are only read.
Input reconstruction: the input of the final turn (conversation so far + that turn's instruction) is verified against the recorded input_sha → the last output (retry_raw if there was a re-request)
as assistant → if the last turn has an observation, observation + "\\n\\n" + FIXED, otherwise FIXED, as user, followed by agent.FORCED_PREFILL (D18a).
answered_check for descriptive statistics: the same method on the recorded input of the last turn of episodes that ended with ANSWER (sha verified) + FORCED_PREFILL.

python src/medgapqa/forced_answer.py check                                 # verify all reconstruction sha (CPU)
CUDA_VISIBLE_DEVICES=4 python src/medgapqa/forced_answer.py run --model qwen2.5-7b [--only-det] --out runs/forced_answer_v043
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/medgapqa"))
from abstain_probe import history_before_abstain  # noqa: E402
from agent import FORCED_PREFILL, _sha, _with_trailer, load_config  # noqa: E402

FIXED = "Answer now with exactly one option letter: A, B, C, or D."
FIXED_SHA256 = "ed9e81b19d5b9fdbe76ea20ea43c983496d99ea62dee701a85ab4e66ed038d72"
LETTERS = ["A", "B", "C", "D"]
TAG = "freeze-v0.4.3"
MANIFEST3_REL = "metadata/freeze3_manifest.json"
MODELS = ["qwen2.5-7b", "olmo2-7b", "meerkat-8b", "medgemma-4b"]
VIEW = ROOT / "runs" / "analysis_v042" / "view"
RUNS = {"v7": VIEW / "runs", "v9": VIEW / "runs_v9"}
CELLS = {"v7": [(c, s) for c in ("C0", "C1") for s in ("S0", "S1")] + [(c, s) for c in ("C2", "C3", "C4") for s in ("S0", "S1", "S2")],
         "v9": [("C4", "S1")]}
CHECK_CELLS = [("C0", "S1"), ("C4", "S1")]
CHECK_N = 200
DET_N = 20
SEED = 20261016


def guard3(root=ROOT, tag=TAG, tag_message=None):
    """Run conditions (same approach as D13): (i) freeze-v0.4.3 tag (ii) manifest file sha equals the value in the tag message
    (iii) frozen file hashes in the manifest match (iv) existing EVAL guard (freeze-v0.4.2) []. List of problems (empty list = allowed)."""
    from run_agent import _git_tag_message, eval_guard
    tag_message = tag_message or _git_tag_message
    root = Path(root)
    msg = tag_message(tag)
    if msg is None:
        return [f"tag {tag} missing"]
    problems = []
    man_path = root / MANIFEST3_REL
    m = re.search(re.escape(MANIFEST3_REL) + r" sha256 ([0-9a-f]{64})", msg)
    if not m:
        problems.append(f"tag {tag} message has no {MANIFEST3_REL} sha256")
    elif hashlib.sha256(man_path.read_bytes()).hexdigest() != m.group(1):
        problems.append(f"{MANIFEST3_REL} hash differs from tag {tag}")
    man = json.loads(man_path.read_text())
    for f, h in {**man.get("code_hashes", {}), **man.get("data_hashes", {})}.items():
        p = root / f
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            problems.append(f"frozen file changed or missing: {f}")
    if hashlib.sha256(FIXED.encode()).hexdigest() != FIXED_SHA256 or man.get("fixed_sha256") != FIXED_SHA256:
        problems.append("fixed phrase sha mismatch")
    problems += [f"eval_guard: {p}" for p in eval_guard()]
    return problems


def last_output(rec):
    return rec.get("retry_raw", rec["raw"])


def turn_input(t, item, cfg):
    """Input of the final turn and whether it matches its recorded sha. (hist: record up to just before that turn, ok)"""
    hist, trailer = history_before_abstain(t, item, t["cond"], cfg)
    return hist, _sha(_with_trailer(hist, trailer)) == t["turns"][-1]["input_sha"]


def forced_messages(t, item, cfg):
    hist, ok = turn_input(t, item, cfg)
    last = t["turns"][-1]
    tail = last["observation"] + "\n\n" + FIXED if "observation" in last else FIXED
    return hist + [{"role": "assistant", "content": last_output(last)}, {"role": "user", "content": tail}], ok


def check_messages(t, item, cfg):
    hist, ok = turn_input(t, item, cfg)
    _, trailer = history_before_abstain(t, item, t["cond"], cfg)
    return _with_trailer(hist, trailer), ok


def load_trs(run, model, cond, state):
    p = RUNS[run] / model / cond / state / "trajectories.jsonl"
    return [json.loads(x) for x in open(p)] if p.exists() else []


def targets(model, items, cfg):
    """[(key, msgs, meta)]. key = (run, kind, cond, state, id). kind: forced / answered_check."""
    out = []
    for run, cells in CELLS.items():
        for c, s in cells:
            for t in load_trs(run, model, c, s):
                if t["outcome"] in ("abstain", "unresolved"):
                    msgs, ok = forced_messages(t, items[t["hash"]], cfg)
                    out.append(((run, "forced", c, s, t["id"]), msgs, {"hash": t["hash"], "outcome": t["outcome"], "sha_ok": ok,
                                                                        "end_reason": t["end_reason"]}))
    for c, s in CHECK_CELLS:
        ans = [t for t in load_trs("v7", model, c, s) if t["outcome"] == "answer"]
        pick = np.random.default_rng(SEED).choice(len(ans), size=min(CHECK_N, len(ans)), replace=False)
        for k in sorted(int(i) for i in pick):
            t = ans[k]
            msgs, ok = check_messages(t, items[t["hash"]], cfg)
            out.append((("v7", "answered_check", c, s, t["id"]), msgs, {"hash": t["hash"], "outcome": "answer", "sha_ok": ok,
                                                                         "answer": t["answer"]}))
    return out


def det_subset(tg):
    """20 determinism checks: chosen by seed among forced targets (filled with answered_check if fewer than 20 targets)."""
    f = [x for x in tg if x[0][1] == "forced"]
    rest = [x for x in tg if x[0][1] != "forced"]
    rng = np.random.default_rng(SEED)
    pick = [f[int(i)] for i in rng.permutation(len(f))[:DET_N]]
    if len(pick) < DET_N:
        pick += [rest[int(i)] for i in rng.permutation(len(rest))[:DET_N - len(pick)]]
    return pick


def letter_ids(tok, template):
    """Checks that each of A–D after FORCED_PREFILL is a single token and returns those ids."""
    base = tok(template, add_special_tokens=False)["input_ids"]
    ids = []
    for L in LETTERS:
        x = tok(template + L, add_special_tokens=False)["input_ids"]
        assert x[:len(base)] == base and len(x) == len(base) + 1, f"letter {L} is not a single token after prefill"
        ids.append(x[-1])
    assert len(set(ids)) == 4
    return ids


def restricted(logits, ids):
    """logits: full-vocabulary logits (1-D float array). Restricted A–D maximum, the four logits, sum of A–D probabilities, and whether the unrestricted maximum is in A–D."""
    lg = np.asarray(logits, dtype=np.float64)
    four = lg[ids]
    p = np.exp(lg - lg.max())
    p /= p.sum()
    return {"letter": LETTERS[int(np.argmax(four))], "logits4": [float(x) for x in four],
            "p4_sum": float(p[ids].sum()), "argmax_is_letter": int(np.argmax(lg)) in ids}


def logits4_sha(logits4):
    return hashlib.sha256(np.asarray(logits4, dtype=np.float32).tobytes()).hexdigest()


class Scorer:
    def __init__(self, name):
        import torch
        from run_agent import HFModel, template_messages
        self.torch, self.tm, self.name = torch, template_messages, name
        self.hf = HFModel(name)
        tpl = self.hf.tok.apply_chat_template(template_messages(name, [{"role": "user", "content": "x"}]),
                                              add_generation_prompt=True, tokenize=False) + FORCED_PREFILL
        self.ids = letter_ids(self.hf.tok, tpl)

    def __call__(self, msgs):
        prompt = self.hf.tok.apply_chat_template(self.tm(self.name, msgs), add_generation_prompt=True, tokenize=False) + FORCED_PREFILL
        x = self.hf.tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.hf.model.device)
        with self.torch.no_grad():
            lg = self.hf.model(x).logits[0, -1].float().cpu().numpy()
        return {**restricted(lg, self.ids), "n_in": int(x.shape[1]), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def load_items():
    return {x["hash"]: x for x in map(json.loads, open(ROOT / "data" / "perturbed_eval.jsonl"))}


def cmd_check(a):
    cfg, items, rep = load_config(allow_provisional=False), load_items(), {}
    for m in MODELS:
        tg = targets(m, items, cfg)
        bad = [list(k) for k, _, meta in tg if not meta["sha_ok"]]
        rep[m] = {"forced": sum(k[1] == "forced" for k, _, _ in tg), "answered_check": sum(k[1] == "answered_check" for k, _, _ in tg),
                  "sha_mismatch": len(bad), "examples": bad[:5]}
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "reconstruct_check.json").write_text(json.dumps(rep, indent=1))
    return 0 if all(v["sha_mismatch"] == 0 for v in rep.values()) else 1


def cmd_run(a):
    from budget import Budget
    from instances import verify_instance
    from run_agent import REF_HASHES, consistency_probes
    problems = guard3()
    if problems:
        sys.exit("forced_answer refused: " + "; ".join(problems))
    cfg, items = load_config(allow_provisional=False), load_items()
    tg = targets(a.model, items, cfg)
    if any(not meta["sha_ok"] for _, _, meta in tg):
        sys.exit("reconstruction sha mismatch (run check first)")
    sc = Scorer(a.model)
    ok, got = verify_instance(sc.hf.chat, consistency_probes(), json.load(open(REF_HASHES))[a.model])
    if not ok:
        sys.exit(f"instance verification failed for {a.model}: {got}")
    print(f"instance verified: {a.model}, letter ids {sc.ids}", flush=True)
    import os
    inst = f"{os.uname().nodename}:gpu{os.environ.get('CUDA_VISIBLE_DEVICES')}:pid{os.getpid()}:{a.model}"
    out = Path(a.out) / a.model
    out.mkdir(parents=True, exist_ok=True)
    work = det_subset(tg) if a.only_det else tg
    f = out / ("det.jsonl" if a.only_det else "forced.jsonl")
    done = {tuple(r["key"]) for r in map(json.loads, open(f))} if f.exists() else set()
    if f.exists() and not a.resume:
        sys.exit(f"{f} exists (no overwrite; use --resume)")
    budget = Budget("eval")
    with open(f, "a") as fo:
        for k, msgs, meta in work:
            if tuple(k) in done:
                continue
            budget.check()
            t0 = time.time()
            r = sc(msgs)
            budget.charge(time.time() - t0, 1.0, note=f"forced_answer/{a.model}/{k[0]}/{k[2]}/{k[3]}")
            fo.write(json.dumps({"key": list(k), "model": a.model, **{x: meta[x] for x in meta if x != "sha_ok"},
                                 "input_sha": _sha(msgs), **r, "logits4_sha256": logits4_sha(r["logits4"]), "instance": inst}) + "\n")
            fo.flush()
    print(f"wrote {f}", flush=True)


def cmd_det(a):
    """Compares two det files from the same 20 cases run on two GPUs: passes only if letters and the sha of the four logits all match."""
    x = {tuple(r["key"]): r for r in map(json.loads, open(a.a))}
    y = {tuple(r["key"]): r for r in map(json.loads, open(a.b))}
    same = [k for k in x if k in y and x[k]["letter"] == y[k]["letter"] and x[k]["logits4_sha256"] == y[k]["logits4_sha256"]]
    rep = {"n_a": len(x), "n_b": len(y), "n_same": len(same), "ok": len(x) == len(y) == len(same) == DET_N,
           "instances": sorted({r["instance"] for r in list(x.values()) + list(y.values())})}
    print(json.dumps(rep))
    return 0 if rep["ok"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run", "det"])
    ap.add_argument("--model")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only-det", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--a")
    ap.add_argument("--b")
    a = ap.parse_args()
    if a.cmd == "check":
        sys.exit(cmd_check(a))
    if a.cmd == "det":
        sys.exit(cmd_det(a))
    cmd_run(a)


if __name__ == "__main__":
    main()
