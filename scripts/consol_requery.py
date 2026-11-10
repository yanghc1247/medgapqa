"""D23 consolidated re-query (exploratory arm, post hoc). Frozen files are only imported (guard3 must pass). Gold labels are not read.

For each trajectory with at least one tool reply (observation), the information obtained is gathered and asked once more in C0 format, and the answer is taken from A–D restricted logits.
- system: agent.system_prompt("C0", cfg)
- user: case_text(item in that state, options) + "\\n\\nInformation obtained so far:\\n" + blocks + "\\n\\nYour action:"
  blocks = ASK and SEARCH turns with an observation, in time order: ASK "Question asked: {content}\\n{observation}",
  SEARCH "Search query: {content}\\n{observation}", separated by "\\n\\n". Not filtered by simulator label.
  With no blocks the whole information part is dropped and the input equals the first C0 turn (pre-check a).
- decoding: forced_answer.Scorer (FORCED_PREFILL, maximum over A–D restricted, MedGemma merged via run_agent.template_messages).
- stored: letter, logits4, logits4 sha, n_in, prompt sha, p4_sum, argmax_is_letter only. Prompt and output text are not stored (StatPearls).

python scripts/consol_requery.py check-render                       # pre-check a (CPU)
CUDA_VISIBLE_DEVICES=4 python scripts/consol_requery.py run --model qwen2.5-7b [--only-det] --out runs/consol_requery_v043
python scripts/consol_requery.py det --a <det.jsonl> --b <det.jsonl>  # pre-check b
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/medgapqa"))
from agent import FIRST_TRAILER, _sha, case_text, load_config, system_prompt  # noqa: E402
from forced_answer import DET_N, MODELS, RUNS, SEED, Scorer, guard3, load_items, load_trs, logits4_sha  # noqa: E402

HEADER = "\n\nInformation obtained so far:\n"
CELLS = {"v7": [(c, s) for c in ("C2", "C3", "C4") for s in ("S0", "S1", "S2")], "v9": [("C4", "S1")]}


def blocks(t):
    out = []
    for x in t["turns"]:
        if x.get("action") in ("ASK", "SEARCH") and "observation" in x:
            lab = "Question asked" if x["action"] == "ASK" else "Search query"
            out.append(f"{lab}: {x['content']}\n{x['observation']}")
    return out


def consol_messages(item, state, bl, cfg):
    visible = item["question"] if state == "S0" else item["pert_question"]
    user = case_text(visible, item["options"]) + (HEADER + "\n\n".join(bl) if bl else "") + "\n\n" + FIRST_TRAILER
    return [{"role": "system", "content": system_prompt("C0", cfg)}, {"role": "user", "content": user}]


def targets(model, items, cfg):
    out = []
    for run, cells in CELLS.items():
        for c, s in cells:
            for t in load_trs(run, model, c, s):
                bl = blocks(t)
                if bl:
                    out.append(((run, "consol", c, s, t["id"]), consol_messages(items[t["hash"]], s, bl, cfg),
                                {"hash": t["hash"], "n_blocks": len(bl), "outcome": t["outcome"]}))
    return out


def cmd_check_render(a):
    """Pre-check a: does the rendering with empty blocks equal the recorded input_sha of the first C0 turn (all C0 S0 and S1 per model)?"""
    cfg, items, rep = load_config(allow_provisional=False), load_items(), {}
    for m in MODELS:
        r = {}
        for s in ("S0", "S1"):
            trs = load_trs("v7", m, "C0", s)
            ok = sum(_sha(consol_messages(items[t["hash"]], s, [], cfg)) == t["turns"][0]["input_sha"] for t in trs)
            r[s] = {"n": len(trs), "match": ok}
        tg = targets(m, items, cfg)
        r["targets"] = {f"{k[0]}/{k[2]}/{k[3]}": 0 for k, _, _ in tg}
        for k, _, _ in tg:
            r["targets"][f"{k[0]}/{k[2]}/{k[3]}"] += 1
        r["targets_total"] = len(tg)
        rep[m] = r
    print(json.dumps(rep, indent=1))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "render_check.json").write_text(json.dumps(rep, indent=1))
    ok = all(v[s]["match"] == v[s]["n"] == 1014 for v in rep.values() for s in ("S0", "S1"))
    return 0 if ok else 1


def det_subset(tg):
    return [tg[int(i)] for i in np.random.default_rng(SEED).permutation(len(tg))[:DET_N]]


def cmd_run(a):
    from budget import Budget
    from instances import verify_instance
    from run_agent import REF_HASHES, consistency_probes
    problems = guard3()
    if problems:
        sys.exit("consol_requery refused: " + "; ".join(problems))
    cfg, items = load_config(allow_provisional=False), load_items()
    tg = targets(a.model, items, cfg)
    sc = Scorer(a.model)
    ok, got = verify_instance(sc.hf.chat, consistency_probes(), json.load(open(REF_HASHES))[a.model])
    if not ok:
        sys.exit(f"instance verification failed for {a.model}: {got}")
    print(f"instance verified: {a.model}, letter ids {sc.ids}, targets {len(tg)}", flush=True)
    inst = f"{os.uname().nodename}:gpu{os.environ.get('CUDA_VISIBLE_DEVICES')}:pid{os.getpid()}:{a.model}"
    out = Path(a.out) / a.model
    out.mkdir(parents=True, exist_ok=True)
    work = det_subset(tg) if a.only_det else tg
    f = out / ("det.jsonl" if a.only_det else "consol.jsonl")
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
            budget.charge(time.time() - t0, 1.0, note=f"consol_requery/{a.model}/{k[0]}/{k[2]}/{k[3]}")
            fo.write(json.dumps({"key": list(k), "model": a.model, **meta, "letter": r["letter"], "logits4": r["logits4"],
                                 "logits4_sha256": logits4_sha(r["logits4"]), "n_in": r["n_in"], "prompt_sha256": r["prompt_sha256"],
                                 "p4_sum": r["p4_sum"], "argmax_is_letter": r["argmax_is_letter"], "instance": inst}) + "\n")
            fo.flush()
    print(f"wrote {f}", flush=True)


def cmd_det(a):
    x = {tuple(r["key"]): r for r in map(json.loads, open(a.a))}
    y = {tuple(r["key"]): r for r in map(json.loads, open(a.b))}
    same = [k for k in x if k in y and x[k]["letter"] == y[k]["letter"] and x[k]["logits4_sha256"] == y[k]["logits4_sha256"]]
    rep = {"n_a": len(x), "n_b": len(y), "n_same": len(same), "ok": len(x) == len(y) == len(same) == DET_N,
           "instances": sorted({r["instance"] for r in list(x.values()) + list(y.values())})}
    print(json.dumps(rep))
    return 0 if rep["ok"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check-render", "run", "det"])
    ap.add_argument("--model")
    ap.add_argument("--out", default=str(ROOT / "runs" / "consol_requery_v043"))
    ap.add_argument("--only-det", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--a")
    ap.add_argument("--b")
    a = ap.parse_args()
    if a.cmd == "check-render":
        sys.exit(cmd_check_render(a))
    if a.cmd == "det":
        sys.exit(cmd_det(a))
    cmd_run(a)


if __name__ == "__main__":
    main()
