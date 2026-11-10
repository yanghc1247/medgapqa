"""D30dev DEV-K2 exploratory run wrapper (new file). Calls the same code paths as stage3 (D26b) (src/medgapqa/pipeline_k.cmd_arms, src/medgapqa/c4_ref_k.cmd_run, agent.run_episode C0)
on the DEV-K2 items (runs/v5/devk2/devk2.jsonl) with states S1K and S2K only (S0 is not run). Settings, prompts and simulator v10 unchanged.
CUDA_VISIBLE_DEVICES=4 python scripts/run_devk2_explore.py arms --model qwen2.5-7b --sim-ports 8704,8705,8706,8707
CUDA_VISIBLE_DEVICES=4 python scripts/run_devk2_explore.py c4   --model qwen2.5-7b --sim-ports ...
CUDA_VISIBLE_DEVICES=4 python scripts/run_devk2_explore.py c0s2k --model qwen2.5-7b
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/medgapqa"))
import c4_ref_k  # noqa: E402
import pipeline_k  # noqa: E402
ITEMS = ROOT / "runs" / "v5" / "devk2" / "devk2.jsonl"
OUT = ROOT / "runs" / "v5" / "eval_devk2"


def c0s2k(model):
    from agent import load_config, run_episode
    from budget import Budget
    cfg = load_config(allow_provisional=False)
    m = pipeline_k.EvalModel(model)
    ok, how = m.verify()
    if not ok:
        sys.exit(f"instance check failed for {model} ({how})")
    d = OUT / model / "C0" / "S2K"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "trajectories.jsonl"
    if f.exists():
        sys.exit(f"{f} exists")
    budget, inst = Budget("dev"), pipeline_k.inst_id(model)
    with open(f, "w") as fo:
        for it in map(json.loads, open(ITEMS)):
            t0 = time.time()
            tr = run_episode({k: it[k] for k in ("id", "hash", "question", "pert_question", "removed_fact", "removed_type", "options")}, "S2", "C0", m, cfg)
            tr["seconds"], tr["instance"], tr["state"] = round(time.time() - t0, 2), inst, "S2K"
            budget.charge(tr["seconds"], 1.0, note=f"v5_devk2_c0/{model}/S2K")
            fo.write(json.dumps(tr, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["arms", "c4", "c0s2k"]); ap.add_argument("--model", required=True)
    ap.add_argument("--sim-ports", default="8707"); a = ap.parse_args()
    pipeline_k.ITEMS = ITEMS
    pipeline_k.STATES = {"S1K": "s1k", "S2K": "s2k"}
    c4_ref_k.ITEMS, c4_ref_k.OUT = ITEMS, OUT
    c4_ref_k.STATE = {"S1K": "S1", "S2K": "S2"}
    if a.cmd == "arms":
        pipeline_k.cmd_arms(argparse.Namespace(model=a.model, n=None, sim_ports=a.sim_ports, out=str(OUT), only_det=False))
    elif a.cmd == "c4":
        c4_ref_k.cmd_run(argparse.Namespace(model=a.model, sim_ports=a.sim_ports))
    else:
        c0s2k(a.model)
