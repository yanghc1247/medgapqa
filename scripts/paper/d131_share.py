"""D131-r3: share of the both-facts gain reached by the targeted ask pipeline, as for the Instant 'about 20%' (PtTargetedShareOfBothFactsPct =
main-mean targeted Delta S1 / mean over models of (O2 accuracy - C0 S1 accuracy)), for the 7 hybrid models in Instant and in Think mode, and for the
9-model Instant main pool as a check against the paper macro. Interval: ratio of the two item-level means recomputed in every resample of the shared
matrix (p2_analysis.Boot, B 10,000, seed 20261016), 95% percentile. REUSED_PATHS/REUSED emptied as in scripts/p2/analysis_vllm.py (D130-m).
python scripts/paper/d131_share.py
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts" / "p2")]
import p2_analysis as PA  # noqa: E402
PA.REUSED_PATHS = {}; PA.REUSED = set()
AV = ROOT / "runs" / "p2" / "analysis_vllm"; OUT = ROOT / "runs" / "p2" / "analysis_vllm_paper" / "share_both_facts.json"
THINK7 = ["qwen3-8b", "qwen3-14b", "meerkat2-14b", "qwen3-30b-a3b", "meerkat2-30b-a3b", "qwen3-32b", "meerkat2-32b"]
MAIN9 = PA.MAIN


def vectors(root, models, order, gold):
    num, den = [], []
    for m in models:
        M = PA.Model(str(root), m, order)
        c1 = M.correct("C0", PA.ST["S1"], gold)
        num.append(M.correct("P6", PA.ST["S1"], gold) - c1); den.append(M.correct("O:O2", None, gold) - c1)
    return np.mean(num, axis=0), np.mean(den, axis=0)


def main():
    its = PA.jl(str(ROOT / "runs" / "v5" / "evalk" / "evalk.jsonl")); order = [x["hash"] for x in its]; gold = {x["hash"]: x["answer_idx"] for x in its}
    bt = PA.Boot(len(order)); out = {"meta": {"analysis": __doc__.split("\n")[0], "B": PA.B, "seed": PA.SEED}}
    for key, root, models in (("think7_think", AV / "think" / "root", THINK7), ("think7_instant", AV / "instant" / "root", THINK7), ("main9_instant", AV / "instant" / "root", MAIN9)):
        n, d = vectors(root, models, order, gold)
        rs = np.array([n[i].mean() / d[i].mean() for i in bt.idx])
        out[key] = {"models": models, "delta_P6_S1": float(n.mean()), "both_facts_gain": float(d.mean()), "share": float(n.mean() / d.mean()), "share_ci95": PA.pct(rs, 0.95)}
        print(key, round(n.mean(), 4), round(d.mean(), 4), round(n.mean() / d.mean(), 4), [round(x, 4) for x in out[key]["share_ci95"]])
    # paired difference of the shares (Think - Instant, same 7 models, same resamples)
    nt, dt = vectors(AV / "think" / "root", THINK7, order, gold); ni, di = vectors(AV / "instant" / "root", THINK7, order, gold)
    diff = np.array([nt[i].mean() / dt[i].mean() - ni[i].mean() / di[i].mean() for i in bt.idx])
    out["think7_diff"] = {"share": out["think7_think"]["share"] - out["think7_instant"]["share"], "share_ci95": PA.pct(diff, 0.95)}
    print("diff", round(out["think7_diff"]["share"], 4), [round(x, 4) for x in out["think7_diff"]["share_ci95"]])
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
