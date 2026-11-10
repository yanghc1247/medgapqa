"""D122: NP probe AUROC difference within each fine-tune/base pair (Meerkat2 - Qwen3), post hoc, exploratory, no multiplicity correction; no GPU.
Reuses the scoring of p2_analysis (score logit(No) - logit(Yes), S1 positive against S0 negative, PA.auc with ties counted half) and the shared resampling
matrix (PA.Boot: B 10,000, seed 20261016, 736 items); in every resample both models' AUROC are computed on the same rows; 95% two-sided percentiles.
Does not modify p2_analysis.py or any existing JSON.
python scripts/p2/p2_probe_pairs.py --root runs/p2/eval --items runs/v5/evalk/evalk.jsonl --out runs/p2/analysis/p2_probe_pairs.json --approved
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts" / "p2")]
import p2_analysis as PA  # noqa: E402
import p2_probe_strata as PP  # noqa: E402

META = {"analysis": "D122 NP probe AUROC difference within fine-tune/base pairs", "status": "post hoc, exploratory, no multiplicity correction",
        "score": PP.META["score"], "bootstrap": "shared resampling matrix of p2_analysis.Boot (B 10000, seed 20261016, 736 items); per resample, both AUROC on the same rows; 95% percentile",
        "check": "point AUROC of each model reproduces p2_analysis np_probe AUROC"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--approved", action="store_true"); a = ap.parse_args()
    if Path(a.root).resolve() == (ROOT / "runs" / "p2" / "eval").resolve() and not a.approved:
        sys.exit("refusing to analyze EVAL outputs without --approved")
    order = [x["hash"] for x in PA.jl(a.items)]
    bt = PA.Boot(len(order)); ones = np.ones(len(order))
    J = json.load(open(ROOT / "runs" / "p2" / "analysis" / "p2_analysis.json"))
    rep = {"meta": META, "n_items": len(order), "B": PA.B, "seed": PA.SEED, "pairs": {}}
    for ft, base in PA.PAIRS:
        res = {}
        for m in (ft, base):
            s0, s1 = PP.scores(Path(a.root), m, order)
            pt = PA.auc(s1, s0); assert abs(pt - J["models"][m]["np_probe"]["AUROC"]) < 1e-12, m
            res[m] = (pt, PP.boot_auc(s0, s1, ones, bt))
        d = res[ft][1] - res[base][1]
        rep["pairs"][f"{ft}_vs_{base}"] = {"pair": [ft, base], "auroc": {m: res[m][0] for m in (ft, base)},
                                          "diff": {"value": res[ft][0] - res[base][0], "ci95": PA.pct(d, 0.95)}}
    rep["meta"] = dict(rep["meta"], p2_analysis_py_sha256=sha(ROOT / "scripts" / "p2" / "p2_analysis.py"), script_sha256=sha(__file__), items_sha256=sha(a.items))
    Path(a.out).write_text(json.dumps(rep, indent=1))
    for k, v in rep["pairs"].items():
        print(k, {m: round(x, 3) for m, x in v["auroc"].items()}, round(v["diff"]["value"], 3), [round(x, 3) for x in v["diff"]["ci95"]])


if __name__ == "__main__":
    main()
