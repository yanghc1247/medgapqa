"""D89 item 2: recovery ratio by stratum (post hoc, exploratory, no multiplicity correction; added after seeing the stratum differences in C0 cost in the D88 table). Does not modify p2_analysis.py, p2_strata.py or their JSON.
Strata, scoring and sensitivity as in p2_strata (D88). Ratios (stratum s, item mask w):
  recovery_<P> = sum_w Delta(policy - C0, S1) / sum_w C0 cost (S0 - S1),   s1s2_over_cost_<P> = sum_w (policy S1 - S2) / sum_w C0 cost
The main mean is the ratio of the item vectors of the model mean (mean Delta / mean cost; not the mean of per-model ratios). The two large models per model.
Intervals: ratio of the stratum-item sums in every resample of the shared matrix of p2_analysis.Boot (B 10,000, seed 20261016, 736 items), 95% percentiles.
Resamples whose denominator (stratum cost sum) is <= 0 are dropped and counted (the 95% interval of the denominator is also given; unstable = true if it contains 0).
python scripts/p2/p2_strata_recovery.py --root runs/p2/eval --items runs/v5/evalk/evalk.jsonl --out runs/p2/analysis/p2_strata_recovery.json --approved
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts" / "p2")]
import p2_analysis as PA  # noqa: E402
import p2_strata as PS  # noqa: E402

META = {"analysis": "D89 stratum recovery ratios", "status": "post hoc, exploratory, no multiplicity correction; added after seeing the stratum differences in C0 cost (D88 tables)",
        "labels": PS.META["labels"], "sensitivity": PS.META["sensitivity"],
        "ratios": {"recovery_<P>": "sum over stratum items of (policy S1 - C0 S1) / sum of (C0 S0 - C0 S1)",
                   "s1s2_over_cost_<P>": "sum over stratum items of (policy S1 - policy S2) / sum of (C0 S0 - C0 S1)"},
        "main_mean": "ratio of the model-mean item vectors over the 9-model main pool (mean delta / mean cost), not the mean of per-model ratios",
        "bootstrap": "shared resampling matrix of p2_analysis.Boot (B 10000, seed 20261016, 736 items); per resample, ratio of stratum sums; replicates with denominator <= 0 dropped and counted; 95% percentile; unstable = denominator 95% CI includes 0"}


def ratio(num, den, mask, bt):
    n, d = (num * mask)[bt.idx].sum(1), (den * mask)[bt.idx].sum(1)
    dm = d / mask[bt.idx].sum(1).clip(min=1)  # mean stratum cost (per resample)
    ok = d > 0
    dci = PA.pct(dm, 0.95)
    return {"value": float((num * mask).sum() / (den * mask).sum()) if (den * mask).sum() > 0 else None, "ci95": PA.pct(n[ok] / d[ok], 0.95) if ok.any() else None,
            "den_cost_mean": float((den * mask).sum() / mask.sum()), "den_cost_ci95": dci, "n_replicates_den_nonpositive": int((~ok).sum()), "unstable": bool(dci[0] <= 0)}


def run(root, items_path, L, art):
    its = PA.jl(items_path)
    order = [x["hash"] for x in its]; gold = {x["hash"]: x["answer_idx"] for x in its}
    present = [m for m in PA.MAIN + PA.LARGE if PA.available(root, m)]
    main = [m for m in PA.MAIN if m in present]; large = [m for m in PA.LARGE if m in present]
    bt = PA.Boot(len(order))
    V = {m: PS.vectors(PA.Model(root, m, order), gold) for m in present}
    groups = {"main_mean": {k: np.mean([V[m][k] for m in main], axis=0) for k in V[main[0]]}} if main else {}
    groups.update({m: V[m] for m in large})
    rep = {"meta": META, "n_items": len(order), "main_pool": main, "large": large, "main_mean_missing": [m for m in PA.MAIN if m not in present], "excluded": PA.EXCLUDED, "tables": {}}
    for sens in ("all", "no_artifact"):
        tab = {}
        for st in PS.STRATA:
            mask = ((L == st) & (~art if sens == "no_artifact" else True)).astype(float)
            row = {"n": int(mask.sum())}
            for g, v in groups.items():
                row[g] = {}
                for pol in PA.POLICIES:
                    row[g][f"recovery_{pol}"] = ratio(v[f"delta_{pol}_S1"], v["C0_cost_S0_minus_S1"], mask, bt)
                    row[g][f"s1s2_over_cost_{pol}"] = ratio(v[f"s1_minus_s2_{pol}"], v["C0_cost_S0_minus_S1"], mask, bt)
            tab[st] = row
        rep["tables"][sens] = tab
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--approved", action="store_true"); a = ap.parse_args()
    if Path(a.root).resolve() == (ROOT / "runs" / "p2" / "eval").resolve() and not a.approved:
        sys.exit("refusing to analyze EVAL outputs without --approved")
    L, art = PS.load_labels(PA.jl(a.items))
    rep = run(a.root, a.items, L, art)
    rep["meta"] |= {"p2_analysis_py_sha256": PS.sha(ROOT / "scripts" / "p2" / "p2_analysis.py"), "p2_strata_py_sha256": PS.sha(ROOT / "scripts" / "p2" / "p2_strata.py"),
                    "script_sha256": PS.sha(__file__), "items_sha256": PS.sha(a.items)}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {a.out}: groups {rep['main_pool'] and ['main_mean']} + {rep['large']}")


if __name__ == "__main__":
    main()
