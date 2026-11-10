"""D90: correlation across models (post hoc, exploratory, no multiplicity correction; added after seeing Tables 3 and 4; no GPU). Does not modify p2_analysis.py or its JSON; only reuses its reading and scoring functions.
Model-level Spearman rank correlation between a per-model acquisition measure (share of items receiving >= 1 fact, S1K) and an accuracy measure.
  Main analysis: targeted ask (P6) share receiving >= 1 fact (S1) versus Delta (S1, P6 - C0), over the completed main-pool models plus the completed large models.
  Secondary: (a) main pool only (large excluded), (b) without the reused model (p2_analysis.REUSED), (c) P6 S1 - S2 instead of Delta, (d) forced inquiry (C4f) share receiving >= 1 fact versus C4f S1 - S2
  (for reference also C4f >= 1 receipt versus Delta (S1, C4f - C0)).
Scoring and receipt definitions as in p2_analysis (PA.Model.correct, PA.recv; >= 1 receipt = items that received at least one removed fact).
Intervals: items are resampled with the shared matrix of p2_analysis.Boot (B 10,000, seed 20261016, 736 items, item-clustered, paired); per-model values and Spearman are recomputed in every resample; 95% percentiles.
Ranks are average ranks for ties. A resample in which one side has all ranks equal (correlation undefined) is dropped and counted.
python scripts/p2/p2_modelcorr.py --root runs/p2/eval --items runs/v5/evalk/evalk.jsonl --out runs/p2/analysis/p2_modelcorr.json --approved
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts" / "p2")]
import p2_analysis as PA  # noqa: E402
import p2_strata as PS  # noqa: E402

PAIRS = {"P6_recv_ge1_vs_delta_P6_S1": ("recv_ge1_P6", "delta_P6_S1"),  # main analysis
         "P6_recv_ge1_vs_s1_minus_s2_P6": ("recv_ge1_P6", "s1_minus_s2_P6"),  # (c)
         "C4f_recv_ge1_vs_s1_minus_s2_C4f": ("recv_ge1_C4f", "s1_minus_s2_C4f"),  # (d)
         "C4f_recv_ge1_vs_delta_C4f_S1": ("recv_ge1_C4f", "delta_C4f_S1")}  # (d) for reference
META = {"analysis": "D90 between-model rank correlation", "status": "post hoc, exploratory; model-level n = 10 (8); no multiplicity correction; added after seeing Table 3 and Table 4; "
        "possibly confounded by model strength (stronger models may both receive more facts and gain more)",
        "statistic": "Spearman rank correlation across models (average ranks for ties) of per-model item means",
        "quantities": {"recv_ge1_<P>": "share of items with >= 1 removed fact received, policy P, state S1 (S1K asks)", "delta_<P>_S1": "policy S1 - C0 S1",
                       "s1_minus_s2_<P>": "policy S1 - policy S2"},
        "pairs": {"P6_recv_ge1_vs_delta_P6_S1": "primary", "P6_recv_ge1_vs_s1_minus_s2_P6": "(c) S1 - S2 instead of delta",
                  "C4f_recv_ge1_vs_s1_minus_s2_C4f": "(d) Forced inquiry receipt vs Forced inquiry S1 - S2", "C4f_recv_ge1_vs_delta_C4f_S1": "(d) reference: Forced inquiry receipt vs Forced inquiry delta"},
        "groups": {"all": "main pool + large (primary, n = 10)", "main": "(a) main pool only, large excluded (n = 8)", "all_excluding_reused": "(b) reused model excluded from all",
                   "main_excluding_reused": "(b) reused model excluded from main pool"},
        "bootstrap": "shared resampling matrix of p2_analysis.Boot (B 10000, seed 20261016, 736 items); per resample, per-model means and Spearman recomputed; "
                     "replicates where either rank vector is constant are dropped and counted; 95% percentile"}


def ranks(x):
    """Average ranks for ties (from 1) per column (axis 0 = models). x: (M,) or (M, B)."""
    gt = (x[:, None] > x[None, :]).sum(1); eq = (x[:, None] == x[None, :]).sum(1)
    return gt + (eq + 1) / 2


def spearman(x, y):
    """x, y: (M,) or (M, B) -> Pearson of the ranks. nan if one side's ranks are constant."""
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(0), ry - ry.mean(0)
    den = np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rx * ry).sum(0) / np.where(den > 0, den, 1), np.nan)


def vectors(M, gold):
    v = PS.vectors(M, gold)
    for pol in ("P6", "C4f"):
        v[f"recv_ge1_{pol}"] = np.array([PA.recv(a) >= 1 for a in M.asks(pol, PA.ST["S1"])], float)
    return v


def corr(V, models, a, b, bt):
    X, Y = np.array([V[m][a] for m in models]), np.array([V[m][b] for m in models])
    rho = spearman(X.mean(1), Y.mean(1))
    Xb, Yb = X[:, bt.idx].mean(2), Y[:, bt.idx].mean(2)  # (M, B)
    rb = spearman(Xb, Yb); ok = ~np.isnan(rb)
    return {"rho": None if np.isnan(rho) else float(rho), "ci95": PA.pct(rb[ok], 0.95) if ok.any() else None, "n_models": len(models), "models": models,
            "n_replicates_undefined": int((~ok).sum())}


def run(root, items_path):
    its = PA.jl(items_path)
    order = [x["hash"] for x in its]; gold = {x["hash"]: x["answer_idx"] for x in its}
    present = [m for m in PA.MAIN + PA.LARGE if PA.available(root, m)]
    main = [m for m in PA.MAIN if m in present]; large = [m for m in PA.LARGE if m in present]
    groups = {"all": main + large, "main": main, "all_excluding_reused": [m for m in main + large if m not in PA.REUSED],
              "main_excluding_reused": [m for m in main if m not in PA.REUSED]}
    bt = PA.Boot(len(order))
    V = {m: vectors(PA.Model(root, m, order), gold) for m in present}
    keys = sorted({k for p in PAIRS.values() for k in p})
    rep = {"meta": META, "n_items": len(order), "B": PA.B, "seed": PA.SEED, "main_pool": main, "large": large, "reused_models": [m for m in main if m in PA.REUSED],
           "main_mean_missing": [m for m in PA.MAIN if m not in present], "excluded": PA.EXCLUDED, "groups": groups,
           "per_model": {m: {k: {"value": float(V[m][k].mean()), "ci95": bt.mean_ci(V[m][k])} for k in keys} for m in present},
           "correlations": {g: {p: corr(V, ms, a, b, bt) for p, (a, b) in PAIRS.items()} for g, ms in groups.items() if len(ms) >= 3}}
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--approved", action="store_true"); a = ap.parse_args()
    if Path(a.root).resolve() == (ROOT / "runs" / "p2" / "eval").resolve() and not a.approved:
        sys.exit("refusing to analyze EVAL outputs without --approved")
    rep = run(a.root, a.items)
    rep["meta"] |= {"p2_analysis_py_sha256": PS.sha(ROOT / "scripts" / "p2" / "p2_analysis.py"), "p2_strata_py_sha256": PS.sha(ROOT / "scripts" / "p2" / "p2_strata.py"),
                    "script_sha256": PS.sha(__file__), "items_sha256": PS.sha(a.items)}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    c = rep["correlations"]["all"]["P6_recv_ge1_vs_delta_P6_S1"]
    print(f"wrote {a.out}: models {rep['groups']['all']}; primary rho {c['rho']:+.3f} {c['ci95']}")


if __name__ == "__main__":
    main()
