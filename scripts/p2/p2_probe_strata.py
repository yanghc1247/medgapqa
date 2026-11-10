"""D97: NP probe AUROC by necessity stratum (post hoc, exploratory, no multiplicity correction, added after a reviewer comment; no GPU). Does not modify
p2_analysis.py, p2_strata.py or their output JSONs; only reuses their functions.
Scores and ranks as in p2_analysis: score logit(No) - logit(Yes), S1 (positive) against S0 (negative), PA.auc (ties counted half = average ranks).
Strata = p2_strata.load_labels (D35e labels): necessary / partially / not_necessary, plus 'all' as a check (736 items, must equal p2_analysis np_probe).
Intervals: in each row of the shared resampling matrix (PA.Boot, B 10,000, seed 20261016, 736 items) only the items of the stratum are kept (duplicates included), AUROC; 95% two-sided percentiles.
Main mean (completed models of PA.MAIN): mean of the per-model AUROC; percentiles of the model mean computed on the same row in every resample. Large models (PA.LARGE) per model only.
python scripts/p2/p2_probe_strata.py --root runs/p2/eval --items runs/v5/evalk/evalk.jsonl --out runs/p2/analysis/p2_probe_strata.json --approved
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts" / "p2")]
import p2_analysis as PA  # noqa: E402
import p2_strata as PS  # noqa: E402

STRATA = ["all"] + PS.STRATA
META = {"analysis": "D97 necessity-stratified NP probe AUROC", "status": "post hoc, exploratory, no multiplicity correction; added after reviewer comment",
        "labels": PS.META["labels"],
        "score": "logit(No) - logit(Yes) of the necessity probe; positive = S1 (fact removed), negative = S0; AUROC with ties counted half (p2_analysis.auc)",
        "bootstrap": "shared resampling matrix of p2_analysis.Boot (B 10000, seed 20261016, 736 items); per resample, AUROC over the resampled items in the stratum; 95% two-sided percentile",
        "main_mean": "mean of per-model AUROC over the main pool (per resample, same rows)",
        "check": "stratum 'all' (736 items) reproduces p2_analysis np_probe AUROC and ci95"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scores(root, m, order):
    r = {x["hash"]: x for x in PA.jl(PA.model_paths(root, m)["np"])}
    s0 = np.array([r[h]["S0"]["logit_no"] - r[h]["S0"]["logit_yes"] for h in order])
    s1 = np.array([r[h]["S1"]["logit_no"] - r[h]["S1"]["logit_yes"] for h in order])
    return s0, s1


def boot_auc(s0, s1, mask, bt):
    """In each row, AUROC over the items of the stratum only (resampled duplicates included)."""
    m = mask.astype(bool)
    return np.array([PA.auc(s1[i[m[i]]], s0[i[m[i]]]) for i in bt.idx])


def run(root, items_path, L):
    its = PA.jl(items_path)
    order = [x["hash"] for x in its]
    assert len(L) == len(order)
    present = [m for m in PA.MAIN + PA.LARGE if PA.available(root, m) and PA.model_paths(root, m)["np"].exists()]
    main = [m for m in PA.MAIN if m in present]; large = [m for m in PA.LARGE if m in present]
    bt = PA.Boot(len(order))
    S = {m: scores(root, m, order) for m in present}
    rep = {"meta": META, "n_items": len(order), "B": PA.B, "seed": PA.SEED, "main_pool": main, "large": large,
           "main_mean_missing": [m for m in PA.MAIN if m not in present], "excluded": PA.EXCLUDED,
           "strata_counts": {st: int((L == st).sum()) for st in PS.STRATA}, "strata": {}}
    for st in STRATA:
        mask = np.ones(len(order)) if st == "all" else (L == st).astype(float)
        m_ = mask.astype(bool)
        pt, bs = {}, {}
        for m in present:
            s0, s1 = S[m]
            pt[m], bs[m] = PA.auc(s1[m_], s0[m_]), boot_auc(s0, s1, mask, bt)
        row = {"n": int(mask.sum()), "models": {m: {"AUROC": pt[m], "ci95": PA.pct(bs[m], 0.95)} for m in present}}
        row["main_mean"] = {"AUROC": float(np.mean([pt[m] for m in main])), "ci95": PA.pct(np.mean([bs[m] for m in main], axis=0), 0.95), "n_models": len(main)} if main else None
        rep["strata"][st] = row
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--approved", action="store_true"); a = ap.parse_args()
    if Path(a.root).resolve() == (ROOT / "runs" / "p2" / "eval").resolve() and not a.approved:
        sys.exit("refusing to analyze EVAL outputs without --approved")
    its = PA.jl(a.items)
    L, _ = PS.load_labels(its)
    rep = run(Path(a.root), a.items, L)
    rep["meta"] = dict(rep["meta"], p2_analysis_py_sha256=sha(ROOT / "scripts" / "p2" / "p2_analysis.py"), p2_strata_py_sha256=sha(ROOT / "scripts" / "p2" / "p2_strata.py"),
                       script_sha256=sha(__file__), items_sha256=sha(a.items), label_files_sha256={f.name: sha(f) for f in PS.LABEL_FILES})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {a.out}: models {rep['main_pool'] + rep['large']}, strata {rep['strata_counts']}")


if __name__ == "__main__":
    main()
