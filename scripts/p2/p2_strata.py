"""D88 analysis by necessity stratum (exploratory, outside the pre-registration, no multiplicity correction; no GPU). Does not modify p2_analysis.py or its JSON; only reuses its reading and scoring functions.
Strata = the full D35e labels (736 rows, Claude Sonnet 5.5 LLM labels, not human; fixed before results were seen): necessary / partially / not_necessary (the three strata of D38, user decision).
Sensitivity: without the items with artifact_flag = 1 (as in D38).
Computed per model and stratum: (a) C0 cost S0 - S1, (b) Delta = policy S1 - C0 S1 (C4, C4f, P6), (c) policy S1 - S2. Scoring as in p2_analysis (PA.Model.correct).
Intervals: mean over the stratum items (Boot.masked_mean_ci) in the shared resampling matrix of p2_analysis.Boot (B 10,000, seed 20261016, item-clustered, paired, 736 items), two-sided 95% percentiles.
Main pool (completed models of p2_analysis.MAIN): per-stratum mean (item vector of the model mean) and the mean without the reused model; large models (p2_analysis.LARGE) per model only.
python scripts/p2/p2_strata.py --root runs/p2/eval --items runs/v5/evalk/evalk.jsonl --out runs/p2/analysis/p2_strata.json --approved
"""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts" / "p2")]
import p2_analysis as PA  # noqa: E402

STRATA = ["necessary", "partially", "not_necessary"]
HL, BK = ROOT / "data" / "human_labels", Path("$PROJECT_ROOT/blind_keys")
LABEL_FILES = [HL / f"evalk_necessity_full_part{p:02d}_labels.csv" for p in range(1, 9)]
META = {"analysis": "D88 necessity-stratified analysis", "status": "exploratory, outside the pre-registration; no multiplicity correction",
        "labels": "D35e full necessity labels (736 items), LLM labels by Claude Sonnet 5.5 (not human), fixed before results; strata necessary / partially / not_necessary",
        "sensitivity": "no_artifact = items with artifact_flag = 1 excluded",
        "bootstrap": "shared resampling matrix of p2_analysis.Boot (B 10000, seed 20261016, item-cluster paired, 736 items); stratum mean over resampled items in the stratum; 95% two-sided percentile",
        "quantities": {"C0_cost_S0_minus_S1": "C0 S0 - C0 S1", "delta_<P>_S1": "policy S1 - C0 S1", "s1_minus_s2_<P>": "policy S1 - policy S2"}}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_labels(items):
    """D35e labels (row_id -> original row_id mapping) checked against the registered sha256. Returns label and artifact arrays in item order."""
    reg = dict(reversed(l.split()) for l in open(HL / "evalk_necessity_full_labels_sha256.txt") if l.strip())
    for f in LABEL_FILES:
        assert reg[str(f.relative_to(ROOT))] == sha(f), f"label file hash mismatch: {f}"
    key = {r["row_id"]: r for r in csv.DictReader(open(BK / "evalk_necessity_full_key.csv"))}
    lab = {key[r["row_id"]]["author_row_id"]: r for f in LABEL_FILES for r in csv.DictReader(open(f, encoding="utf-8-sig"))}
    L = np.array([lab[x["row_id"]]["label"] for x in items])
    art = np.array([(lab[x["row_id"]].get("artifact_flag") or "").strip() == "1" for x in items])
    return L, art


def vectors(M, gold):
    acc = {("C0", s): M.correct("C0", PA.ST[s], gold) for s in ("S0", "S1", "S2")}
    for pol in PA.POLICIES:
        for s in ("S1", "S2"):
            acc[(pol, s)] = M.correct(pol, PA.ST[s], gold)
    v = {"C0_cost_S0_minus_S1": acc[("C0", "S0")] - acc[("C0", "S1")]}
    for pol in PA.POLICIES:
        v[f"delta_{pol}_S1"] = acc[(pol, "S1")] - acc[("C0", "S1")]
        v[f"s1_minus_s2_{pol}"] = acc[(pol, "S1")] - acc[(pol, "S2")]
    return v


def cell(x, mask, bt):
    return {"value": float(x[mask > 0].mean()), "ci95": bt.masked_mean_ci(x, mask, 0.95)}


def run(root, items_path, L, art):
    its = PA.jl(items_path)
    order = [x["hash"] for x in its]; gold = {x["hash"]: x["answer_idx"] for x in its}
    assert len(L) == len(art) == len(order)
    present = [m for m in PA.MAIN + PA.LARGE if PA.available(root, m)]
    main = [m for m in PA.MAIN if m in present]; main_nr = [m for m in main if m not in PA.REUSED]
    large = [m for m in PA.LARGE if m in present]
    bt = PA.Boot(len(order))
    V = {m: vectors(PA.Model(root, m, order), gold) for m in present}
    keys = list(V[present[0]])
    rep = {"meta": META, "n_items": len(order), "B": PA.B, "seed": PA.SEED, "main_pool": main, "main_pool_excluding_reused": main_nr,
           "reused_models": [m for m in main if m in PA.REUSED], "large": large, "main_mean_missing": [m for m in PA.MAIN if m not in present], "excluded": PA.EXCLUDED,
           "strata_counts": {st: int((L == st).sum()) for st in STRATA}, "artifact_flag_1": int(art.sum()), "tables": {}}
    for sens in ("all", "no_artifact"):
        tab = {}
        for st in STRATA:
            mask = ((L == st) & (~art if sens == "no_artifact" else True)).astype(float)
            row = {"n": int(mask.sum()), "models": {m: {k: cell(V[m][k], mask, bt) for k in keys} for m in present}}
            for name, ms in (("main_mean", main), ("main_mean_excluding_reused", main_nr)):
                row[name] = {k: cell(np.mean([V[m][k] for m in ms], axis=0), mask, bt) for k in keys} if ms else None
            tab[st] = row
        rep["tables"][sens] = tab
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--approved", action="store_true"); a = ap.parse_args()
    if Path(a.root).resolve() == (ROOT / "runs" / "p2" / "eval").resolve() and not a.approved:
        sys.exit("refusing to analyze EVAL outputs without --approved")
    its = PA.jl(a.items)
    L, art = load_labels(its)
    rep = run(a.root, a.items, L, art)
    rep["meta"] |= {"p2_analysis_py_sha256": sha(ROOT / "scripts" / "p2" / "p2_analysis.py"), "p2_strata_py_sha256": sha(__file__),
                    "items_sha256": sha(a.items), "label_files_sha256": {f.name: sha(f) for f in LABEL_FILES}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {a.out}: models {rep['main_pool'] + rep['large']}, strata {rep['strata_counts']}, artifact {rep['artifact_flag_1']}")


if __name__ == "__main__":
    main()
