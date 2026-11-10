"""D89 item 3: B3 (blind asker) recomputed: blind asker versus the new models' targeted ask (P6), a new exploratory comparison (no multiplicity correction; no GPU or simulator calls).
It is not the D58 registered comparison (blind asker versus the P1b ask pipeline): P6 writes three questions at once with the frozen targeting SUFFIX.
Inputs (read only): blind asker questions data/human_labels/blind_asker_questions.csv, simulator replies runs/v5/d58/b3/b3_replies.jsonl (200 rows; replies from the
simulator C1 server instance on 2026-10-21, same weights as Phase 2, state S1, each question independent), D42 subset runs/v5/d42/subset200.jsonl (sha256 f13ee9de...),
the new models' P6 S1K question records (arms.jsonl asks).
Measures (same definitions as D58 aggregate.b3): disclosure per question = provided questions / questions; item-level share receiving >= 1 and 2 removed facts (recv; F1F2 = both facts). Overall and by the three D35e strata.
Paired differences: blind asker minus P6 (per model and main mean) for disclosure, >= 1 and 2.
Intervals: as in D58, a resampling matrix for the 200 items only (analysis.Bootstrap(200): B 10,000, seed 20261016), 95% percentiles. Stratum cells use the share over the stratum items of the same 200-item matrix.
Main mean: mean of the per-model values (per-model values recomputed in every resample, then averaged).
python scripts/p2/p2_b3.py --root runs/p2/eval --items runs/v5/evalk/evalk.jsonl --out runs/p2/analysis/p2_b3.json --approved
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts" / "p2")]
import p2_analysis as PA  # noqa: E402
import p2_strata as PS  # noqa: E402
from analysis import Bootstrap  # noqa: E402

OLD = Path("$PROJECT_ROOT/runs/v5")  # outputs of the earlier experiment (read only)
B3_REPLIES, SUBSET = OLD / "d58" / "b3" / "b3_replies.jsonl", OLD / "d42" / "subset200.jsonl"
QUESTIONS = ROOT / "data" / "human_labels" / "blind_asker_questions.csv"
SHA = {"b3_replies": "54565efc46cd2779ed628fdd7832a0b3a2ef9a42290018976817e0be56995cef", "subset200": "f13ee9de32403f74cfe8778d06b0d7773611b8491c0876921dc91ae972034805",
       "blind_asker_questions": "c13db095c54359805eed61e418cbedaa833ab7f7200648d489ebb3ac8400ac4a"}
META = {"analysis": "D89 B3 blind asker vs Targeted ask (P6), new models", "status": "new exploratory comparison, not the D58 registered comparison (blind asker vs P1b ask pipeline); no multiplicity correction",
        "difference_from_D58": "comparator is the Targeted ask pipeline (P6): three questions generated at once with the frozen targeting SUFFIX, not P1b",
        "b3_source": "blind asker questions (Claude-family asker, same family as the item author) sent to simulator C1 in state S1, each question independently; replies b3_replies.jsonl obtained on 2026-10-21 from a C1 server instance with the same weights as Phase 2; not re-queried",
        "p6_source": "new-model P6 S1K asks (arms.jsonl) restricted to the 200 D42 subset items",
        "metrics": {"provided_rate": "provided questions / questions", "recv_ge1": "items with >= 1 removed fact received", "recv_2": "items with both removed facts received"},
        "labels": PS.META["labels"],
        "bootstrap": "200-item resampling matrix as in D58 (analysis.Bootstrap(200): B 10000, seed 20261016); stratum cells use the same matrix restricted to stratum items; 95% percentile",
        "main_mean": "mean of per-model values over the 9-model main pool (per resample)"}


def item_stats(sims):
    """Per item: (provided count, question count, facts received)."""
    p = np.array([sum(isinstance(x, dict) and x.get("kind") == "provided" for x in s) for s in sims], float)
    n = np.array([len(s) for s in sims], float)
    r = np.array([PA.recv(s) for s in sims], float)
    return p, n, r


def boot_vals(p, n, r, mask, idx):
    """Per resample: (disclosure, >= 1, 2), over the items in the stratum mask only."""
    w = mask[idx]
    pr = (p[idx] * w).sum(1) / np.where((n[idx] * w).sum(1) > 0, (n[idx] * w).sum(1), np.nan)
    d = np.where(w.sum(1) > 0, w.sum(1), np.nan)
    return {"provided_rate": pr, "recv_ge1": ((r[idx] >= 1) * w).sum(1) / d, "recv_2": ((r[idx] == 2) * w).sum(1) / d}


def point(p, n, r, mask):
    k = mask > 0
    return {"provided_rate": float(p[k].sum() / n[k].sum()) if n[k].sum() else None, "recv_ge1": float((r[k] >= 1).mean()), "recv_2": float((r[k] == 2).mean()),
            "provided": int(p[k].sum()), "asks": int(n[k].sum()), "n_items": int(k.sum())}


def ci(v):
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None


def run(blind, p6, labels, groups):
    """blind: list of sims in item order; p6: {model: list of sims}; labels: stratum labels in item order; groups: {'main_mean': [models]}."""
    nI = len(blind); idx = Bootstrap(nI).idx
    S = {"blind_asker": item_stats(blind), **{m: item_stats(s) for m, s in p6.items()}}
    out = {"n_items": nI, "n_questions_blind": int(S["blind_asker"][1].sum()), "strata_counts": {st: int((labels == st).sum()) for st in PS.STRATA}, "excluded": PA.EXCLUDED, "tables": {}}
    for st in ["all"] + PS.STRATA:
        mask = np.ones(nI) if st == "all" else (labels == st).astype(float)
        bv = {k: boot_vals(*S[k], mask, idx) for k in S}
        row = {}
        for k in S:
            pt = point(*S[k], mask)
            row[k] = pt | {f"{q}_ci95": ci(bv[k][q]) for q in ("provided_rate", "recv_ge1", "recv_2")}
        for g, ms in groups.items():
            row[g] = {q: float(np.mean([row[m][q] for m in ms])) for q in ("provided_rate", "recv_ge1", "recv_2")}
            row[g] |= {f"{q}_ci95": ci(np.mean([bv[m][q] for m in ms], axis=0)) for q in ("provided_rate", "recv_ge1", "recv_2")}
            bv[g] = {q: np.mean([bv[m][q] for m in ms], axis=0) for q in ("provided_rate", "recv_ge1", "recv_2")}
        row["blind_minus"] = {}
        for k in list(p6) + list(groups):
            row["blind_minus"][k] = {q: {"value": row["blind_asker"][q] - row[k][q], "ci95": ci(bv["blind_asker"][q] - bv[k][q])} for q in ("provided_rate", "recv_ge1", "recv_2")}
        out["tables"][st] = row
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--approved", action="store_true"); a = ap.parse_args()
    if Path(a.root).resolve() == (ROOT / "runs" / "p2" / "eval").resolve() and not a.approved:
        sys.exit("refusing to analyze EVAL outputs without --approved")
    for k, f in (("b3_replies", B3_REPLIES), ("subset200", SUBSET), ("blind_asker_questions", QUESTIONS)):
        assert PS.sha(f) == SHA[k], f"{k} hash mismatch"
    its = PA.jl(a.items); order = [x["hash"] for x in its]
    L, art = PS.load_labels(its); lab = dict(zip(order, L))
    sub = [x["hash"] for x in PA.jl(SUBSET)]
    rows = {r["hash"]: r for r in PA.jl(B3_REPLIES)}
    assert set(rows) == set(sub) and len(sub) == 200, "B3 items differ from the D42 subset"
    q = {r["item_id"]: r for r in csv.DictReader(open(QUESTIONS, encoding="utf-8-sig"))}
    assert all([x["q"] for x in rows[h]["asks"]] == [q[rows[h]["item_id"]][k] for k in ("q1", "q2", "q3")] for h in sub), "replies do not match the blind asker questions"
    present = [m for m in PA.MAIN + PA.LARGE if PA.available(a.root, m)]
    main_ = [m for m in PA.MAIN if m in present]
    pos = {h: i for i, h in enumerate(order)}
    p6 = {}
    for m in present:
        A = PA.Model(a.root, m, order).asks("P6", "S1K"); p6[m] = [A[pos[h]] for h in sub]
    rep = run([rows[h]["asks"] for h in sub], p6, np.array([lab[h] for h in sub]), {"main_mean": main_} if main_ else {})
    rep["meta"] = META | {"input_sha256": SHA, "script_sha256": PS.sha(__file__), "p2_analysis_py_sha256": PS.sha(ROOT / "scripts" / "p2" / "p2_analysis.py")}
    rep["main_pool"], rep["large"] = main_, [m for m in PA.LARGE if m in present]
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {a.out}: models {present}, strata {rep['strata_counts']}")


if __name__ == "__main__":
    main()
