"""D136 (post hoc, exploratory): conditional use of received facts in the targeted ask pipeline (P6) at S1.
Within the same items, P6 accuracy on items where the pipeline received only fact a / only fact b / both facts is compared with the oracle that appends
exactly those facts (O1a / O1b / O2). Pooled over received items: difference P6 - matched oracle, and use efficiency
(P6 - C0 S1) / (matched oracle - C0 S1) (sums over received items). Paired item bootstrap, shared p2_analysis.Boot matrix (B 10000, seed 20261016), 95%.
Means over models use the same resampled items for every model (per-model masked means, then averaged; efficiency = mean numerator / mean denominator).
Fact a = removed_facts[0] = simulator label F1 (O1a), fact b = removed_facts[1] = F2 (O1b). P6 and the oracles are scored from the A-D letter logits;
C0 from the parsed generated answer. Also reported: efficiency against L-S1K (the logit-scored S1 baseline, same scoring as P6 and the oracles).
Analysis code is imported, not edited. Output: runs/p2/analysis_vllm_paper/conditional_use.json
python scripts/paper/d136_conditional_use.py
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts" / "p2")]
import p2_analysis as PA  # noqa: E402
PA.REUSED_PATHS = {}; PA.REUSED = set()
I = str(ROOT / "runs" / "v5" / "evalk" / "evalk.jsonl")
ROOTS = {"instant": ROOT / "runs" / "p2" / "analysis_vllm" / "instant" / "root", "think": ROOT / "runs" / "p2" / "analysis_vllm" / "think" / "root"}
OUT = ROOT / "runs" / "p2" / "analysis_vllm_paper" / "conditional_use.json"
THINK7 = ["qwen3-8b", "qwen3-14b", "meerkat2-14b", "qwen3-30b-a3b", "meerkat2-30b-a3b", "qwen3-32b", "meerkat2-32b"]
LARGE = {"instant": ["meerkat-70b-nf4", "dsv4-0731"], "think": ["dsv4-0731"]}


def pc(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def vectors(root, m, order, gold):
    M = PA.Model(str(root), m, order)
    p6 = M.correct("P6", "S1K", gold); c0 = M.correct("C0", "S1K", gold)
    o = {c: M.correct("O:" + c, None, gold) for c in ("O1a", "O1b", "O2", "L-S1K")}
    rs = []
    for a in M.asks("P6", "S1K"):
        s = set()
        for x in a:
            if isinstance(x, dict) and x.get("kind") == "provided":
                s |= PA.FACTS.get(x.get("label"), set())
        rs.append(s)
    g = {"a_only": np.array([s == {"F1"} for s in rs], float), "b_only": np.array([s == {"F2"} for s in rs], float), "both": np.array([s == {"F1", "F2"} for s in rs], float)}
    om = g["a_only"] * o["O1a"] + g["b_only"] * o["O1b"] + g["both"] * o["O2"]
    g["any"] = g["a_only"] + g["b_only"] + g["both"]
    match = {"a_only": o["O1a"], "b_only": o["O1b"], "both": o["O2"], "any": om}
    return {"p6": p6, "c0": c0, "ls1": o["L-S1K"], "g": g, "match": match}


def summarize(Vs, idx):
    """Vs: list of per-model vectors (one model = per-model result; several = mean over models)."""
    out = {}
    for grp in ("a_only", "b_only", "both", "any"):
        ns = [float(v["g"][grp].sum()) for v in Vs]
        if min(ns) == 0:
            out[grp] = {"n_items": ns, "note": "no items in this group for at least one model"}; continue
        def mm(x, mask):
            return float((x * mask).sum() / mask.sum())
        def mmb(x, mask):
            n, d = (x * mask)[idx].sum(axis=1), mask[idx].sum(axis=1)
            return np.where(d > 0, n / np.where(d > 0, d, 1), np.nan)
        p6 = [mm(v["p6"], v["g"][grp]) for v in Vs]; orc = [mm(v["match"][grp], v["g"][grp]) for v in Vs]
        db = np.nanmean(np.stack([mmb(v["p6"] - v["match"][grp], v["g"][grp]) for v in Vs]), axis=0)
        r = {"n_items": ns if len(Vs) > 1 else ns[0], "p6_acc": float(np.mean(p6)), "oracle_acc": float(np.mean(orc)),
             "diff": {"value": float(np.mean(p6) - np.mean(orc)), "ci95": pc(db[~np.isnan(db)])}}
        if grp == "any":
            for base, key in (("c0", "efficiency_vs_C0S1"), ("ls1", "efficiency_vs_LS1K")):
                num = [mm(v["p6"] - v[base], v["g"][grp]) for v in Vs]; den = [mm(v["match"][grp] - v[base], v["g"][grp]) for v in Vs]
                nb = np.nanmean(np.stack([mmb(v["p6"] - v[base], v["g"][grp]) for v in Vs]), axis=0)
                dbb = np.nanmean(np.stack([mmb(v["match"][grp] - v[base], v["g"][grp]) for v in Vs]), axis=0)
                ok = (~np.isnan(nb)) & (~np.isnan(dbb)) & (dbb != 0)
                r[key] = {"value": float(np.mean(num) / np.mean(den)) if np.mean(den) else None, "ci95": pc(nb[ok] / dbb[ok]),
                          "num": float(np.mean(num)), "den": float(np.mean(den)), "den_ci95": pc(dbb[~np.isnan(dbb)])}
        out[grp] = r
    return out


def main():
    its = PA.jl(I); order = [x["hash"] for x in its]; gold = {x["hash"]: x["answer_idx"] for x in its}; idx = PA.Boot(len(order)).idx
    V = {md: {} for md in ROOTS}
    for m in PA.MAIN + LARGE["instant"]:
        V["instant"][m] = vectors(ROOTS["instant"], m, order, gold)
    for m in THINK7 + LARGE["think"]:
        V["think"][m] = vectors(ROOTS["think"], m, order, gold)
    res = {"meta": {"analysis": __doc__.strip().splitlines()[0], "B": PA.B, "seed": PA.SEED, "state": "S1K", "policy": "P6 (targeted ask pipeline)"}, "per_model": {}, "means": {}}
    for md, d in V.items():
        for m, v in d.items():
            res["per_model"][f"{m}_{md}"] = summarize([v], idx)
    res["means"]["main9_instant"] = summarize([V["instant"][m] for m in PA.MAIN], idx)
    res["means"]["think7_instant"] = summarize([V["instant"][m] for m in THINK7], idx)
    res["means"]["think7_think"] = summarize([V["think"][m] for m in THINK7], idx)
    # Think minus Instant, 7 hybrid models, pooled difference P6 - matched oracle (item sets differ between modes; each mode's masked mean per resample)
    def mmb(v, grp):
        x, mask = v["p6"] - v["match"][grp], v["g"][grp]; n, dd = (x * mask)[idx].sum(axis=1), mask[idx].sum(axis=1)
        return np.where(dd > 0, n / np.where(dd > 0, dd, 1), np.nan)
    dI = np.nanmean(np.stack([mmb(V["instant"][m], "any") for m in THINK7]), axis=0); dT = np.nanmean(np.stack([mmb(V["think"][m], "any") for m in THINK7]), axis=0)
    dd = dT - dI
    res["means"]["think7_think_minus_instant_any_diff"] = {"value": res["means"]["think7_think"]["any"]["diff"]["value"] - res["means"]["think7_instant"]["any"]["diff"]["value"], "ci95": pc(dd[~np.isnan(dd)])}
    # D137: efficiency difference Think - Instant (7 hybrid models; each mode's mean numerator / mean denominator per resample), and P6 - L-S1K on items
    # that received no fact (mean over models of masked means), for the main pool (Instant) and the 7 hybrid models in both modes
    def effb(Vs):
        nb = np.nanmean(np.stack([mb(v["p6"] - v["c0"], v["g"]["any"]) for v in Vs]), axis=0)
        db = np.nanmean(np.stack([mb(v["match"]["any"] - v["c0"], v["g"]["any"]) for v in Vs]), axis=0)
        return nb / db
    def mb(x, mask):
        n, dd = (x * mask)[idx].sum(axis=1), mask[idx].sum(axis=1)
        return np.where(dd > 0, n / np.where(dd > 0, dd, 1), np.nan)
    e = effb([V["think"][m] for m in THINK7]) - effb([V["instant"][m] for m in THINK7]); e = e[~np.isnan(e)]
    res["means"]["think7_efficiency_think_minus_instant"] = {"value": res["means"]["think7_think"]["any"]["efficiency_vs_C0S1"]["value"] - res["means"]["think7_instant"]["any"]["efficiency_vs_C0S1"]["value"], "ci95": pc(e)}
    res["no_fact_items"] = {"definition": "P6 minus L-S1K accuracy on items that received no removed fact (mean over models of per-model means)"}
    for key, md, ms in (("main9_instant", "instant", PA.MAIN), ("think7_instant", "instant", THINK7), ("think7_think", "think", THINK7)):
        Vs = [V[md][m] for m in ms]
        val = float(np.mean([((v["p6"] - v["ls1"]) * (1 - v["g"]["any"])).sum() / (1 - v["g"]["any"]).sum() for v in Vs]))
        bb = np.nanmean(np.stack([mb(v["p6"] - v["ls1"], 1 - v["g"]["any"]) for v in Vs]), axis=0)
        res["no_fact_items"][key] = {"value": val, "ci95": pc(bb[~np.isnan(bb)]), "n_items_mean": float(np.mean([(1 - v["g"]["any"]).sum() for v in Vs]))}
    # D136: scoring routes. Share of items whose C0 answer at S1 (parsed generated answer) has the same correctness as the logit-scored S1 baseline (L-S1K),
    # mean over models; used in the supplement's scoring paragraph (Instant, main pool) and reported for Think (7 hybrid models)
    res["scoring_agreement"] = {"definition": "mean over models of share of items where C0 S1K correctness equals L-S1K correctness",
                                "main9_instant": float(np.mean([(V["instant"][m]["c0"] == V["instant"][m]["ls1"]).mean() for m in PA.MAIN])),
                                "think7_think": float(np.mean([(V["think"][m]["c0"] == V["think"][m]["ls1"]).mean() for m in THINK7]))}
    OUT.write_text(json.dumps(res, indent=1)); print("wrote", OUT)


if __name__ == "__main__":
    main()
