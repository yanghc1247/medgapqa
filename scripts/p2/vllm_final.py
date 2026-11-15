"""D130-n: final (or provisional) vLLM table set. Inputs: runs/p2/analysis_vllm/<mode>/root (evaluation root built by analysis_vllm.py) with its p2_analysis JSON, and the transformers reference analysis_hf.
All intervals: shared p2_analysis.Boot resampling matrix (B 10,000, seed 20261016, 736 items), 95% percentiles. Differences (Think - Instant, pairs, strata) are paired on the same resampling rows.
Contents: (1) Instant main mean over 9 models (preregistered criterion) (2) 'Mean, 7 models with thinking': Instant, Think, difference (3) Think - Instant per model (4) behavior and acquisition: free-agent immediate answers,
question precision (disclosed per question; targeted ask and forced inquiry), at least one fact received, forced-inquiry abstention - Instant, Think, difference (5) R (oracle recovery ratio) (6) Think gain by necessity stratum
(7) Meerkat2 - Qwen3 pairs (Instant and Think) (8) invalid sets (9) vLLM against transformers (10) supplement: Think targeted ask seed 1 against seed 2.
Large models: Meerkat-70B = 'transformers, NF4', DeepSeek-V4-Flash = transformers (changes if the TP4 vLLM run of D130-n 2 succeeds) - Instant large-model rows only.
python scripts/p2/vllm_final.py [--provisional] --out runs/p2/analysis_vllm/final_tables.json --md logs/d130_final_report.md
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts"), str(ROOT / "scripts" / "p2"), str(ROOT / "scripts" / "think")]
import p2_analysis as PA  # noqa: E402
import p2_strata as PS  # noqa: E402
P2 = ROOT / "runs" / "p2"
ITEMS = ROOT / "runs" / "v5" / "evalk" / "evalk.jsonl"
POL = {"C4": "Free agent", "C4f": "Forced inquiry", "P6": "Targeted ask"}
THINK7 = ["qwen3-8b", "qwen3-14b", "meerkat2-14b", "qwen3-30b-a3b", "meerkat2-30b-a3b", "qwen3-32b", "meerkat2-32b"]
FLAG = {"meerkat-70b-nf4": "transformers, NF4", "dsv4-0731": "transformers, FP8"}
if (P2 / "eval_vllm_instant" / "dsv4-0731" / "DONE").exists():  # D130-t: once the DeepSeek Instant vLLM run exists, the tables use its values
    FLAG["dsv4-0731"] = "vLLM TP4"


def ci(v, bt):
    v = np.asarray(v, float); return {"value": float(v.mean()), "ci95": bt.mean_ci(v)}


def f(c):
    return "–" if c is None else f"{c['value']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"


def u(c):
    return "–" if c is None else f"{c['value']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]"


def ratio_vec(asks_list):
    n = np.array([len(a) for a in asks_list], float)
    p = np.array([sum(isinstance(x, dict) and x.get("kind") == "provided" for x in a) for a in asks_list], float)
    return p, n


def ratio_ci(p, n, bt):
    sp, sn = p[bt.idx].sum(axis=1), n[bt.idx].sum(axis=1); ok = sn > 0
    return {"value": float(p.sum() / max(n.sum(), 1)), "ci95": [float(np.percentile(sp[ok] / sn[ok], 2.5)), float(np.percentile(sp[ok] / sn[ok], 97.5))]}


def ratio_diff_ci(pa, na, pb, nb, bt):
    """(rate a - rate b), on the same resampling rows."""
    ra = pa[bt.idx].sum(axis=1) / np.maximum(na[bt.idx].sum(axis=1), 1); rb = pb[bt.idx].sum(axis=1) / np.maximum(nb[bt.idx].sum(axis=1), 1)
    d = ra - rb
    return {"value": float(pa.sum() / max(na.sum(), 1) - pb.sum() / max(nb.sum(), 1)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def mean_ratio_ci(pairs, bt):
    """Mean of a rate over several models (each model's rate computed in every resample, then averaged)."""
    reps = np.mean([p[bt.idx].sum(axis=1) / np.maximum(n[bt.idx].sum(axis=1), 1) for p, n in pairs], axis=0)
    val = float(np.mean([p.sum() / max(n.sum(), 1) for p, n in pairs]))
    return {"value": val, "ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}, reps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--provisional", action="store_true"); ap.add_argument("--out", required=True); ap.add_argument("--md", required=True); a = ap.parse_args()
    PA.REUSED_PATHS = {}; PA.REUSED = set()
    tag = "_provisional" if a.provisional else ""
    J = {m: json.load(open(P2 / "analysis_vllm" / m / f"p2_analysis{tag}.json")) for m in ("instant", "think")}
    H = json.load(open(P2 / "analysis_hf" / "p2_analysis.json"))
    its = PA.jl(str(ITEMS)); order = [x["hash"] for x in its]; gold = {x["hash"]: x["answer_idx"] for x in its}; bt = PA.Boot(len(order))
    L_, _ = PS.load_labels(its); strata = ["necessary", "partially", "not_necessary"]
    roots = {m: P2 / "analysis_vllm" / m / "root" for m in ("instant", "think")}
    M = {mode: {m: PA.Model(roots[mode], m, order) for m in (PA.MAIN + PA.LARGE) if PA.available(roots[mode], m)} for mode in roots}
    t7 = [m for m in THINK7 if m in M["think"] and m in M["instant"]]
    LT = [m for m in PA.LARGE if m in M["think"] and m in M["instant"]]  # D130-t: a large model with Think (DeepSeek) - per-model rows only, not in the mean
    V = {}  # V[mode][m] = dict of item vectors

    def vecs(mode, m):
        X = M[mode][m]; c0 = X.correct("C0", "S1K", gold); d = {"C0_S1": c0, "C0_S0": X.correct("C0", "S0", gold)}
        for p in ("C4", "C4f", "P6"):
            s1, s2 = X.correct(p, "S1K", gold), X.correct(p, "S2K", gold)
            d[f"d_{p}_S1"], d[f"d_{p}_S2"], d[f"s12_{p}"] = s1 - c0, s2 - c0, s1 - s2
        T = X.traj[("C4", "S1K")]; d["immediate"] = np.array([PA.first_action(T[h]) == "ANSWER" for h in order], float)
        T = X.traj[("C4f", "S1K")]; d["abstain_C4f"] = np.array([T[h]["outcome"] == "abstain" for h in order], float)
        for p in ("P6", "C4f"):
            al = X.asks(p, "S1K"); d[f"prov_{p}"], d[f"nask_{p}"] = ratio_vec(al); d[f"ge1_{p}"] = np.array([PA.recv(x) >= 1 for x in al], float)
        o = {c: X.correct("O:" + c, None, gold) for c in ("O2",)}
        d["R_num"], d["R_den"] = o["O2"] - c0, d["C0_S0"] - c0
        return d
    for mode in M:
        V[mode] = {m: vecs(mode, m) for m in M[mode]}
    rep = {"meta": {"provisional": a.provisional, "think7": t7, "bootstrap": f"p2_analysis.Boot({len(order)}) B {PA.B} seed {PA.SEED}", "flags": FLAG}}
    L = [f"# D130 vLLM {'provisional' if a.provisional else 'final'} table set (Instant · Think)", "",
         f"7 thinking models: {', '.join(t7)}. Large models: Meerkat-70B '{FLAG['meerkat-70b-nf4']}', DeepSeek-V4-Flash '{FLAG['dsv4-0731']}'({'Instant and Think per-model rows' if 'dsv4-0731' in LT else 'Instant large-model rows only'}; not in the 7-model mean). Intervals 95% (criterion 97.5%); paired = the same resampling rows.", ""]
    # (1) Instant main mean over 9
    mm = J["instant"]["main_mean"]
    L += ["## 1. Mean, all 9 models (Instant; preregistered criterion)", "", "| Policy | Δ S1 | 97.5% | Δ S2 | S1 − S2 | Improvement |", "|---|---|---|---|---|---|"]
    m9 = [m for m in PA.MAIN if m in V["instant"]]
    rep["mean9_instant"] = {}
    for p in ("C4", "C4f", "P6"):
        x = mm[p]; s12 = ci(np.mean([V["instant"][m][f"s12_{p}"] for m in m9], axis=0), bt)
        rep["mean9_instant"][p] = {"d_s1": x["d_s1"], "d_s1_ci95": x["d_s1_ci95"], "d_s1_ci975": x["d_s1_ci975"], "d_s2": x["d_s2"], "d_s2_ci95": x["d_s2_ci95"], "s1_minus_s2": s12, "improved": x.get("improved")}
        L.append(f"| {POL[p]} | {f({'value': x['d_s1'], 'ci95': x['d_s1_ci95']})} | [{x['d_s1_ci975'][0]:+.3f}, {x['d_s1_ci975'][1]:+.3f}] | {f({'value': x['d_s2'], 'ci95': x['d_s2_ci95']})} | {f(s12)} | {x.get('improved', 'n/a')} |")
    # (2) 7-model mean: Instant, Think, difference
    L += ["", "## 2. Mean, 7 models with thinking (the same 7 models; no criterion for Think)", "", "| Policy | Measure | Instant | Think | Think − Instant |", "|---|---|---|---|---|"]
    rep["mean7"] = {}
    for p in ("C4", "C4f", "P6"):
        for k, lab in ((f"d_{p}_S1", "Δ S1"), (f"d_{p}_S2", "Δ S2"), (f"s12_{p}", "S1 − S2")):
            vi = np.mean([V["instant"][m][k] for m in t7], axis=0); vt = np.mean([V["think"][m][k] for m in t7], axis=0)
            rep["mean7"][f"{p}_{lab}"] = {"instant": ci(vi, bt), "think": ci(vt, bt), "diff": ci(vt - vi, bt)}
            L.append(f"| {POL[p]} | {lab} | {f(ci(vi, bt))} | {f(ci(vt, bt))} | {f(ci(vt - vi, bt))} |")
    vi = np.mean([V["instant"][m]["C0_S1"] for m in t7], axis=0); vt = np.mean([V["think"][m]["C0_S1"] for m in t7], axis=0)
    L.append(f"| C0 | Accuracy S1 | {u(ci(vi, bt))} | {u(ci(vt, bt))} | {f(ci(vt - vi, bt))} |")
    # (3) Think - Instant per model
    L += ["", "## 3. Think − Instant per model (Δ S1)", "", "| Model | Policy | Instant | Think | Think − Instant |", "|---|---|---|---|---|"]
    rep["per_model"] = {}
    for m in t7 + LT:
        for p in ("C4", "C4f", "P6"):
            k = f"d_{p}_S1"; vi, vt = V["instant"][m][k], V["think"][m][k]
            rep["per_model"][f"{m}_{p}"] = {"instant": ci(vi, bt), "think": ci(vt, bt), "diff": ci(vt - vi, bt)}
            L.append(f"| {m}{' (' + FLAG[m] + ')' if m in FLAG else ''} | {POL[p]} | {f(ci(vi, bt))} | {f(ci(vt, bt))} | {f(ci(vt - vi, bt))} |")
    # (4) behavior and acquisition
    L += ["", "## 4. Behavior and acquisition (S1): Instant · Think · Think − Instant", "", "| Model | Measure | Instant | Think | Think − Instant |", "|---|---|---|---|---|"]
    rep["behavior"] = {}
    for m in t7 + LT + ["Mean (7)"]:
        for key, lab in (("immediate", "Free agent immediate answers"), ("abstain_C4f", "Forced inquiry abstention"), ("ge1_P6", "Targeted ask facts ≥1"), ("ge1_C4f", "Forced inquiry facts ≥1")):
            if m == "Mean (7)":
                vi = np.mean([V["instant"][x][key] for x in t7], axis=0); vt = np.mean([V["think"][x][key] for x in t7], axis=0)
            else:
                vi, vt = V["instant"][m][key], V["think"][m][key]
            r = {"instant": ci(vi, bt), "think": ci(vt, bt), "diff": ci(vt - vi, bt)}; rep["behavior"][f"{m}_{key}"] = r
            L.append(f"| {m} | {lab} | {u(r['instant'])} | {u(r['think'])} | {f(r['diff'])} |")
        for p, lab in (("P6", "Targeted ask disclosed/question"), ("C4f", "Forced inquiry disclosed/question")):
            if m == "Mean (7)":
                ri, repi = mean_ratio_ci([(V["instant"][x][f"prov_{p}"], V["instant"][x][f"nask_{p}"]) for x in t7], bt)
                rt, rept = mean_ratio_ci([(V["think"][x][f"prov_{p}"], V["think"][x][f"nask_{p}"]) for x in t7], bt)
                dd = {"value": rt["value"] - ri["value"], "ci95": [float(np.percentile(rept - repi, 2.5)), float(np.percentile(rept - repi, 97.5))]}
            else:
                pi, ni, pt, nt = V["instant"][m][f"prov_{p}"], V["instant"][m][f"nask_{p}"], V["think"][m][f"prov_{p}"], V["think"][m][f"nask_{p}"]
                ri, rt, dd = ratio_ci(pi, ni, bt), ratio_ci(pt, nt, bt), ratio_diff_ci(pt, nt, pi, ni, bt)
            rep["behavior"][f"{m}_prov_{p}"] = {"instant": ri, "think": rt, "diff": dd}
            L.append(f"| {m} | {lab} | {u(ri)} | {u(rt)} | {f(dd)} |")
    # (5) R
    L += ["", "## 5. Oracle recovery ratio R = (O2 − C0 S1)/(C0 S0 − C0 S1)", "", "| Model | Instant | Think |", "|---|---|---|"]
    rep["R"] = {}
    for m in sorted(set(V["instant"]) | set(V["think"])):
        cells = []
        for mode in ("instant", "think"):
            if m in V[mode]:
                r = PA.ratio_R(V[mode][m]["R_num"], V[mode][m]["R_den"], bt); rep["R"][f"{m}_{mode}"] = r
                cells.append(f"{r['R']:.2f} [{r['R_ci95'][0]:.2f}, {r['R_ci95'][1]:.2f}]" if r.get("R") is not None else "– (denominator interval includes 0)")
            else:
                cells.append("–")
        L.append(f"| {m}{' (' + FLAG[m] + ')' if m in FLAG else ''} | {cells[0]} | {cells[1]} |")
    # (6) Think gain by stratum
    L += ["", "## 6. By necessity stratum (mean of 7, Δ S1)", "", "| Policy | Stratum | n | Instant | Think | Think − Instant |", "|---|---|---|---|---|---|"]
    rep["strata"] = {}
    for p in ("C4", "C4f", "P6"):
        vi = np.mean([V["instant"][m][f"d_{p}_S1"] for m in t7], axis=0); vt = np.mean([V["think"][m][f"d_{p}_S1"] for m in t7], axis=0)
        for s in strata:
            mk = (L_ == s).astype(float)
            ri = {"value": float(vi[mk > 0].mean()), "ci95": bt.masked_mean_ci(vi, mk)}; rt = {"value": float(vt[mk > 0].mean()), "ci95": bt.masked_mean_ci(vt, mk)}
            rd = {"value": float((vt - vi)[mk > 0].mean()), "ci95": bt.masked_mean_ci(vt - vi, mk)}
            rep["strata"][f"{p}_{s}"] = {"n": int(mk.sum()), "instant": ri, "think": rt, "diff": rd}
            L.append(f"| {POL[p]} | {s} | {int(mk.sum())} | {f(ri)} | {f(rt)} | {f(rd)} |")
    # (7) pairs
    L += ["", "## 7. Meerkat2 − Qwen3 pairs (by mode)", "", "| Pair | Mode | Δ S1 Targeted | Δ S1 Forced | Disclosed/question Targeted | C0 S1 accuracy | Free agent immediate |", "|---|---|---|---|---|---|---|"]
    rep["pairs"] = {}
    for a_, b_ in PA.PAIRS:
        for mode in ("instant", "think"):
            if a_ not in V[mode] or b_ not in V[mode]:
                continue
            A_, B_ = V[mode][a_], V[mode][b_]
            r = {"d_P6_S1": ci(A_["d_P6_S1"] - B_["d_P6_S1"], bt), "d_C4f_S1": ci(A_["d_C4f_S1"] - B_["d_C4f_S1"], bt),
                 "prov_P6": ratio_diff_ci(A_["prov_P6"], A_["nask_P6"], B_["prov_P6"], B_["nask_P6"], bt), "C0_S1": ci(A_["C0_S1"] - B_["C0_S1"], bt), "immediate": ci(A_["immediate"] - B_["immediate"], bt)}
            rep["pairs"][f"{a_}_minus_{b_}_{mode}"] = r
            L.append(f"| {a_} − {b_} | {mode} | {f(r['d_P6_S1'])} | {f(r['d_C4f_S1'])} | {f(r['prov_P6'])} | {f(r['C0_S1'])} | {f(r['immediate'])} |")
    # (8) invalid sets
    L += ["", "## 8. Invalid sets (targeted ask, items, after D130-k)", "", "| Model | Instant | Think |", "|---|---|---|"]
    for m in sorted(set(V["instant"]) | set(V["think"])):
        cells = []
        for mode in ("instant", "think"):
            p = roots[mode] / m / "p6" / m / "arms.jsonl"
            if p.exists():
                rows = [json.loads(l) for l in open(p)]; cells.append(str(sum(not r["gens"]["Q3"].get("valid") for r in rows if r["state"] == "S1K")))
            else:
                cells.append("–")
        L.append(f"| {m}{' (' + FLAG[m] + ')' if m in FLAG else ''} | {cells[0]} | {cells[1]} |")
    # (9) vLLM against transformers
    hm = H["main_mean"]
    L += ["", "## 9. Instant vLLM against transformers (analysis_hf, D126 1,024; supplement engine check)", "", "| Measure | transformers | vLLM |", "|---|---|---|"]
    for p in ("C4", "C4f", "P6"):
        L.append(f"| Main mean Δ S1 {POL[p]} | {f({'value': hm[p]['d_s1'], 'ci95': hm[p]['d_s1_ci95']})} | {f({'value': mm[p]['d_s1'], 'ci95': mm[p]['d_s1_ci95']})} |")
    L.append(f"| Main mean C0 cost | {f(hm['C0_cost_S0_minus_S1'])} | {f(mm['C0_cost_S0_minus_S1'])} |")
    # (10) seed supplement
    L += ["", "## 10. Supplement: Think targeted ask seed 1 (D130 main table) against seed 2 (source per row)", ""]
    try:
        import think_analysis as TA
        rows = []
        for m in t7:
            X = M["think"][m]
            v2 = P2 / "eval_vllm_think_seed2" / m  # D130-o 1: seed 2 rerun under the D130 settings (veval format; currently qwen3-8b)
            if (v2 / "p6" / m / "arms.jsonl").exists() and (v2 / "c0" / m / "C0" / "S1K" / "trajectories.jsonl").exists():
                # seed 2 has only C0 and P6, so the two files are read directly instead of PA.Model (which needs every cell) (same letter rule)
                t2 = {r["hash"]: r for r in PA.jl(v2 / "c0" / m / "C0" / "S1K" / "trajectories.jsonl")}
                a2 = {(r["hash"], r["state"]): r for r in PA.jl(v2 / "p6" / m / "arms.jsonl")}
                Lc2 = [PA.fcr_letter(t2[h]) for h in order]; Lp2 = [a2[(h, "S1K")]["arms"]["P6"]["letter"] for h in order]; src2 = "eval_vllm_think_seed2 (D130 settings)"
            else:
                s2c = P2 / "thinkseed2" / "c0" / m
                s2p = P2 / "thinkseed2_d130" / m / "think" if (P2 / "thinkseed2_d130" / m / "think" / "DONE").exists() else P2 / "thinkseed2" / "targeted" / m / "think"
                if not (s2p / "DONE").exists():
                    continue
                c0b = {r["hash"]: r for r in map(json.loads, open(s2c / "think_C0_S1K.jsonl"))}
                pb = {(r["hash"], r["state"]): r for r in map(json.loads, open(s2p / "p6.jsonl"))}
                Lc2 = [TA.letter("C0_S1K", c0b[h]) for h in order]; Lp2 = [pb[(h, "S1K")]["letter"] for h in order]
                src2 = "thinkseed2_d130 (1,024 cut continued)" if "thinkseed2_d130" in str(s2p) else "thinkseed2 (D126-d)"
            Lc1 = [PA.fcr_letter(X.traj[("C0", "S1K")][h]) for h in order]
            Lp1 = [X.p6[(h, "S1K")]["arms"]["P6"]["letter"] for h in order]
            g = np.array([gold[h] for h in order])
            c1, c2 = (np.array(Lc1) == g).astype(float), (np.array(Lc2) == g).astype(float); p1, p2 = (np.array(Lp1) == g).astype(float), (np.array(Lp2) == g).astype(float)
            d1, d2 = p1 - c1, p2 - c2
            r = {"seed2_source": src2, "agree_C0_S1": float(np.mean(np.array(Lc1) == np.array(Lc2))), "agree_P6_S1": float(np.mean(np.array(Lp1) == np.array(Lp2))),
                 "delta_S1_seed1": ci(d1, bt), "delta_S1_seed2": ci(d2, bt), "seed_diff": ci(d2 - d1, bt), "delta_S1_mean2": ci((d1 + d2) / 2, bt)}
            rep.setdefault("seeds", {})[m] = r; rows.append((m, r))
        L += ["| Model | seed 2 source | Agreement C0 S1 | Agreement Targeted S1 | Δ S1 seed 1 | Δ S1 seed 2 | seed 2 − 1 | Mean of two seeds |", "|---|---|---|---|---|---|---|---|"]
        for m, r in rows:
            L.append(f"| {m} | {r['seed2_source']} | {r['agree_C0_S1']:.3f} | {r['agree_P6_S1']:.3f} | {f(r['delta_S1_seed1'])} | {f(r['delta_S1_seed2'])} | {f(r['seed_diff'])} | {f(r['delta_S1_mean2'])} |")
        L.append("\nNote: C0 of seed 1 is the new D130 run (no answer-turn limit). C0 of seed 2 is a new run under the D130 settings for qwen3-8b and the D126-d run for the others (answer turn 256; no output reached 256, so this equals no limit). Models without seed 2 are not shown (marked 'incomplete' while pending).")
    except Exception as e:  # noqa: BLE001
        L.append(f"(seed supplement failed: {e!r})")
    # D130-u: large-model C0 accuracy (per-model rows, outside the mean; models without Think in Instant only). Paired differences on the same items and resampling rows.
    L += ["", "## 11. Large-model C0 accuracy and cost of removal (per model; not in the mean)", "", "| Model | Measure | Instant | Think | Think − Instant |", "|---|---|---|---|---|"]
    rep["large_c0"] = {}
    def c0vecs(mode, m):
        X = M[mode][m]; s0, s1, s2 = (X.correct("C0", st, gold) for st in ("S0", "S1K", "S2K"))
        return {"C0_S0": s0, "C0_S1": s1, "C0_S2": s2, "O2": X.correct("O:O2", None, gold), "cost_S1": s0 - s1, "cost_S2": s0 - s2}
    for m in [x for x in PA.LARGE if x in M["instant"]]:
        ci_ = c0vecs("instant", m); ct = c0vecs("think", m) if m in M["think"] else None
        for key, lab, acc in (("C0_S0", "C0 accuracy S0", 1), ("C0_S1", "C0 accuracy S1", 1), ("C0_S2", "C0 accuracy S2", 1), ("O2", "Oracle O2 accuracy", 1),
                              ("cost_S1", "Cost of removal S0 − S1", 0), ("cost_S2", "Cost of removal S0 − S2", 0)):
            fm = u if acc else f
            r = {"instant": ci(ci_[key], bt), "think": ci(ct[key], bt) if ct else None, "diff": ci(ct[key] - ci_[key], bt) if ct else None}
            rep["large_c0"][f"{m}_{key}"] = r
            L.append(f"| {m}{' (' + FLAG[m] + ')' if m in FLAG else ''} | {lab} | {fm(r['instant'])} | {fm(r['think']) if ct else '–'} | {f(r['diff']) if ct else '–'} |")
    L.append("\nNote: C0 asks no questions, so the model input is the same in S1 and S2 (the two states differ only in the facts the simulator may disclose; the C0 S1K and S2K records are identical apart from the state field). Hence C0 S2 = C0 S1 and the cost S0 − S2 = S0 − S1. Large models without Think fill the Instant columns only.")
    dt = P2 / "eval_vllm_think" / "dsv4-0731"
    # D130-s: items not yet processed (decided from the data; they disappear automatically once done)
    inc = []
    if "d130k" not in json.loads((P2 / "eval_vllm_think" / "meerkat2-32b" / "DONE").read_text()):
        inc.append("meerkat2-32b Think: D130-k redo of 1 item pending (current value = previous output)")
    if not (P2 / "eval_hf_nocap" / "meerkat-70b-nf4" / "D130O_NOCAP.json").exists():
        inc.append("Meerkat-70B Free agent S1/S2: 2 episodes (1 item) still at the 256 action-turn cap (current value = capped output)")
    if not (P2 / "thinkseed2_d130" / "meerkat2-32b" / "think" / "DONE").exists():
        inc.append("seed-2 table: meerkat2-32b uses the uncontinued seed-2 output (1 item cut at 1,024)")
    if not (P2 / "eval_vllm_think_seed2" / "qwen3-8b").exists():
        inc.append("seed-2 table: qwen3-8b seed 2 not run (row absent)")
    if not (dt / "DONE").exists():
        done_c = [c for c, f_ in (("C0", dt / "c0" / "dsv4-0731" / "C0" / "S2K" / "trajectories.jsonl"), ("oracle", dt / "oracle.jsonl"), ("Targeted ask", dt / "p6" / "dsv4-0731" / "arms.jsonl"),
                  ("Free agent", dt / "c4" / "dsv4-0731" / "C4" / "S2K" / "trajectories.jsonl"), ("Forced inquiry", dt / "c4f" / "dsv4-0731" / "C4" / "S2K" / "trajectories.jsonl"))
                  if f_.exists() and sum(1 for _ in open(f_)) >= (1472 if c == "Targeted ask" else 736)]
        inc.append("DeepSeek-V4-Flash Think: " + (f"done cells {', '.join(done_c)}; remaining cells running" if done_c else "not run (rows absent)"))
    if not (P2 / "eval_vllm_instant" / "dsv4-0731" / "DONE").exists():
        inc.append("DeepSeek-V4-Flash Instant: transformers FP8 values (vLLM run pending)")
    rep["incomplete"] = inc
    L[1:1] = ["", "## Incomplete", ""] + [f"- incomplete: {x}" for x in inc] + ([] if inc else ["- none"])
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); Path(a.md).write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
