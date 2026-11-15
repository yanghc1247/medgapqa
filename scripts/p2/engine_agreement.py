"""D130-o 3: engine agreement check (vLLM Instant against transformers Instant). Raw outputs runs/p2/d130_checks/eng_instant/<model>/{c4,c4f}/<model>/C4/S1K (100 items) and
the earlier transformers runs/p2/eval/<model> (same items). Measures: first-action agreement, identical raw trajectories, final-answer agreement, accuracy (FCR scoring) difference with a 95% interval (resampling matrix for the 100 items,
p2_analysis.Boot(100), B 10,000, seed 20261016). The main-mean comparison is analysis_vllm/instant/p2_analysis[_provisional].json against analysis_hf/p2_analysis.json.
python scripts/p2/engine_agreement.py --out runs/p2/analysis_vllm/engine_agreement.json --md logs/d130_engine_agreement.md"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts"), str(ROOT / "scripts" / "p2")]
import p2_analysis as PA  # noqa: E402
P2 = ROOT / "runs" / "p2"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--md", required=True); a = ap.parse_args()
items = [json.loads(l) for l in open(ROOT / "runs" / "v5" / "evalk" / "evalk.jsonl")][:100]
gold = {x["hash"]: x["answer_idx"] for x in items}; order = [x["hash"] for x in items]; bt = PA.Boot(100)
rep = {"meta": {"items": 100, "state": "S1K", "vllm": "d130_checks/eng_instant (batch-invariant, greedy, no cap)", "transformers": "runs/p2/eval (D74 run: action turn 256)",
                "bootstrap": "p2_analysis.Boot(100), B 10,000, seed 20261016, 95% percentile"}, "rows": []}
L = ["# D130 engine agreement check: vLLM Instant against transformers Instant", "", "100 items (first 100 of EVAL), S1K. vLLM = batch-invariant greedy, no generation limit. transformers = earlier D74 run (action turn 256). Intervals: resampling matrix for the 100 items (B 10,000, seed 20261016).", "",
     "| Model | Policy | First action agrees | Raw trajectory identical | Final answer agrees | Accuracy tf | Accuracy vLLM | Difference (vLLM - tf) [95%] | First actions tf | First actions vLLM |", "|---|---|---|---|---|---|---|---|---|---|"]
for m in ("qwen3-14b", "meerkat2-32b"):
    for pol, sub in (("Free agent", "c4"), ("Forced inquiry", "c4f")):
        Vv = {r["hash"]: r for r in map(json.loads, open(P2 / "d130_checks" / "eng_instant" / m / sub / m / "C4" / "S1K" / "trajectories.jsonl"))}
        Hh = {r["hash"]: r for r in map(json.loads, open(P2 / "eval" / m / sub / m / "C4" / "S1K" / "trajectories.jsonl"))}
        fa = np.mean([PA.first_action(Vv[h]) == PA.first_action(Hh[h]) for h in order]); full = np.mean([[t.get("raw") for t in Vv[h]["turns"]] == [t.get("raw") for t in Hh[h]["turns"]] for h in order])
        agree = np.mean([PA.fcr_letter(Vv[h]) == PA.fcr_letter(Hh[h]) for h in order])
        lv = np.array([PA.fcr_letter(Vv[h]) == gold[h] for h in order], float); lh = np.array([PA.fcr_letter(Hh[h]) == gold[h] for h in order], float)
        d = lv - lh; c = bt.mean_ci(d)
        dist = lambda T: {k: sum(PA.first_action(T[h]) == k for h in order) for k in ("ANSWER", "ASK", "SEARCH", "ABSTAIN")}
        r = {"model": m, "policy": pol, "first_action_agree": float(fa), "trajectory_identical": float(full), "final_answer_agree": float(agree), "acc_tf": float(lh.mean()), "acc_vllm": float(lv.mean()),
             "acc_diff": {"value": float(d.mean()), "ci95": c}, "first_action_tf": dist(Hh), "first_action_vllm": dist(Vv)}
        rep["rows"].append(r)
        L.append(f"| {m} | {pol} | {fa:.2f} | {full:.2f} | {agree:.2f} | {lh.mean():.3f} | {lv.mean():.3f} | {d.mean():+.3f} [{c[0]:+.3f}, {c[1]:+.3f}] | {r['first_action_tf']} | {r['first_action_vllm']} |")
f = lambda c: f"{c['value']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
vj = P2 / "analysis_vllm" / "instant" / "p2_analysis.json"
if not vj.exists():
    vj = P2 / "analysis_vllm" / "instant" / "p2_analysis_provisional.json"
V, H = json.load(open(vj)), json.load(open(P2 / "analysis_hf" / "p2_analysis.json"))
L += ["", f"## Main-mean comparison (Instant, 9 models): vLLM({vj.name}) against transformers(analysis_hf = D126 1,024)", "", "| Measure | transformers | vLLM | Difference (vLLM - tf) |", "|---|---|---|---|"]
rep["main_mean"] = {}
for p, lab in (("C4", "Free agent"), ("C4f", "Forced inquiry"), ("P6", "Targeted ask")):
    for k, kl in (("d_s1", "Δ S1"), ("d_s2", "Δ S2")):
        hv, vv = H["main_mean"][p][k], V["main_mean"][p][k]
        rep["main_mean"][f"{p}_{k}"] = {"tf": {"value": hv, "ci95": H["main_mean"][p][k + "_ci95"]}, "vllm": {"value": vv, "ci95": V["main_mean"][p][k + "_ci95"]}, "diff_point": vv - hv}
        L.append(f"| {lab} {kl} | {f({'value': hv, 'ci95': H['main_mean'][p][k + '_ci95']})} | {f({'value': vv, 'ci95': V['main_mean'][p][k + '_ci95']})} | {vv - hv:+.3f} |")
L.append(f"| C0 cost S0 − S1 | {f(H['main_mean']['C0_cost_S0_minus_S1'])} | {f(V['main_mean']['C0_cost_S0_minus_S1'])} | {V['main_mean']['C0_cost_S0_minus_S1']['value'] - H['main_mean']['C0_cost_S0_minus_S1']['value']:+.3f} |")
L.append("\nThe main-mean difference combines the engine and the generation limit (transformers 1,024 against vLLM no limit + D130-k). The model set (9) is the same. No interval is computed for the main-mean difference, since the two analyses are different runs (point difference only).")
rep["main_mean_source"] = {"vllm": str(vj.relative_to(ROOT)), "tf": "runs/p2/analysis_hf/p2_analysis.json"}
# D130-p: main-mean (9 models) vLLM - transformers difference, cluster bootstrap pairing the same items (shared 736-item resampling matrix)
its = PA.jl(str(ROOT / "runs" / "v5" / "evalk" / "evalk.jsonl")); ordr = [x["hash"] for x in its]; gld = {x["hash"]: x["answer_idx"] for x in its}; btf = PA.Boot(len(ordr))
saved = dict(PA.REUSED_PATHS)
tf_models = {m: PA.Model(P2 / "eval_cap1024", m, ordr) for m in PA.MAIN if PA.available(P2 / "eval_cap1024", m)}  # MedGemma-4B = reused v5 (transformers)
PA.REUSED_PATHS = {}
vroot = P2 / "analysis_vllm" / "instant" / "root"
v_models = {m: PA.Model(vroot, m, ordr) for m in PA.MAIN if PA.available(vroot, m)}
PA.REUSED_PATHS = saved
ms = [m for m in PA.MAIN if m in tf_models and m in v_models]
def mvec(Ms, fn):
    return np.mean([fn(Ms[m]) for m in ms], axis=0)
metrics = {"C0 cost S0 − S1": lambda X: X.correct("C0", "S0", gld) - X.correct("C0", "S1K", gld)}
for p_, lab in (("C4", "Free agent"), ("C4f", "Forced inquiry"), ("P6", "Targeted ask")):
    metrics[f"{lab} Δ S1"] = (lambda p_: lambda X: X.correct(p_, "S1K", gld) - X.correct("C0", "S1K", gld))(p_)
    metrics[f"{lab} Δ S2"] = (lambda p_: lambda X: X.correct(p_, "S2K", gld) - X.correct("C0", "S1K", gld))(p_)
L += ["", f"## Main-mean vLLM − transformers difference: cluster bootstrap pairing the same items (736 items, B {PA.B}, seed {PA.SEED}, 95%; models {len(ms)})", "",
      "| Measure | transformers | vLLM | Difference (vLLM - tf) [95%] |", "|---|---|---|---|"]
rep["main_mean_paired"] = {"models": ms}
for k, fn in metrics.items():
    t, v = mvec(tf_models, fn), mvec(v_models, fn); d = v - t
    r = {"tf": {"value": float(t.mean()), "ci95": btf.mean_ci(t)}, "vllm": {"value": float(v.mean()), "ci95": btf.mean_ci(v)}, "diff": {"value": float(d.mean()), "ci95": btf.mean_ci(d)}}
    rep["main_mean_paired"][k] = r
    L.append(f"| {k} | {f(r['tf'])} | {f(r['vllm'])} | {f(r['diff'])} |")
L.append("\nThe paired intervals pair the same items of the two runs (D130-p). The values must equal the point differences above.")
# D130-t: DeepSeek-V4-Flash Instant - vLLM (TP4, batch invariant off, no limit) minus transformers (FP8, D126 1,024), paired on the same items
dv = P2 / "eval_vllm_instant" / "dsv4-0731"
if (dv / "DONE").exists():
    saved = dict(PA.REUSED_PATHS); PA.REUSED_PATHS = {}
    Xv = PA.Model(P2 / "eval_vllm_instant", "dsv4-0731", ordr); Xt = PA.Model(P2 / "eval_cap1024", "dsv4-0731", ordr); PA.REUSED_PATHS = saved
    L += ["", f"## DeepSeek-V4-Flash Instant: vLLM (TP4) − transformers (FP8), paired on the same items (736 items, B {PA.B}, seed {PA.SEED}, 95%)", "",
          "| Measure | transformers | vLLM | Difference (vLLM - tf) [95%] |", "|---|---|---|---|"]
    rep["dsv4_paired"] = {}
    for k, fn in metrics.items():
        t, v = fn(Xt), fn(Xv); d = v - t
        r = {"tf": {"value": float(t.mean()), "ci95": btf.mean_ci(t)}, "vllm": {"value": float(v.mean()), "ci95": btf.mean_ci(v)}, "diff": {"value": float(d.mean()), "ci95": btf.mean_ci(d)}}
        rep["dsv4_paired"][k] = r; L.append(f"| {k} | {f(r['tf'])} | {f(r['vllm'])} | {f(r['diff'])} |")
    ag = {}
    for cell, st in (("C0", "S1K"), ("C4", "S1K"), ("C4f", "S1K")):
        Tv, Tt = Xv.traj[(cell, st)], Xt.traj[(cell, st)]
        ag[f"{cell}_{st}_final_letter"] = float(np.mean([PA.fcr_letter(Tv[h]) == PA.fcr_letter(Tt[h]) for h in ordr]))
        ag[f"{cell}_{st}_first_action"] = float(np.mean([PA.first_action(Tv[h]) == PA.first_action(Tt[h]) for h in ordr]))
    rep["dsv4_agreement"] = ag
    L.append("\nAgreement (S1): " + ", ".join(f"{k} {v:.3f}" for k, v in ag.items()) + ". DeepSeek Instant in the tables uses the vLLM values; the transformers FP8 values are used only for this engine check.")
Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); Path(a.md).write_text("\n".join(L) + "\n")
print("\n".join(L))
