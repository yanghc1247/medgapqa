"""D131: rerun the paper's extra analyses on the vLLM Instant root (runs/p2/analysis_vllm/instant/root) with the same run-time setting as
scripts/p2/analysis_vllm.py (D130-m): p2_analysis.REUSED_PATHS and REUSED are emptied, since MedGemma-4B has a vLLM run. The analysis code is
imported, not edited. Outputs go to runs/p2/analysis_vllm_paper/instant/.
p2_probe_pairs checks each point AUROC against a reference p2_analysis.json read from its ROOT; ROOT is pointed at a folder whose
runs/p2/analysis/p2_analysis.json is the vLLM Instant analysis (FAKEROOT, built here), so the check compares against the vLLM values.
python scripts/paper/d131_rerun_extra.py {probe_strata|b3|probe_pairs|pre32b|think7|c0|c0_dsv4}
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts" / "p2")]
import p2_analysis as PA  # noqa: E402
PA.REUSED_PATHS = {}; PA.REUSED = set()
AV = ROOT / "runs" / "p2" / "analysis_vllm" / "instant"
R = str(AV / "root"); I = str(ROOT / "runs" / "v5" / "evalk" / "evalk.jsonl"); O = ROOT / "runs" / "p2" / "analysis_vllm_paper" / "instant"
THINK7 = ["qwen3-8b", "qwen3-14b", "meerkat2-14b", "qwen3-30b-a3b", "meerkat2-30b-a3b", "qwen3-32b", "meerkat2-32b"]


def main(job):
    O.mkdir(parents=True, exist_ok=True)
    if job == "probe_strata":
        import p2_probe_strata as M
        sys.argv = ["x", "--root", R, "--items", I, "--out", str(O / "p2_probe_strata.json")]; M.main()
    elif job == "b3":
        import p2_b3 as M
        sys.argv = ["x", "--root", R, "--items", I, "--out", str(O / "p2_b3.json")]; M.main()
    elif job == "probe_pairs":
        import p2_probe_pairs as M
        fr = ROOT / ".cache" / "d131_fakeroot"; (fr / "runs" / "p2" / "analysis").mkdir(parents=True, exist_ok=True); (fr / "tools" / "p2").mkdir(parents=True, exist_ok=True)
        for link, target in ((fr / "runs" / "p2" / "analysis" / "p2_analysis.json", AV / "p2_analysis.json"), (fr / "tools" / "p2" / "p2_analysis.py", ROOT / "scripts" / "p2" / "p2_analysis.py")):
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(target, link)
        M.ROOT = fr
        sys.argv = ["x", "--root", R, "--items", I, "--out", str(O / "p2_probe_pairs.json")]; M.main()
    elif job == "pre32b":
        rep = PA.run(R, I, models=[m for m in PA.MAIN + PA.LARGE if m != "meerkat2-32b"])
        (O / "p2_analysis_pre32b.json").write_text(json.dumps(rep, indent=1)); print("pre32b", list(rep["models"]))
    elif job in ("c0", "c0_dsv4"):  # D132: 7-model (c0) or DeepSeek-V4-Flash (c0_dsv4, D133) C0 accuracy and cost, Think minus Instant, paired over items
        import numpy as np
        its = PA.jl(I); order = [x["hash"] for x in its]; gold = {x["hash"]: x["answer_idx"] for x in its}; bt = PA.Boot(len(order))
        roots = {"instant": R, "think": str(ROOT / "runs" / "p2" / "analysis_vllm" / "think" / "root")}; v = {}
        for md, root in roots.items():
            cost, s1, s0 = [], [], []
            for m in (THINK7 if job == "c0" else ["dsv4-0731"]):
                M = PA.Model(root, m, order); a0 = M.correct("C0", PA.ST["S0"], gold); a1 = M.correct("C0", PA.ST["S1"], gold)
                cost.append(a0 - a1); s1.append(a1); s0.append(a0)
            v[md] = {"cost": np.mean(cost, axis=0), "s1": np.mean(s1, axis=0), "s0": np.mean(s0, axis=0)}
        out = {"meta": {"analysis": "D131: mean C0 accuracy and cost of removal, Think minus Instant, paired over items (shared p2_analysis.Boot matrix, B 10000, seed 20261016, 95%)", "models": THINK7 if job == "c0" else ["dsv4-0731"]}}
        for k in ("cost", "s1", "s0"):
            out[k] = {md: {"value": float(v[md][k].mean()), "ci95": bt.mean_ci(v[md][k])} for md in roots}
            d = v["think"][k] - v["instant"][k]; out[k]["diff"] = {"value": float(d.mean()), "ci95": bt.mean_ci(d)}
        (O.parent / ("think_vs_instant_c0.json" if job == "c0" else "think_vs_instant_c0_dsv4.json")).write_text(json.dumps(out, indent=1)); print(job, {k: round(out[k]["diff"]["value"], 4) for k in ("cost", "s1", "s0")})
    elif job == "think7":
        rep = PA.run(R, I, models=THINK7)
        (O / "p2_analysis_think7.json").write_text(json.dumps(rep, indent=1)); print("think7", list(rep["models"]))


if __name__ == "__main__":
    main(sys.argv[1])
