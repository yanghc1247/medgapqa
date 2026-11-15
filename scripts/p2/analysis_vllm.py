"""D130 4 / D130-m: analysis of the vLLM results (the analysis code p2_analysis, p2_strata, p2_strata_recovery, p2_modelcorr, p2_pair_acc and p2_probe_pairs is imported unchanged; only run-time settings change).
Evaluation root per mode (symbolic-link tree) runs/p2/analysis_vllm/<mode>/root/:
  instant: runs/p2/eval_vllm_instant/<model> for the 9 main-pool models + for the large Meerkat-70B and DeepSeek-V4-Flash the transformers results (runs/p2/eval_cap1024/<model>, 1,024; flag engine=transformers).
  think: runs/p2/eval_vllm_think/<model> (only those with DONE). No NP probe (run in Instant only).
MedGemma-4B has a vLLM run, so p2_analysis.REUSED_PATHS (the old transformers reuse paths) and REUSED are emptied.
The preregistered improvement criterion is applied to the Instant main mean only (the Think criterion fields are reported only, not a judgment).
With --provisional, the output meta and file names carry a provisional mark (jobs without DONE are left out).
python scripts/p2/analysis_vllm.py --provisional
"""
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts"), str(ROOT / "scripts" / "p2"), str(ROOT / "scripts" / "d56"), str(ROOT / "scripts" / "d58")]
P2 = ROOT / "runs" / "p2"
ITEMS = ROOT / "runs" / "v5" / "evalk" / "evalk.jsonl"
HF_LARGE = {"meerkat-70b-nf4": "transformers NF4 (vLLM 0.32 has no bitsandbytes; D130-d)", "dsv4-0731": "transformers FP8 (vLLM TP2 infeasible; D130-e r1)"}


def build_root(mode, out):
    r = out / "root"; r.mkdir(parents=True, exist_ok=True)
    src = P2 / f"eval_vllm_{mode}"
    have = {}
    for d in sorted(src.iterdir()):
        if d.is_dir() and d.name != "p6" and (d / "DONE").exists():
            link = r / d.name
            if link.is_symlink() and Path(os.readlink(link)) != d.resolve():
                link.unlink()
            if not link.exists():
                os.symlink(d.resolve(), link)
            have[d.name] = "vllm" + (" (TP4; batch invariant off)" if d.name == "dsv4-0731" else "")
    if mode == "instant":
        for m, note in HF_LARGE.items():
            if m in have:  # D130-t: a large model with a finished vLLM run (DeepSeek) uses the vLLM values (the transformers values serve the engine check)
                continue
            link = r / m
            srcm = P2 / "eval_hf_nocap" / m if (P2 / "eval_hf_nocap" / m).exists() else P2 / "eval_cap1024" / m  # D130-o 2: the uncapped continuation takes precedence
            if link.is_symlink() and Path(os.readlink(link)) != srcm.resolve():
                link.unlink()
            if not link.exists():
                os.symlink(srcm.resolve(), link)
            have[m] = note + ("; cap-hit episodes rerun without cap (D130-o 2)" if (P2 / "eval_hf_nocap" / m / "D130O_NOCAP.json").exists() else "")
    return r, have


def run(mode, provisional):
    import p2_analysis as PA
    PA.REUSED_PATHS = {}; PA.REUSED = set()
    out = P2 / "analysis_vllm" / mode; out.mkdir(parents=True, exist_ok=True)
    root, have = build_root(mode, out)
    tag = "_provisional" if provisional else ""
    meta = {"mode": mode, "engine_by_model": have, "provisional": provisional, "items": str(ITEMS),
            "note": "D130 vLLM analysis; preregistered improvement criterion applies to the Instant main mean only" + ("; Think criterion fields are descriptive, not a judgment" if mode == "think" else "")}
    results = {}
    import p2_strata as PS, p2_strata_recovery as PR, p2_modelcorr as PC, p2_pair_acc as PP
    jobs = [("p2_analysis", lambda: PA.run(str(root), str(ITEMS)) if hasattr(PA, "run") else None)]
    # call each script's main with changed arguments only (--approved, --root = our root)
    for name, mod in (("p2_analysis", PA), ("p2_strata", PS), ("p2_strata_recovery", PR), ("p2_modelcorr", PC), ("p2_pair_acc", PP)):
        o = out / f"{name}{tag}.json"
        sys.argv = [name, "--root", str(root), "--items", str(ITEMS), "--out", str(o), "--approved"]
        try:
            mod.main()
            d = json.loads(o.read_text()); d["_vllm_meta"] = meta; o.write_text(json.dumps(d, indent=1, default=float))
            results[name] = "ok"
        except SystemExit as e:
            results[name] = f"exit {e}"
        except Exception as e:  # noqa: BLE001
            results[name] = f"error {repr(e)[:200]}"
    (out / f"run{tag}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print(mode, results, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--provisional", action="store_true"); ap.add_argument("--modes", default="instant,think"); a = ap.parse_args()
    for mode in a.modes.split(","):
        run(mode, a.provisional)


if __name__ == "__main__":
    main()
