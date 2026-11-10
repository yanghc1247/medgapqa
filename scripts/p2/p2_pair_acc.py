"""D95: supplement to the pair comparisons (user instruction: include S0 accuracy and cost of removal in the pair differences). Uses Model, Boot and PAIRS of p2_analysis
unchanged (same shared resampling matrix) and computes, for medical fine-tune minus base, the differences in C0 S0 accuracy, C0 S1 accuracy and cost of removal (S0 - S1)
with item-level paired bootstrap 95% intervals. Descriptive; not a new hypothesis test.
python scripts/p2/p2_pair_acc.py --root runs/p2/eval --items runs/v5/evalk/evalk.jsonl --out runs/p2/analysis/p2_pair_acc.json --approved
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_analysis as A  # noqa: E402


def run(root, items_path):
    its = A.jl(items_path)
    order = [x["hash"] for x in its]; gold = {x["hash"]: x["answer_idx"] for x in its}
    present = [m for m in A.MAIN + A.LARGE if A.available(root, m)]
    bt = A.Boot(len(order))
    out = {"n_items": len(order), "B": A.B, "seed": A.SEED, "note": "D95 descriptive supplement; same resampling matrix as p2_analysis", "excluded": A.EXCLUDED, "pairs": {}}
    for a, b in A.PAIRS:
        if a not in present or b not in present:
            continue
        acc = {}
        for m in (a, b):
            M = A.Model(root, m, order)
            acc[m] = {s: M.correct("C0", A.ST[s], gold) for s in ("S0", "S1")}
        d = {"C0_S0": acc[a]["S0"] - acc[b]["S0"], "C0_S1": acc[a]["S1"] - acc[b]["S1"],
             "C0_cost_S0_minus_S1": (acc[a]["S0"] - acc[a]["S1"]) - (acc[b]["S0"] - acc[b]["S1"])}
        out["pairs"][f"{a}_vs_{b}"] = {"pair": [a, b], **{k: {"value": float(v.mean()), "ci95": bt.mean_ci(v)} for k, v in d.items()}}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--approved", action="store_true"); a = ap.parse_args()
    if not a.approved:
        sys.exit("EVAL aggregation requires --approved")
    rep = run(Path(a.root), a.items)
    Path(a.out).write_text(json.dumps(rep, indent=1)); print(f"wrote {a.out}: pairs {list(rep['pairs'])}")


if __name__ == "__main__":
    main()
