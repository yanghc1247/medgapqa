"""EVAL analysis (draft, for user review before freeze 2). Plan §6 metrics and §7 statistics.

Input: {runs}/{model}/{cond}/{state}/trajectories*.jsonl (shards are merged), item file (data/perturbed_{split}.jsonl).
Output: {out}/analysis.json — outcome category shares for every model, condition and state, and the pre-specified comparisons (COMPARISONS).

Principles
- Gold labels (answer_idx) are separated in split_gold and passed only to grade(). They are never used for item selection, stratification or filtering.
- Outcome categories (one per item): correct = ANSWER with the gold letter, wrong = ANSWER with another letter,
  abstain = ended with ABSTAIN, unresolved = ended by exhausting the maximum turns T or exceeding the ASK/SEARCH repetition limit (forced answers ignored),
  parse_fail = no valid action even after one format re-request.
- CorrectResolution = correct / n (n = all items run for that model, condition and state; nothing is removed from the denominator).
  CorrectResolution is never reported alone: the five category shares and conditional accuracy = correct / (correct + wrong) are given in every cell.
- CI: item-level cluster bootstrap. Item indices are resampled once and applied to every model, condition and state (pairs kept),
  B = 10,000, fixed seed, two-sided 95% percentile. No p-values.
- No single conclusion pooling models (e.g. mean Δ over models) is defined. Every comparison is reported per model.
- Only comparisons pre-specified in COMPARISONS are computed. Analyses added after the freeze are registered with kind="exploratory"
  and marked "exploratory" in the output.
- EVAL runs only if the freeze-2 tag (freeze-v0.4) exists (CLAUDE.md: no EVAL aggregation before the freeze).

Usage: python src/medgapqa/analysis.py --split dev --runs runs/xxx --out results/dev_xxx [--necessity file.jsonl]
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FREEZE_TAG = "freeze-v0.4.2"  # D15 re-freeze
MODELS = ["qwen2.5-7b", "olmo2-7b", "meerkat-8b", "medgemma-4b"]  # medgemma-4b: deviations D15
CONDS = ["C0", "C1", "C2", "C3", "C4"]
STATES = ["S0", "S1", "S2"]
NO_SIM_CONDS = ("C0", "C1")  # no simulator is used, so one PERT run (S1) serves both S1 and S2
OUTCOMES = ["correct", "wrong", "abstain", "unresolved", "parse_fail"]
FIRST = ["ANSWER", "ASK", "SEARCH", "ABSTAIN", "parse_fail"]
REMOVED_TYPES = ["history", "physical_exam", "lab_imaging", "demographic"]
B = 10_000
SEED = 20261016  # proposed values (before user approval)
GUARD_DROP = 0.02  # §7 guard: if C4 is 2 points or more below C0 at S0
GUARD_OVER = 0.10  # §7 guard: if the S0 over-intervention rate exceeds 10%


# ---------- input ----------
def tag_exists(tag):
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
                          capture_output=True).returncode == 0


def require_freeze(split, tag_exists=tag_exists):
    if split == "eval" and not tag_exists(FREEZE_TAG):
        raise SystemExit(f"EVAL analysis is allowed only after freeze-2 tag {FREEZE_TAG}")


def split_gold(items):
    """(Item metadata without the answer, {hash: answer}). The answer is used only in grade()."""
    meta = {it["hash"]: {k: v for k, v in it.items() if k != "answer_idx"} for it in items}
    return meta, {it["hash"]: it["answer_idx"] for it in items}


def load_runs(root, models, conds, states):
    runs = {}
    for m in models:
        for c in conds:
            for s in states:
                src = "S1" if (c in NO_SIM_CONDS and s == "S2") else s
                d = Path(root) / m / c / src
                files = sorted(d.glob("trajectories*.jsonl")) if d.exists() else []
                if not files:
                    continue
                trs = {}
                for f in files:
                    for line in f.read_text().splitlines():
                        t = json.loads(line)
                        assert t["hash"] not in trs, f"duplicate {t['hash']} in {d}"
                        trs[t["hash"]] = t
                runs[(m, c, s)] = trs
    return runs


# ---------- scoring and summary ----------
def grade(t, gold):
    if t["outcome"] == "answer":
        return "correct" if t["answer"] == gold[t["hash"]] else "wrong"
    return t["outcome"]  # abstain / unresolved / parse_fail. forced_answer is not included in the main metric


def _div(a, b):
    return a / b if b else None


def summarize(trs, gold):
    n = len(trs)
    cat = Counter(grade(t, gold) for t in trs)
    first = Counter(t["first_action"] for t in trs)
    answered = cat["correct"] + cat["wrong"]
    return {
        "n": n,
        "count": {o: cat[o] for o in OUTCOMES},
        "rate": {o: cat[o] / n for o in OUTCOMES},
        "correct_resolution": cat["correct"] / n,
        "coverage": answered / n,
        "conditional_accuracy": _div(cat["correct"], answered),
        "selective_risk": _div(cat["wrong"], answered),
        "first_action": {a: first[a] / n for a in FIRST},
        "abstain_timing": {"first": sum(abstain_timing(t) == "first" for t in trs) / n,
                           "after_tool": sum(abstain_timing(t) == "after_tool" for t in trs) / n,
                           "after_tool_last_turn": sum(map(abstain_at_last_turn, trs)) / n},
        "unresolved_cause": dict(Counter(t.get("unresolved_cause") or t.get("end_reason") for t in trs
                                         if t["outcome"] == "unresolved")),
        "budget_exhausted": (None if all(t.get("budget_exhausted") is None for t in trs)
                             else sum(bool(t.get("budget_exhausted")) for t in trs) / n),
        "over_intervention": {**{a: first[a] / n for a in ("ASK", "SEARCH", "ABSTAIN")},
                              "total": sum(first[a] for a in ("ASK", "SEARCH", "ABSTAIN")) / n},
        "cost": {"mean_turns": float(np.mean([t["n_turns"] for t in trs])),
                 "mean_tokens_in": float(np.mean([t["tokens"]["in"] for t in trs])),
                 "mean_tokens_out": float(np.mean([t["tokens"]["out"] for t in trs])),
                 "action_use": {a: float(np.mean([t["action_counts"][a] > 0 for t in trs])) for a in ("ASK", "SEARCH")}},
    }


def abstain_timing(t):
    """When an ABSTAIN ending happened: 'first' (first action) / 'after_tool' (after ASK or SEARCH, including the last turn). None if not ABSTAIN."""
    if t["outcome"] != "abstain":
        return None
    return "first" if t["first_action"] == "ABSTAIN" else "after_tool"


def abstain_at_last_turn(t):
    """Did an ABSTAIN after tool use occur on the last turn (T, the turn where v2 allows only ANSWER and ABSTAIN)? False if T is not recorded (v1)."""
    return abstain_timing(t) == "after_tool" and t.get("T") is not None and t["n_turns"] == t["T"]


def info_summary(trs):
    """S1 and S2 information metrics (separate from correctness): items where the simulator gave the removed fact, items answered without the information."""
    def provided(t):
        return any(x.get("sim", {}).get("kind") == "provided" for x in t["turns"])
    n = len(trs)
    return {"info_acquired": sum(map(provided, trs)) / n,
            "answered_without_info": sum(t["outcome"] == "answer" and not provided(t) for t in trs) / n}


def after_first_ask(t):
    """Next action after the first ASK reply (unit of the §3.3 S1 vs S2 comparison). None if there is no ASK; 'none' if the episode ended with no further action."""
    acts = [x["action"] for x in t["turns"]]
    if "ASK" not in acts:
        return None
    k = acts.index("ASK")
    return acts[k + 1] if k + 1 < len(acts) else "none"


# ---------- bootstrap ----------
class Bootstrap:
    """Builds the item-index resampling matrix once and uses it for every statistic (item-level cluster, pairs kept)."""

    def __init__(self, n_items, B=B, seed=SEED):
        self.idx = np.random.default_rng(seed).integers(0, n_items, size=(B, n_items), dtype=np.int32)

    def ci(self, stat):
        vals = np.array([stat(i) for i in self.idx], dtype=float)
        vals = vals[~np.isnan(vals)]
        if not len(vals):
            return None
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta(a, b, boot):
    """a, b: per-item values (0/1) in the same item order. Mean difference and 95% CI. Not computed if either run is missing items."""
    missing = int(np.isnan(a).sum() + np.isnan(b).sum())
    if missing:
        return {"missing_cells": missing, "n": len(a)}
    return {"delta": float(a.mean() - b.mean()), "ci95": boot.ci(lambda i: a[i].mean() - b[i].mean()), "n": len(a)}


def ratio_ci(num, den, boot):
    def f(i):
        d = den[i].sum()
        return num[i].sum() / d if d else np.nan
    return {"value": _div(float(num.sum()), float(den.sum())), "ci95": boot.ci(f), "n_den": int(den.sum())}


# ---------- pre-specified comparisons ----------
class Ctx:
    def __init__(self, runs, gold, meta, order, necessity=None):
        self.runs, self.gold, self.meta, self.order, self.necessity = runs, gold, meta, order, necessity
        self.pos = {h: k for k, h in enumerate(order)}
        self.boot = Bootstrap(len(order))

    def vec(self, m, c, s, fn):
        """Array of fn(trajectory) in item order (order). Items missing from the run are NaN (does not happen for a full run)."""
        r = self.runs.get((m, c, s), {})
        return np.array([fn(r[h]) if h in r else np.nan for h in self.order], dtype=float)

    def correct(self, m, c, s):
        return self.vec(m, c, s, lambda t: grade(t, self.gold) == "correct")


def cmp_primary(ctx):
    """Main analysis: at S1, CorrectResolution(C4) − CorrectResolution(C0), per model."""
    return {m: paired_delta(ctx.correct(m, "C4", "S1"), ctx.correct(m, "C0", "S1"), ctx.boot) for m in MODELS}


def cmp_s1_vs_s2(ctx):
    """S1 − S2 contrast (C2, C4): CorrectResolution difference and the distribution of the next action after the first ASK reply."""
    out = {}
    for m in MODELS:
        for c in ("C2", "C4"):
            d = paired_delta(ctx.correct(m, c, "S1"), ctx.correct(m, c, "S2"), ctx.boot)
            trans = {}
            for s in ("S1", "S2"):
                xs = [after_first_ask(t) for t in ctx.runs.get((m, c, s), {}).values()]
                xs = [x for x in xs if x is not None]
                trans[s] = {"n_with_ask": len(xs), **{a: _div(xs.count(a), len(xs)) for a in
                                                       ("ANSWER", "ASK", "SEARCH", "ABSTAIN", "parse_fail", "none")}}
            out[f"{m}/{c}"] = {"cr_delta": d, "after_first_ask": trans}
    return out


def cmp_s0_over_intervention(ctx):
    """S0 over-intervention: shares of first actions ASK / SEARCH / ABSTAIN, separately, per model for C1–C4, 95% CI."""
    out = {}
    for m in MODELS:
        for c in ("C1", "C2", "C3", "C4"):
            row = {}
            for a in ("ASK", "SEARCH", "ABSTAIN"):
                v = ctx.vec(m, c, "S0", lambda t, a=a: t["first_action"] == a)
                if np.isnan(v).all():
                    continue
                row[a] = {"rate": float(np.nanmean(v)), "ci95": ctx.boot.ci(lambda i, v=v: np.nanmean(v[i]))}
            v = ctx.vec(m, c, "S0", lambda t: t["first_action"] in ("ASK", "SEARCH", "ABSTAIN"))
            if np.isnan(v).all():
                continue
            row["total"] = {"rate": float(np.nanmean(v)), "ci95": ctx.boot.ci(lambda i: np.nanmean(v[i]))}
            out[f"{m}/{c}"] = row
    return out


def cmp_s0_guard(ctx):
    """Guardrail: at S0, CorrectResolution(C4) − CorrectResolution(C0). If the drop is ≥ 2 points or C4 over-intervention > 10%,
    it is not described as 'an improvement within the pre-specified S0 performance and intervention bounds' (point estimates, CI shown)."""
    out = {}
    for m in MODELS:
        d = paired_delta(ctx.correct(m, "C4", "S0"), ctx.correct(m, "C0", "S0"), ctx.boot)
        over = float(np.nanmean(ctx.vec(m, "C4", "S0", lambda t: t["first_action"] in ("ASK", "SEARCH", "ABSTAIN"))))
        out[m] = {"cr_delta": d, "over_intervention_C4": over,
                  "guard_failed": None if "delta" not in d else bool(d["delta"] <= -GUARD_DROP or over > GUARD_OVER)}
    return out


def _stratified_primary(ctx, groups):
    out = {}
    for g, hs in groups.items():
        sel = np.array([h in hs for h in ctx.order])
        if not sel.any():
            continue
        pos = np.flatnonzero(sel)
        sub = Bootstrap(len(pos))  # resample items within the stratum (same seed)
        out[g] = {m: paired_delta(ctx.correct(m, "C4", "S1")[pos], ctx.correct(m, "C0", "S1")[pos], sub)
                  for m in MODELS}
    return out


def cmp_removed_type(ctx):
    """Main-analysis Δ by removed-fact type (per model)."""
    return _stratified_primary(ctx, {t: {h for h, x in ctx.meta.items() if x["removed_type"] == t} for t in REMOVED_TYPES})


def cmp_necessity_quartile(ctx):
    """[Exploratory] Main-analysis Δ by quartile of the post hoc necessity score (drop in the generator's own answer-letter probability; gold labels not used)."""
    if not ctx.necessity:
        return {"skipped": "no necessity score file"}
    hs = [h for h in ctx.order if h in ctx.necessity]
    q = np.quantile([ctx.necessity[h] for h in hs], [0.25, 0.5, 0.75])
    groups = {f"Q{k + 1}": set() for k in range(4)}
    for h in hs:
        groups[f"Q{int(np.searchsorted(q, ctx.necessity[h], side='right')) + 1}"].add(h)
    return {"cuts": q.tolist(), **_stratified_primary(ctx, groups)}


LETTERS = ["A", "B", "C", "D"]


def gold_letter_distribution(gold):
    """(Position bias, descriptive statistics) Distribution of the gold letter in EVAL items. Not used for design or selection."""
    c = Counter(gold.values())
    return {k: c[k] / len(gold) for k in LETTERS}


def chosen_letter_distribution(trs):
    """Distribution of chosen letters in trajectories that ended with ANSWER (over answered episodes)."""
    c = Counter(t["answer"] for t in trs if t["outcome"] == "answer")
    n = sum(c.values())
    return {k: (c[k] / n if n else None) for k in LETTERS}


def acc_by_gold_position(trs, gold):
    """CorrectResolution by gold position (A–D) (denominator = all items whose answer is at that position)."""
    out = {}
    for k in LETTERS:
        sub = [t for t in trs if gold[t["hash"]] == k]
        out[k] = {"n": len(sub), "correct_resolution": (sum(grade(t, gold) == "correct" for t in sub) / len(sub)
                                                        if sub else None)}
    return out


def position_bias(ctx):
    """Position-bias descriptive statistics (not a design change, deviations D12): gold-letter distribution, chosen-letter distribution per cell,
    CorrectResolution by gold position for C0, C4 × S0, S1."""
    return {"gold_letters": gold_letter_distribution(ctx.gold),
            "chosen_letters": {f"{m}/{c}/{s}": chosen_letter_distribution(list(r.values()))
                               for (m, c, s), r in ctx.runs.items()},
            "acc_by_gold_position": {f"{m}/{c}/{s}": acc_by_gold_position(list(ctx.runs[(m, c, s)].values()), ctx.gold)
                                     for m in MODELS for c in ("C0", "C4") for s in ("S0", "S1")
                                     if (m, c, s) in ctx.runs}}


def cmp_position_delta(ctx):
    """[Exploratory] Main-analysis Δ_S1 by gold position (A–D), per model. Not a main analysis."""
    return _stratified_primary(ctx, {k: {h for h, g in ctx.gold.items() if g == k} for k in LETTERS})


def cmp_abstain_c4_vs_c1(ctx):
    """ABSTAIN rate: C4 − C1, by model and state (95% CI). Also gives the timing breakdown of C4 ABSTAIN (first action / after a tool / of these, last turn)."""
    out = {}
    for m in MODELS:
        for s in STATES:
            a4 = ctx.vec(m, "C4", s, lambda t: t["outcome"] == "abstain")
            a1 = ctx.vec(m, "C1", s, lambda t: t["outcome"] == "abstain")
            r = ctx.runs.get((m, "C4", s), {})
            n = len(r) or None
            out[f"{m}/{s}"] = {"delta_C4_minus_C1": paired_delta(a4, a1, ctx.boot),
                               "C4_timing": None if not n else {
                                   "first": sum(abstain_timing(t) == "first" for t in r.values()) / n,
                                   "after_tool": sum(abstain_timing(t) == "after_tool" for t in r.values()) / n,
                                   "after_tool_last_turn": sum(map(abstain_at_last_turn, r.values())) / n}}
    return out


def transition_table(c0, c4, gold):
    """Count table (5×5) of C0 outcome (rows) × C4 outcome (columns) for the same items. Only items present in both runs."""
    tab = {a: {b: 0 for b in OUTCOMES} for a in OUTCOMES}
    for h in c0.keys() & c4.keys():
        tab[grade(c0[h], gold)][grade(c4[h], gold)] += 1
    return tab


def cmp_transition(ctx):
    """Paired transition table: for S0 and S1 each, per model, 5×5 counts of C0 × C4 outcomes."""
    return {f"{m}/{s}": transition_table(ctx.runs.get((m, "C0", s), {}), ctx.runs.get((m, "C4", s), {}), ctx.gold)
            for m in MODELS for s in ("S0", "S1")}


COMPARISONS = [
    {"id": "primary_delta_S1_C4_vs_C0", "kind": "primary", "fn": cmp_primary},
    {"id": "s1_minus_s2_C2_C4", "kind": "secondary", "fn": cmp_s1_vs_s2},
    {"id": "s0_over_intervention_by_first_action", "kind": "secondary", "fn": cmp_s0_over_intervention},
    {"id": "s0_guard_C4_vs_C0", "kind": "secondary", "fn": cmp_s0_guard},
    {"id": "removed_type_delta_S1", "kind": "secondary", "fn": cmp_removed_type},
    {"id": "abstain_rate_C4_vs_C1", "kind": "secondary", "fn": cmp_abstain_c4_vs_c1},
    {"id": "c0_x_c4_transition_S0_S1", "kind": "secondary", "fn": cmp_transition},
    {"id": "necessity_quartile_delta_S1", "kind": "exploratory", "fn": cmp_necessity_quartile},
    {"id": "position_delta_S1", "kind": "exploratory", "fn": cmp_position_delta},
]


def search_intent_table(judgments, verified=None):
    """SEARCH intent table (per model, condition and state, deduplicated by judgment key). Distinguishes 'judge pfs' (upper bound including false positives)
    from 'verified pfs'. verified: {cell: share} built from human labels of the EVAL 150 sample — None before labeling
    (deviations L8: judge pfs and fact shares are reported only as upper bounds and are replaced once human validation results arrive)."""
    seen, out = set(), {}
    for r in judgments:
        if r["kind"] != "SEARCH":
            continue
        key = (r["model"], r["cond"], r["state"], r["hash"], r["norm_content"])
        if key in seen:
            continue
        seen.add(key)
        cell = f"{r['model']}/{r['cond']}/{r['state']}"
        t = out.setdefault(cell, {"n": 0, "judge_parse_fail": 0,
                                  "judge": {k: 0 for k in ("patient_fact_search", "general_knowledge", "mixed_or_unclear")},
                                  "judge_targets": {k: 0 for k in ("fact", "topic", "none")}})
        if r["status"] != "ok":
            t["judge_parse_fail"] += 1
            continue
        t["n"] += 1
        t["judge"][r["label"]["intent"]] += 1
        t["judge_targets"][r["label"]["targets_removed_fact"]] += 1
    for cell, t in out.items():
        t["judge_pfs_rate_upper_bound"] = _div(t["judge"]["patient_fact_search"], t["n"])
        t["judge_fact_rate_upper_bound"] = _div(t["judge_targets"]["fact"], t["n"])
        t["verified_pfs_rate"] = (verified or {}).get(cell)
        t["note"] = ("judge pfs and fact rates are upper bounds that include false positives (DEV LLM label agreement pfs 0/19, fact 0/12). "
                     "Validated values are to be filled from the EVAL 150-sample labels")
    return out


def first_action_s0_vs_pert(runs):
    """§3.3: first actions are compared only S0 vs PERT (S1 and S2 are identical until the first ASK reply, so PERT uses the S1 runs)."""
    out = {}
    for (m, c, s), r in runs.items():
        if s in ("S0", "S1") and c not in ("C0",):
            first = Counter(t["first_action"] for t in r.values())
            out[f"{m}/{c}/{'S0' if s == 'S0' else 'PERT'}"] = {a: first[a] / len(r) for a in FIRST}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "eval"], required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--items", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--necessity", default=None, help="jsonl {hash, score} (exploratory quartile strata)")
    ap.add_argument("--judgments", default=None, help="judge output directory (judgments*.jsonl): SEARCH intent table")
    a = ap.parse_args()
    require_freeze(a.split)
    items = [json.loads(x) for x in open(a.items or ROOT / "data" / f"perturbed_{a.split}.jsonl")]
    meta, gold = split_gold(items)
    order = [it["hash"] for it in items]
    runs = load_runs(a.runs, MODELS, CONDS, STATES)
    nec = {x["hash"]: x["score"] for x in map(json.loads, open(a.necessity))} if a.necessity else None
    ctx = Ctx(runs, gold, meta, order, nec)
    cells = {}
    for (m, c, s), r in runs.items():
        trs = list(r.values())
        cells[f"{m}/{c}/{s}"] = summarize(trs, gold)
        cr = ctx.correct(m, c, s)
        ans = ctx.vec(m, c, s, lambda t: grade(t, gold) in ("correct", "wrong"))
        cells[f"{m}/{c}/{s}"]["ci95"] = {"correct_resolution": ctx.boot.ci(lambda i: np.nanmean(cr[i])),
                                         "conditional_accuracy": ratio_ci(np.nan_to_num(cr), np.nan_to_num(ans), ctx.boot)["ci95"]}
        if s in ("S1", "S2") and c not in NO_SIM_CONDS:
            cells[f"{m}/{c}/{s}"]["info"] = info_summary(trs)
    report = {"split": a.split, "n_items": len(order), "bootstrap": {"B": B, "seed": SEED},
              "cells": cells, "first_action_S0_vs_PERT": first_action_s0_vs_pert(runs),
              "position_bias": position_bias(ctx),
              "comparisons": [{"id": c["id"], "kind": c["kind"],
                               "label": "exploratory" if c["kind"] == "exploratory" else c["kind"],
                               "result": c["fn"](ctx)} for c in COMPARISONS]}
    if a.judgments:
        js = [json.loads(x) for f in sorted(Path(a.judgments).glob("judgments*.jsonl")) for x in open(f)]
        report["search_intent"] = search_intent_table(js)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "analysis.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {Path(a.out) / 'analysis.json'}")


if __name__ == "__main__":
    main()
