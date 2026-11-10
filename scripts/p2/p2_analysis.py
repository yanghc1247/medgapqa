"""D75 Phase 2 pre-registered analysis (plan section 5, following D74 and D74-r4). Validated on synthetic data only before any EVAL result was seen (scripts/p2/test_p2_analysis.py).
Runs on the real EVAL outputs (runs/p2/eval) only with --approved, after the user's approval (refused otherwise).

Input (--root, the run_eval.py output layout for each model): <root>/<model>/c0/<model>/C0/{S0,S1K,S2K}/trajectories.jsonl, c4 and c4f/<model>/C4/{S1K,S2K}/trajectories.jsonl,
p6/<model>/arms.jsonl, oracle.jsonl, np_probe.jsonl. Answers are read from --items (answer_idx); item order is the order of the items file.
Scoring: C0, C4, C4f = FCR (answer -> its letter; parse_fail -> G1 rescued letter, else A; abstain and unresolved -> forced_v5 letter), P6 = consolidated re-query letter,
oracle = argmax of the letter logits. States: S0, S1 (=S1K), S2 (=S2K).
Statistics: item-clustered paired bootstrap; the resampling matrix of analysis.Bootstrap (B 10,000, seed 20261016) is shared by all statistics. Two-sided 95% percentile intervals;
as in D38, the improvement decision uses the lower bound of the two-sided 97.5% interval (1.25-98.75 percentiles).
Output: --out JSON (key layout in docs/p2_analysis_output_design.md). Includes behavior and format measures besides accuracy and Delta.
python scripts/p2/p2_analysis.py --root runs/p2/eval --items runs/v5/evalk/evalk.jsonl --out runs/p2/analysis/p2_analysis.json --approved
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts" / "p2")]
from g1 import g1_letter  # noqa: E402

B, SEED = 10_000, 20261016
MAIN = ["qwen3-8b", "qwen3-14b", "meerkat2-14b", "medgemma-4b", "qwen3-30b-a3b", "meerkat2-30b-a3b", "qwen3-32b", "medgemma-27b", "meerkat2-32b"]  # small + mid Instant (D74, pool rule of D76)
EXCLUDED = {"medgemma-1.5-4b": "pilot format failure (D76)",
            "qwen3-235b-a22b": "bf16 needed more GPUs than were available within the schedule; partial outputs not analyzed (D95)"}  # excluded from the main mean and the analyses; only the replacement history is reported
REUSED = {"medgemma-4b"}  # D76: models whose existing EVAL-K outputs are reused (results already seen) -> sensitivity mean 'excluding reused models'
OLD_V5 = Path("$PROJECT_ROOT/runs/v5")  # D79: location of the reused outputs (main checkout, read only)
REUSED_PATHS = {"medgemma-4b": {  # D78, D79: Phase 2 cells -> existing outputs (mapping table in logs/p2_medgemma4b_reuse_check.md, internal). Originals are only read; G1 only at the scoring step (fcr_letter)
    **{("C0", st): f"eval_evalk/medgemma-4b/C0/{st}/trajectories.jsonl" for st in ("S0", "S1K", "S2K")},
    **{("C4", st): f"eval_evalk/medgemma-4b/C4/{st}/trajectories.jsonl" for st in ("S1K", "S2K")},
    **{("C4f", st): f"eval_evalk_c4f_v3/medgemma-4b/C4/{st}/trajectories.jsonl" for st in ("S1K", "S2K")},
    "p6": "eval_evalk_p6_v1/medgemma-4b/arms.jsonl", "oracle": "d56/b1_oracle/medgemma-4b.jsonl", "np": "d58/np_probe/medgemma-4b.jsonl"}}


def model_paths(root, m):
    """Input files per cell. Reused model: REUSED_PATHS (relative to OLD_V5); others: the run_eval.py output layout (<root>/<model>/...)."""
    if m in REUSED_PATHS:
        return {k: OLD_V5 / v for k, v in REUSED_PATHS[m].items()}
    d = Path(root) / m
    out = {("C0", st): d / "c0" / m / "C0" / st / "trajectories.jsonl" for st in ("S0", "S1K", "S2K")}
    for pol, sub in (("C4", "c4"), ("C4f", "c4f")):
        out.update({(pol, st): d / sub / m / "C4" / st / "trajectories.jsonl" for st in ("S1K", "S2K")})
    out.update(p6=d / "p6" / m / "arms.jsonl", oracle=d / "oracle.jsonl", np=d / "np_probe.jsonl", done=d / "DONE")
    return out


def available(root, m):
    P = model_paths(root, m)
    return P["done"].exists() if "done" in P else all(f.exists() for f in P.values())
LARGE = ["meerkat-70b-nf4", "qwen3-235b-a22b", "dsv4-0731"]  # reported per model
PAIRS = [("meerkat2-14b", "qwen3-14b"), ("meerkat2-30b-a3b", "qwen3-30b-a3b"), ("meerkat2-32b", "qwen3-32b")]  # (medical fine-tune, base); D94: 32B pair added (user instruction; computed only when both models are present)
POLICIES = ["C4", "C4f", "P6"]  # Free agent, Forced inquiry, Targeted ask pipeline
CRITERION_POLICIES = ["C4f", "P6"]
PAPER_NAME = {"C0": "C0 (answer only)", "C4": "Free agent", "C4f": "Forced inquiry", "P6": "Targeted ask pipeline"}
FIRST = ["ANSWER", "ASK", "SEARCH", "ABSTAIN", "other"]
FACTS = {"F1": {"F1"}, "F2": {"F2"}, "F1F2": {"F1", "F2"}}
ST = {"S0": "S0", "S1": "S1K", "S2": "S2K"}


# ---------- bootstrap ----------
class Boot:
    def __init__(self, n):
        self.idx = np.random.default_rng(SEED).integers(0, n, size=(B, n), dtype=np.int32)  # same matrix as analysis.Bootstrap

    def mean_ci(self, x, q=0.95):
        v = x[self.idx].mean(axis=1)
        return pct(v, q)

    def masked_mean_ci(self, x, mask, q=0.95):  # mean over the mask items only (the resampling matrix stays the one over all items)
        n, d = (x * mask)[self.idx].sum(axis=1), mask[self.idx].sum(axis=1)
        return pct(n[d > 0] / d[d > 0], q)

    def ratio(self, num, den, q=0.95):
        n, d = num[self.idx].sum(axis=1), den[self.idx].sum(axis=1)
        v = np.where(d > 0, n / np.where(d > 0, d, 1), np.nan)
        v = v[~np.isnan(v)]
        val = float(num.sum() / den.sum()) if den.sum() else None
        return {"value": val, "ci95": pct(v, q) if len(v) else None, "num": float(num.sum()), "den": float(den.sum())}


def pct(v, q):
    lo, hi = (1 - q) / 2 * 100, (1 + q) / 2 * 100
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


def auc(pos, neg):
    x = np.concatenate([pos, neg]); _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    ranks = ((np.cumsum(cnt) - cnt) + (cnt + 1) / 2)[inv]
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------- scoring and reading ----------
def fcr_letter(t):
    if t["outcome"] == "answer":
        return t["answer"]
    if t["outcome"] == "parse_fail":
        return g1_letter(t) or "A"
    return t["forced_v5"]["letter"]


def first_action(t):
    a = t["turns"][0].get("action") if t.get("turns") else None
    if a == "parse_fail" and len(t["turns"]) == 1 and g1_letter(t):
        a = "ANSWER"  # G1: if rescued on the first turn, the first action is also ANSWER
    return a if a in FIRST[:4] else "other"


def recv(sims):
    s = set()
    for x in sims:
        if isinstance(x, dict) and x.get("kind") == "provided":
            s |= FACTS.get(x.get("label"), set())
    return len(s)


def jl(p):
    return [json.loads(x) for x in open(p)]


class Model:
    def __init__(self, root, m, order):
        P = model_paths(root, m)
        self.m, self.order, self.reused = m, order, m in REUSED_PATHS
        self.traj = {}
        for st in ("S0", "S1K", "S2K"):
            self.traj[("C0", st)] = self._by_hash(P[("C0", st)])
        for pol in ("C4", "C4f"):
            for st in ("S1K", "S2K"):
                self.traj[(pol, st)] = self._by_hash(P[(pol, st)])
        rows = jl(P["p6"])
        self.p6 = {(r["hash"], r["state"]): r for r in rows}
        self.oracle = {r["hash"]: r for r in jl(P["oracle"])}
        self.np = {r["hash"]: r for r in jl(P["np"])} if P["np"].exists() else None

    def _by_hash(self, f):
        r = {t["hash"]: t for t in jl(f)}
        missing = [h for h in self.order if h not in r]
        if missing:
            raise SystemExit(f"{f}: {len(missing)} items missing (no partial analysis)")
        return r

    def correct(self, pol, st, gold):
        if pol == "P6":
            return np.array([self.p6[(h, st)]["arms"]["P6"]["letter"] == gold[h] for h in self.order], float)
        if pol.startswith("O:"):
            c = pol[2:]
            return np.array([self.oracle[h][c]["letter"] == gold[h] for h in self.order], float)
        T = self.traj[(pol, st)]
        return np.array([fcr_letter(T[h]) == gold[h] for h in self.order], float)

    def p6_valid(self, st):
        """Whether the P6 question set was generated validly (valid of the frozen pipeline_k.generate; if invalid, the consolidated re-query has 0 questions = the same prompt as oracle L-S1K)."""
        return np.array([bool(self.p6[(h, st)]["gens"]["Q3"]["valid"]) for h in self.order], float)

    def asks(self, pol, st):
        """List of ASK records per item (simulator dicts)."""
        if pol == "P6":
            return [self.p6[(h, st)]["arms"]["P6"]["asks"] for h in self.order]
        T = self.traj[(pol, st)]
        return [[x["sim"] for x in T[h]["turns"] if x.get("action") == "ASK" and isinstance(x.get("sim"), dict)] for h in self.order]


# ---------- analysis ----------
def per_model(M, gold, bt):
    out = {"acc": {}, "delta": {}, "behavior": {}, "acquisition": {}, "oracle": {}, "np_probe": None, "format": {}}
    acc = {("C0", s): M.correct("C0", ST[s], gold) for s in ("S0", "S1", "S2")}
    for pol in POLICIES:
        for s in ("S1", "S2"):
            acc[(pol, s)] = M.correct(pol, ST[s], gold)
    for (pol, s), v in acc.items():
        out["acc"][f"{pol}_{s}"] = {"value": float(v.mean()), "ci95": bt.mean_ci(v)}
    out["acc"]["C0_cost_S0_minus_S1"] = {"value": float((acc[("C0", "S0")] - acc[("C0", "S1")]).mean()), "ci95": bt.mean_ci(acc[("C0", "S0")] - acc[("C0", "S1")])}
    vec = {}
    for pol in POLICIES:
        for s in ("S1", "S2"):
            d = acc[(pol, s)] - acc[("C0", s)]; vec[(pol, s)] = d
            out["delta"][f"{pol}_{s}"] = {"value": float(d.mean()), "ci95": bt.mean_ci(d)}
    # behavior: immediate-answer share (first C4 action), first-action distribution, abstention rate
    for pol in ("C4", "C4f"):
        for st in ("S1K", "S2K"):
            T = M.traj[(pol, st)]
            fa = [first_action(T[h]) for h in M.order]
            ind = {a: np.array([x == a for x in fa], float) for a in FIRST}
            abst = np.array([T[h]["outcome"] == "abstain" for h in M.order], float)
            out["behavior"][f"{pol}_{st}"] = {"first_action": {a: {"value": float(v.mean()), "ci95": bt.mean_ci(v)} for a, v in ind.items()},
                                              "immediate_answer": {"value": float(ind["ANSWER"].mean()), "ci95": bt.mean_ci(ind["ANSWER"])},
                                              "abstain_rate": {"value": float(abst.mean()), "ci95": bt.mean_ci(abst)},
                                              "outcome": dict(Counter(T[h]["outcome"] for h in M.order))}
    # acquisition: provided/ASK (primary measure), patient-fact questions (simulator qtype patient_fact)/ASK (D77: named separately because it differs from the paper's 'valid questions'),
    # share of items receiving >= 1 fact. For P6 also the paper's 'valid questions' (share of items with a valid question set, step 1 of the D56 funnel).
    for pol in POLICIES:
        for st in ("S1K", "S2K"):
            A = M.asks(pol, st)
            n_ask = np.array([len(a) for a in A], float)
            prov = np.array([sum(isinstance(x, dict) and x.get("kind") == "provided" for x in a) for a in A], float)
            pf = np.array([sum(isinstance(x, dict) and x.get("qtype") == "patient_fact" for x in a) for a in A], float)
            r1 = np.array([recv(a) >= 1 for a in A], float)
            out["acquisition"][f"{pol}_{st}"] = {"asks_per_item": float(n_ask.mean()), "provided_per_ask": bt.ratio(prov, n_ask),
                                                 "patient_fact_per_ask": bt.ratio(pf, n_ask), "ge1_fact_received": {"value": float(r1.mean()), "ci95": bt.mean_ci(r1)},
                                                 "recv_dist": dict(Counter(int(recv(a)) for a in A))}
            if pol == "P6":
                v = M.p6_valid(st)
                out["acquisition"][f"{pol}_{st}"]["valid_gen_items"] = {"value": float(v.mean()), "ci95": bt.mean_ci(v), "n_invalid": int((1 - v).sum())}
    # D77 sensitivity: Delta without items whose P6 question set is invalid (per state, only that state's invalid items are excluded)
    out["sensitivity"] = {"P6_valid_only": {}}
    for s in ("S1", "S2"):
        v = M.p6_valid(ST[s]); d = vec[("P6", s)]
        out["sensitivity"]["P6_valid_only"][f"delta_{s}"] = {"value": float(d[v > 0].mean()), "ci95": bt.masked_mean_ci(d, v), "n_items": int(v.sum())}
    # oracle R (D56/D58 definition): R = (O2 - C0 S1)/(C0 S0 - C0 S1); R_L uses the letter logits L-S0 and L-S1K
    o = {c: M.correct("O:" + c, None, gold) for c in ("L-S0", "L-S1K", "O2", "O1a", "O1b")}
    out["oracle"] = {"R": ratio_R(o["O2"] - acc[("C0", "S1")], acc[("C0", "S0")] - acc[("C0", "S1")], bt),
                     "R_L": ratio_R(o["O2"] - o["L-S1K"], o["L-S0"] - o["L-S1K"], bt),
                     "acc": {c: float(v.mean()) for c, v in o.items()},
                     "O2_minus_C0S0": {"value": float((o["O2"] - acc[("C0", "S0")]).mean()), "ci95": bt.mean_ci(o["O2"] - acc[("C0", "S0")])}}
    if M.np is not None:
        s0 = np.array([M.np[h]["S0"]["logit_no"] - M.np[h]["S0"]["logit_yes"] for h in M.order])
        s1 = np.array([M.np[h]["S1"]["logit_no"] - M.np[h]["S1"]["logit_yes"] for h in M.order])
        bs = [auc(s1[i], s0[i]) for i in bt.idx]
        out["np_probe"] = {"AUROC": auc(s1, s0), "ci95": pct(np.array(bs), 0.95), "n_items": len(M.order)}
    out["format"] = format_metrics(M)
    return out, acc, vec


def ratio_R(num, den, bt):
    n, d = num[bt.idx].mean(axis=1), den[bt.idx].mean(axis=1)
    out = {"den": float(den.mean()), "den_ci95": pct(d, 0.95)}
    if out["den_ci95"][0] <= 0:
        return {**out, "R": None, "note": "not computed: denominator 95% lower bound <= 0"}
    return {**out, "R": float(num.mean() / den.mean()), "R_ci95": pct(n[d != 0] / d[d != 0], 0.95)}


def n_json_objects(text):
    dec, n, i = json.JSONDecoder(), 0, text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
            n += isinstance(obj, dict); i = text.find("{", end)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
    return n


def format_metrics(M):
    from g1 import RX
    out = {}
    for (pol, st), T in M.traj.items():
        raws, trunc, turns = [], 0, 0
        for h in M.order:
            for x in T[h]["turns"]:
                turns += 1
                trunc += bool(x.get("truncated")) or bool(x.get("retry_truncated"))
                raws += [x[k] for k in ("raw", "retry_raw") if x.get(k)]
        nobj = Counter(min(n_json_objects(r), 2) for r in raws)

        def letter_text(r):
            from format_check import parse_last_json
            f = parse_last_json(r); o = f[0] if f else {}
            c = o.get("content")
            return o.get("action") == "ANSWER" and isinstance(c, str) and c.strip().upper() not in list("ABCD") and bool(RX.match(c))
        pf = [T[h] for h in M.order if T[h]["outcome"] == "parse_fail"]
        out[f"{pol}_{st}"] = {"outputs": len(raws), "json_objects": {"0": nobj.get(0, 0), "1": nobj.get(1, 0), ">=2": nobj.get(2, 0)},
                              "truncated_turns": trunc, "turns": turns, "parse_fail_raw": len(pf), "g1_rescued": sum(g1_letter(t) is not None for t in pf),
                              "answer_letter_text_outputs": sum(letter_text(r) for r in raws),
                              "unresolved": sum(T[h]["outcome"] == "unresolved" for h in M.order)}
    gens = [g for r in M.p6.values() for g in r["gens"].values()]
    out["P6_gen"] = {"outputs": sum(bool(g.get(k)) for g in gens for k in ("raw", "retry_raw")), "invalid": sum(not g.get("valid") for g in gens)}
    return out


def mean_block(models, acc, vec, bt):
    """Main mean (item vector of the model mean): Delta per policy with 95% and 97.5% CIs, S2 change, improvement decision (C4f, P6)."""
    out = {"models": models, "n_models": len(models)}
    for pol in POLICIES:
        d1 = np.mean([vec[m][(pol, "S1")] for m in models], axis=0)
        d2 = np.mean([vec[m][(pol, "S2")] for m in models], axis=0)
        r = {"d_s1": float(d1.mean()), "d_s1_ci95": bt.mean_ci(d1, 0.95), "d_s1_ci975": bt.mean_ci(d1, 0.975),
             "d_s2": float(d2.mean()), "d_s2_ci95": bt.mean_ci(d2, 0.95)}
        if pol in CRITERION_POLICIES:
            r["criteria"] = {"d_s1>=0.05": r["d_s1"] >= 0.05, "ci975_low>0": r["d_s1_ci975"][0] > 0, "d_s2>=-0.02": r["d_s2"] >= -0.02}
            r["improved"] = all(r["criteria"].values())
        out[pol] = r
    c = np.mean([acc[m][("C0", "S0")] - acc[m][("C0", "S1")] for m in models], axis=0)
    out["C0_cost_S0_minus_S1"] = {"value": float(c.mean()), "ci95": bt.mean_ci(c)}
    return out


def p6_valid_only_mean(models, Ms, vec, bt):
    """D77 sensitivity: main-mean P6 Delta (S1 on items valid at S1K, S2 on items valid at S2K) and the decision conditions, using only items whose P6 question set is valid for every model in the pool."""
    out = {"models": models}
    for s in ("S1", "S2"):
        mask = np.prod([Ms[m].p6_valid(ST[s]) for m in models], axis=0)
        d = np.mean([vec[m][("P6", s)] for m in models], axis=0)
        out[f"d_{s.lower()}"] = float(d[mask > 0].mean()); out[f"d_{s.lower()}_ci95"] = bt.masked_mean_ci(d, mask, 0.95)
        if s == "S1":
            out["d_s1_ci975"] = bt.masked_mean_ci(d, mask, 0.975)
        out[f"n_items_{s.lower()}"] = int(mask.sum())
    out["criteria"] = {"d_s1>=0.05": out["d_s1"] >= 0.05, "ci975_low>0": out["d_s1_ci975"][0] > 0, "d_s2>=-0.02": out["d_s2"] >= -0.02}
    out["improved"] = all(out["criteria"].values())
    return out


def pair_block(a, b, Ms, acc, vec, bt):
    """Medical fine-tune (a) minus base (b). Item-level paired bootstrap."""
    out = {"pair": [a, b], "delta_diff": {}, "first_action_diff": {}, "abstain_rate_diff": {}, "patient_fact_per_ask_diff": {}, "provided_per_ask_diff": {}}
    for pol in POLICIES:
        for s in ("S1", "S2"):
            d = vec[a][(pol, s)] - vec[b][(pol, s)]
            out["delta_diff"][f"{pol}_{s}"] = {"value": float(d.mean()), "ci95": bt.mean_ci(d)}
    for pol in ("C4", "C4f"):
        Ta, Tb = Ms[a].traj[(pol, "S1K")], Ms[b].traj[(pol, "S1K")]
        fa = {k: np.array([first_action(Ta[h]) == k for h in Ms[a].order], float) for k in FIRST}
        fb = {k: np.array([first_action(Tb[h]) == k for h in Ms[b].order], float) for k in FIRST}
        out["first_action_diff"][pol] = {k: {"value": float((fa[k] - fb[k]).mean()), "ci95": bt.mean_ci(fa[k] - fb[k])} for k in FIRST}
        ab = np.array([Ta[h]["outcome"] == "abstain" for h in Ms[a].order], float) - np.array([Tb[h]["outcome"] == "abstain" for h in Ms[b].order], float)
        out["abstain_rate_diff"][pol] = {"value": float(ab.mean()), "ci95": bt.mean_ci(ab)}
    for pol in POLICIES:
        Aa, Ab = Ms[a].asks(pol, "S1K"), Ms[b].asks(pol, "S1K")
        for key, f in (("provided_per_ask_diff", lambda x: x.get("kind") == "provided"), ("patient_fact_per_ask_diff", lambda x: x.get("qtype") == "patient_fact")):
            na, nb = np.array([len(x) for x in Aa], float), np.array([len(x) for x in Ab], float)
            ka = np.array([sum(f(y) for y in x if isinstance(y, dict)) for x in Aa], float)
            kb = np.array([sum(f(y) for y in x if isinstance(y, dict)) for x in Ab], float)
            sa, sb = ka[bt.idx].sum(1), kb[bt.idx].sum(1); da, db = na[bt.idx].sum(1), nb[bt.idx].sum(1)
            ok = (da > 0) & (db > 0)
            diff = sa[ok] / da[ok] - sb[ok] / db[ok]
            val = (ka.sum() / na.sum() - kb.sum() / nb.sum()) if na.sum() and nb.sum() else None
            out[key][pol] = {"value": None if val is None else float(val), "ci95": pct(diff, 0.95) if len(diff) else None}
    return out


def run(root, items_path, models=None):
    its = jl(items_path)
    order = [x["hash"] for x in its]; gold = {x["hash"]: x["answer_idx"] for x in its}
    present = [m for m in MAIN + LARGE if available(root, m)]
    if models:
        present = [m for m in present if m in models]
    bt = Boot(len(order))
    Ms, rep, acc, vec = {}, {"n_items": len(order), "B": B, "seed": SEED, "models": {}, "paper_names": PAPER_NAME}, {}, {}
    for m in present:
        Ms[m] = Model(root, m, order)
        rep["models"][m], acc[m], vec[m] = per_model(Ms[m], gold, bt)
        rep["models"][m]["reused"] = Ms[m].reused  # D78: mark reused models
    main = [m for m in MAIN if m in present]
    rep["main_mean"] = mean_block(main, acc, vec, bt) if main else None
    rep["main_mean_missing"] = [m for m in MAIN if m not in present]
    rep["excluded"] = EXCLUDED
    rep["reused_models"] = [m for m in present if m in REUSED_PATHS]
    rep["sensitivity"] = {"P6_valid_only_main": p6_valid_only_mean(main, Ms, vec, bt) if main else None}
    reused = [m for m in main if m in REUSED]  # D76: only when a reused model is in the pool
    rep["sensitivity"]["main_mean_excluding_reused"] = (mean_block([m for m in main if m not in REUSED], acc, vec, bt) | {"excluded_reused": reused}) if reused else None
    rep["large"] = [m for m in LARGE if m in present]
    rep["pairs"] = {f"{a}_vs_{b}": pair_block(a, b, Ms, acc, vec, bt) for a, b in PAIRS if a in present and b in present}
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--approved", action="store_true"); a = ap.parse_args()
    if Path(a.root).resolve() == (ROOT / "runs" / "p2" / "eval").resolve() and not a.approved:
        sys.exit("refusing to analyze EVAL outputs without --approved (D75: run only after user approval)")
    rep = run(a.root, a.items)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"wrote {a.out}: models {list(rep['models'])}")


if __name__ == "__main__":
    main()
