"""D26b C4 reference (p3, protocol v2) rerun on DEV-K (new file; frozen files only imported).

Adapter SimKAdapter: connects respond(mode, visible, withheld, question), called by agent.run_episode, to simulator_k.SimulatorK.respond (2 facts).
State names s0→s0, recoverable→s1k, unrecoverable→s2k. The fact list is looked up by item id (the withheld string passed by run_episode is not used).
States: S0 (run_episode "S0"), S1K (run_episode "S1" = recoverable), S2K (run_episode "S2" = unrecoverable). Visible item: original for S0, rewritten for S1K and S2K.
Forced answer for FCR: abstain and unresolved trajectories go straight to forced_answer.forced_messages (D18a fixed sentence, reconstruction sha checked) + A–D restricted logits.

CUDA_VISIBLE_DEVICES=4 python src/medgapqa/c4_ref_k.py run --model qwen2.5-7b --sim-ports 8704,8705,8706,8707
python src/medgapqa/c4_ref_k.py report
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/medgapqa"))
from agent import load_config, run_episode  # noqa: E402

ITEMS = ROOT / "runs" / "v5" / "devk" / "devk.jsonl"
OUT = ROOT / "runs" / "v5" / "eval_devk"
MODE = {"s0": "s0", "recoverable": "s1k", "unrecoverable": "s2k"}
STATE = {"S0": "S0", "S1K": "S1", "S2K": "S2"}
MAIN = ["qwen2.5-7b", "meerkat-8b", "medgemma-4b", "qwen2.5-32b"]


class SimKAdapter:
    def __init__(self, simk, facts_by_id):
        self.simk, self.facts_by_id, self.item_id = simk, facts_by_id, None

    def respond(self, mode, visible, withheld, question):
        return self.simk.respond(MODE[mode], visible, self.facts_by_id[self.item_id], question)


def cmd_run(a):
    from budget import Budget
    from forced_answer import forced_messages
    from instances import PooledBackend
    from llm_client import RemoteGenerator
    from pipeline_k import EvalModel, inst_id
    from retrieve import Retriever
    from run_agent import load_tok
    from simulator_k import SimulatorK
    cfg = load_config(allow_provisional=False)
    items = [json.loads(x) for x in open(ITEMS)]
    m = EvalModel(a.model)
    ok, how = m.verify()
    if not ok:
        sys.exit(f"instance check failed for {a.model} ({how})")
    ports = [int(p) for p in a.sim_ports.split(",")]
    simk = SimulatorK(PooledBackend([RemoteGenerator(p) for p in ports], [f"sim:{p}" for p in ports]))
    sim = SimKAdapter(simk, {it["id"]: it["removed_facts"] for it in items})
    retriever = Retriever.load(cfg["search_top_k"], cfg["search_max_tokens"], load_tok(cfg["search_truncation_tokenizer"]))
    budget, inst = Budget("dev"), inst_id(a.model)
    print(f"{a.model} instance ok ({how})", flush=True)
    for st, rs in STATE.items():
        d = OUT / a.model / "C4" / st
        d.mkdir(parents=True, exist_ok=True)
        f = d / "trajectories.jsonl"
        done = {json.loads(x)["id"] for x in open(f)} if f.exists() else set()
        with open(f, "a") as fo:
            for it in items:
                if it["id"] in done:
                    continue
                sim.item_id = it["id"]
                ep = {k: it[k] for k in ("id", "hash", "question", "pert_question", "removed_fact", "removed_type", "options")}
                t0 = time.time()
                tr = run_episode(ep, rs, "C4", m, cfg, sim, retriever)
                if tr["outcome"] in ("abstain", "unresolved"):
                    msgs, sha_ok = forced_messages(tr, ep, cfg)
                    r = m.letters(msgs)
                    tr["forced_v5"] = {"letter": r["letter"], "logits4": r["logits4"], "logits4_sha256": r["logits4_sha256"],
                                       "sha_ok": sha_ok}
                for x in tr["turns"]:
                    if x.get("action") == "SEARCH" and "observation" in x:
                        x["observation"] = None  # StatPearls text is not stored (document ids remain in the search field)
                tr["seconds"], tr["instance"], tr["state_k"] = round(time.time() - t0, 2), inst, st
                budget.charge(tr["seconds"], 1.0, note=f"v5_c4/{a.model}/{st}")
                fo.write(json.dumps(tr, ensure_ascii=False) + "\n")
                fo.flush()
    print("c4 done", flush=True)


def fcr_vec(trs, gold, order):
    out = []
    for h in order:
        t = trs[h]
        if t["outcome"] == "answer":
            L = t["answer"]
        elif t["outcome"] == "parse_fail":
            L = "A"
        else:
            L = t["forced_v5"]["letter"]
        out.append(L == gold[h])
    return np.array(out, dtype=float)


def cmd_report(a):
    import analysis as A
    from v5_report import c0_vecs
    items = [json.loads(x) for x in open(ITEMS)]
    gold, order = {x["hash"]: x["answer_idx"] for x in items}, [x["hash"] for x in items]
    boot = A.Bootstrap(len(order))
    rep, rows = {}, []
    for m in MAIN:
        T = {}
        for st in STATE:
            f = OUT / m / "C4" / st / "trajectories.jsonl"
            if not f.exists():
                break
            T[st] = {t["hash"]: t for t in map(json.loads, open(f))}
        c0 = c0_vecs(m, gold, order)
        if len(T) < 3 or c0 is None or any(len(T[s]) < len(order) for s in T):
            rep[m] = {"incomplete": True}
            continue
        r = {}
        for st in STATE:
            fcr = fcr_vec(T[st], gold, order)
            cr = np.array([T[st][h]["outcome"] == "answer" and T[st][h]["answer"] == gold[h] for h in order], dtype=float)
            ref = c0["S0" if st == "S0" else "S1K"]
            r[st] = {"fcr": float(fcr.mean()), "cr": float(cr.mean()), "d_fcr_vs_c0": A.paired_delta(fcr, ref["fcr"], boot),
                     "d_cr_vs_c0": A.paired_delta(cr, ref["cr"], boot),
                     "first_action": dict(Counter(T[st][h]["first_action"] for h in order)),
                     "outcome": dict(Counter(T[st][h]["outcome"] for h in order)),
                     "provided_item_rate": float(np.mean([any((x.get("sim") or {}).get("kind") == "provided" for x in T[st][h]["turns"]) for h in order])),
                     "forced_sha_ok": all(T[st][h].get("forced_v5", {}).get("sha_ok", True) for h in order), "_fcr": fcr, "_cr": cr}
        r["s1k_minus_s2k"] = {"fcr": A.paired_delta(r["S1K"]["_fcr"], r["S2K"]["_fcr"], boot), "cr": A.paired_delta(r["S1K"]["_cr"], r["S2K"]["_cr"], boot)}
        for st in STATE:
            r[st].pop("_fcr"), r[st].pop("_cr")
        rep[m] = r
        rows.append([m] + [f"{r[s]['fcr']:.3f} / {r[s]['cr']:.3f}" for s in STATE] + [_dl(r[s]["d_fcr_vs_c0"]) for s in STATE] +
                    [_dl(r[s]["d_cr_vs_c0"]) for s in STATE] + [_dl(r["s1k_minus_s2k"]["fcr"]), f"{r['S1K']['provided_item_rate']:.3f}",
                                                               json.dumps(r["S1K"]["first_action"])])
    (ROOT / "runs" / "v5" / "c4_report.json").write_text(json.dumps(rep, indent=1))
    L = ["# D26b C4 reference (p3, v2) DEV-K (no interpretation)", "", "- Δ vs C0: S0 vs C0 S0, S1K and S2K vs C0 S1K. FCR = pf A + forced answer for abstain/unresolved (D18a). CI = item-clustered paired (B 10,000, seed 20261016).", "",
         "| model | FCR / CR S0 | S1K | S2K | Δ_FCR vs C0 S0 | S1K | S2K | Δ_CR vs C0 S0 | S1K | S2K | S1K − S2K (FCR) | S1K share of items that received a removed fact | first action S1K |",
         "|" + "---|" * 13] + ["| " + " | ".join(x) + " |" for x in rows]
    L += ["", "first actions and outcome categories by state: c4_report.json"]
    (ROOT / "runs" / "v5" / "c4_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def _dl(d):
    return "—" if not d or "delta" not in d else f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--model")
    ap.add_argument("--sim-ports", default="8707")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    main()
