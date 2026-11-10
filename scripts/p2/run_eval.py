"""D74-r4 main run (EVAL, 736 items, runs/v5/evalk/evalk.jsonl sha256 2da8bef3...91bd). Loads the model once in one process and runs the cells in turn:
C0 (S0, S1K) -> C0 S2K -> oracle (L-S0, L-S1K, O2, O1a, O1b) -> NP probe (S0, S1) -> C4 (S1K, S2K) -> P6 v1 (S1K, S2K) -> C4f v3 (S1K, S2K).
Same code path as run_pilot.py (D74-r1) (registered and FROZEN hash checks, simulator /health, instance check per cell); stop rule p2_models.STOP_AT_FIRST_JSON.
Output runs/p2/eval/<model>/. No resume: does not start if the output directory exists. After an infrastructure stop, as in D41a, the partial output is not read but moved to
runs/p2/aborted/<UTC>_<model>/ and the run restarts from the beginning (done by a person, recorded in the deviations). No answers or accuracy are computed in this script.
CUDA_VISIBLE_DEVICES=3 python scripts/p2/run_eval.py --model qwen3-14b --sim-ports 8714,8715
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts"), str(ROOT / "scripts" / "p2"), str(ROOT / "scripts" / "d56"), str(ROOT / "scripts" / "d58")]
V = ROOT / "runs" / "v5"; P2R = ROOT / "runs" / "p2"
ITEMS = V / "evalk" / "evalk.jsonl"
ITEMS_SHA = "2da8bef3ed4176ce86f6734bd65877d8e6da884034b9b40c97d149a0a77191bd"
SIM_REV = "495f39366efef23836d0cfae4fbe635880d2be31"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--sim-ports", default="8714,8715"); a = ap.parse_args()
    out = P2R / "eval" / a.model
    if out.exists():
        sys.exit(f"{out} exists (no resume)")
    assert hashlib.sha256(ITEMS.read_bytes()).hexdigest() == ITEMS_SHA
    out.mkdir(parents=True)
    import budget
    budget.LIMITS["dev"] = {"warn": float("inf"), "hard": float("inf")}
    budget.LEDGER_DIR = P2R / "budget"
    import p2_models
    p2_models.install()
    assert a.model in p2_models.SPECS and p2_models.STOP_AT_FIRST_JSON
    from run_pilot import check_hashes
    check_hashes()
    import torch
    import agent, c4_ref_k, pipeline_k, prompts_c4f, prompts_p6
    from llm_client import Client
    ports = [int(p) for p in a.sim_ports.split(",")]
    health = {}
    for p in ports:
        h = Client(p).health()
        assert h.get("model") == "c1" and h.get("revision") == SIM_REV, h
        health[p] = h
    pipeline_k.ITEMS = c4_ref_k.ITEMS = ITEMS
    meta = {"model": a.model, "spec": p2_models.SPECS[a.model], "precision": p2_models.PRECISION[a.model], "cache": p2_models.CACHE,
            "stop_at_first_json": p2_models.STOP_AT_FIRST_JSON, "refs": str(p2_models.REFS), "items_sha256": ITEMS_SHA,
            "sim_ports": ports, "sim_health": health, "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "cells": {}}
    t0 = time.time()
    m = pipeline_k.EvalModel(a.model)
    meta["load_seconds"] = round(time.time() - t0, 1)
    PK_Eval = pipeline_k.EvalModel
    pipeline_k.EvalModel = lambda name: m
    cfg = agent.load_config(allow_provisional=False)

    def cell(name, fn):
        torch.cuda.reset_peak_memory_stats(); s = time.time()
        fn()
        meta["cells"][name] = {"seconds": round(time.time() - s, 1), "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                               "end_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        print(name, meta["cells"][name], flush=True)

    sp = ",".join(map(str, ports))
    cell("C0", lambda: pipeline_k.cmd_c0(argparse.Namespace(model=a.model, n=None, out=str(out / "c0"))))

    def c0s2k():
        import run_devk2_explore as R
        R.ITEMS, R.OUT = ITEMS, out / "c0"
        R.c0s2k(a.model)
    cell("C0_S2K", c0s2k)

    def oracle():
        from consol_requery import consol_messages
        import prompts_d56 as P
        with open(out / "oracle.jsonl", "w") as fo:
            for it in map(json.loads, open(ITEMS)):
                f1, f2 = (x.strip() for x in it["removed_facts"])
                conds = {"L-S0": (it, "S0"), "L-S1K": (it, "S1")}
                for c, facts in (("O2", f"{f1} {f2}"), ("O1a", f1), ("O1b", f2)):
                    conds[c] = ({**it, "pert_question": it["pert_question"] + P.ORACLE_SEP + P.ORACLE_PREFIX + facts}, "S1")
                s = time.time(); rec = {"hash": it["hash"], "row_id": it["row_id"], "model": a.model}
                for c, (x, st) in conds.items():
                    r = m.letters(consol_messages(x, st, [], cfg))
                    rec[c] = {"letter": r["letter"], "logits4": r["logits4"], "p4max": r["p4max"], "prompt_sha256": r["prompt_sha256"]}
                rec["seconds"] = round(time.time() - s, 2)
                fo.write(json.dumps(rec) + "\n")
    cell("oracle", oracle)

    def np_probe():  # same input and record as scripts/d58/np_probe.py (same instance, no model reload)
        from agent import case_text
        import prompts_d58 as P
        assert P.sha(P.NP_QUESTION) == "1f9431e06753326cda524193b724ec5c2f7f0de4c78fdf24f9140d451e0b15c8"
        tok, model, render = m.lm.tok, m.lm.model, m.lm.render
        base = render([{"role": "user", "content": "x"}]); ids = {}
        for w in ("Yes", "No"):
            b = tok(base, add_special_tokens=False)["input_ids"]; x = tok(base + w, add_special_tokens=False)["input_ids"]
            assert x[:len(b)] == b and len(x) == len(b) + 1, f"{w} is not a single token after the generation prompt"
            ids[w] = x[-1]
        with open(out / "np_probe.jsonl", "w") as fo:
            for it in map(json.loads, open(ITEMS)):
                s = time.time(); rec = {"hash": it["hash"], "row_id": it["row_id"], "model": a.model, "token_ids": ids}
                for st, q in (("S0", it["question"]), ("S1", it["pert_question"])):
                    prompt = render([{"role": "user", "content": case_text(q, it["options"]) + P.NP_SEP + P.NP_QUESTION}])
                    x = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
                    with torch.no_grad():
                        lg = model(x).logits[0, -1].float().cpu()
                    p = torch.softmax(lg, -1)
                    rec[st] = {"logit_yes": float(lg[ids["Yes"]]), "logit_no": float(lg[ids["No"]]), "p_yes": float(p[ids["Yes"]]), "p_no": float(p[ids["No"]]),
                               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
                rec["seconds"] = round(time.time() - s, 2)
                fo.write(json.dumps(rec) + "\n")
    cell("NP", np_probe)

    c4_ref_k.OUT, c4_ref_k.STATE = out / "c4", {"S1K": "S1", "S2K": "S2"}
    cell("C4", lambda: c4_ref_k.cmd_run(argparse.Namespace(model=a.model, sim_ports=sp)))

    def p6():
        fz = json.load(open(V / "d37/FROZEN_P6_v1.json"))
        assert prompts_p6.sha(pipeline_k.GEN_SYSTEM) == fz["gen_prompt_sha256"]
        pipeline_k.STATES = {"S1K": "s1k", "S2K": "s2k"}
        pipeline_k.SCHEMAS = {"Q3": pipeline_k.SCHEMAS["Q3"]}
        pipeline_k.ARMS = {"P6": ("Q3", 3, 0)}
        pipeline_k.GEN_SYSTEM = prompts_p6.gen_system(pipeline_k.GEN_SYSTEM)
        pipeline_k.cmd_arms(argparse.Namespace(model=a.model, n=None, sim_ports=sp, out=str(out / "p6"), only_det=False))
    cell("P6", p6)

    def c4f():
        import agent_c4f_v3, abstain_probe
        fz = json.load(open(V / "d36/FROZEN_C4F_v3.json"))
        assert prompts_c4f.prompt_sha(cfg) == fz["system_prompt_sha256"]
        c4_ref_k.OUT, c4_ref_k.STATE = out / "c4f", {"S1K": "S1", "S2K": "S2"}
        c4_ref_k.run_episode = agent_c4f_v3.run_episode_c4f_v3
        abstain_probe.system_prompt = prompts_c4f.c4f_system_prompt
        c4_ref_k.cmd_run(argparse.Namespace(model=a.model, sim_ports=sp))
    cell("C4f", c4f)
    pipeline_k.EvalModel = PK_Eval
    meta["total_seconds"] = round(time.time() - t0, 1); meta["end_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "DONE").write_text("")
    print("EVAL DONE", a.model, flush=True)


if __name__ == "__main__":
    main()
