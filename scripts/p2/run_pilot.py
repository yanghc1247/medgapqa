"""D74 throughput pilot (50 DEV-K2 items, runs/p2/pilot_sample.json). Loads the model once in one process and runs the cells in turn:
C0 (S0, S1K) -> C4 (S1K) -> P6 v1 (S1K) -> oracle (L-S0, L-S1K, O2, O1a, O1b) -> C4f v3 (S1K). C4f is last because it overrides globals.
Same code path as scripts/run_evalk.py and scripts/d56/b1_oracle.py (including registered and FROZEN hash checks). Output runs/p2/pilot/<model>/, no resume.
Simulator requests are logged to runs/p2/pilot/<model>/sim_requests.jsonl (for the identity check).
CUDA_VISIBLE_DEVICES=3 python scripts/p2/run_pilot.py --model qwen3-14b --sim-ports 8714,8715
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts"), str(ROOT / "scripts" / "p2"), str(ROOT / "scripts" / "d56")]
V = ROOT / "runs" / "v5"; P2R = ROOT / "runs" / "p2"
SIM_SHA = "68f99cd48cf56d7f2533404e1f913a7ee82d7a23025eb41d392c68bd3445e002"
SIM_REV = "495f39366efef23836d0cfae4fbe635880d2be31"
REG = {"src/medgapqa/agent.py": "58d7ab3cbc242c85d905863a3b437a67a8ff9ada95355a62928bfbfb4fc02d77", "src/medgapqa/c4_ref_k.py": "365cefa457f4f422ca3fef2a7c5a6e5a70771fc0eea4a3524ca7be989dce3afa",
       "src/medgapqa/prompts.py": "0316d72099d176c58a44e46be4731292749f6b5b30699457a737f32b6aae2849", "configs/agent.json": "0cfb0e71c9e71da6e7a03defedb63356484f7117c369c2f9ef93e6f2718f3969",
       "src/medgapqa/simulator_k.py": "c32be77dda0cba85a4222a6676e45a78bf9528de1459111a19248e263bc13254", "src/medgapqa/forced_answer.py": "0295c8f6e558b016a425f41c496834e4c281166819a3d391ce11c2a384958462",
       "src/medgapqa/abstain_probe.py": "9e94d3db3b76166e0ff00fe6fd7df2df4af3e9f4a8cf36e4d05101b807fd71c6",
       "src/medgapqa/pipeline_k.py": "559727c4abc2c9cb8190e8ca591ed3936c208715bb61d64701a1a564e5836f77", "src/medgapqa/panel_necessity.py": "5318e57da0d117349d191b6f0be2f23da58dcc27d12ed9fe23cddb7116e7cd3d"}


def check_hashes():
    import simulator_k
    bad = [p for p, h in REG.items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    if simulator_k.prompt_sha() != SIM_SHA:
        bad.append("simulator_prompt")
    for name, fz in (("c4f", "d36/FROZEN_C4F_v3.json"), ("p6", "d37/FROZEN_P6_v1.json")):
        d = json.load(open(V / fz))
        bad += [f"{name}:{p}" for p, h in d["files_sha256"].items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    if bad:
        sys.exit(f"registered hash mismatch: {bad}")


def pilot_items(out, n=None):
    s = json.loads((P2R / "pilot_sample.json").read_text())
    src = ROOT / s["source"]
    assert hashlib.sha256(src.read_bytes()).hexdigest() == s["source_sha256"]
    rows = [json.loads(x) for x in open(src)]
    f = out / "items.jsonl"
    with open(f, "w") as fo:
        for i in s["indices"][:n]:  # n: first n items only (D74-r7 235B speed measurement)
            fo.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--sim-ports", default="8714,8715"); ap.add_argument("--out-root", default="pilot"); ap.add_argument("--n", type=int); a = ap.parse_args()
    out = P2R / a.out_root / a.model  # the D74-r4 rerun goes to pilot_r4 (original pilot kept)
    if out.exists():
        sys.exit(f"{out} exists (no resume)")
    out.mkdir(parents=True)
    import budget
    budget.LIMITS["dev"] = {"warn": float("inf"), "hard": float("inf")}
    budget.LEDGER_DIR = P2R / "budget"
    import p2_models
    p2_models.install()
    assert a.model in p2_models.SPECS
    check_hashes()
    import torch
    import agent, c4_ref_k, pipeline_k, prompts_c4f, prompts_p6, llm_client
    from llm_client import Client
    ports = [int(p) for p in a.sim_ports.split(",")]
    health = {}
    for p in ports:
        h = Client(p).health()
        assert h.get("model") == "c1" and h.get("revision") == SIM_REV, h
        health[p] = h
    simlog = open(out / "sim_requests.jsonl", "a")
    orig_chat = llm_client.RemoteGenerator.chat

    def logged_chat(self, msgs, prefill=""):
        t0 = time.time(); r = self.c.chat(msgs, prefill)
        simlog.write(json.dumps({"port": int(self.c.url.rsplit(":", 1)[1]), "messages": msgs, "prefill": prefill,
                                 "resp_sha": r.get("resp_sha"), "seconds_client": round(time.time() - t0, 3),
                                 "seconds_server": r.get("seconds")}, ensure_ascii=False) + "\n"); simlog.flush()
        return r["text"], r["in_tokens"], r["out_tokens"]
    llm_client.RemoteGenerator.chat = logged_chat
    items = pilot_items(out, a.n)
    pipeline_k.ITEMS = c4_ref_k.ITEMS = items
    meta = {"stop_at_first_json": p2_models.STOP_AT_FIRST_JSON, "refs": str(p2_models.REFS), "model": a.model, "spec": p2_models.SPECS[a.model], "precision": p2_models.PRECISION[a.model], "cache": p2_models.CACHE,
            "sim_ports": ports, "sim_health": health, "cells": {}}
    t0 = time.time()
    m = pipeline_k.EvalModel(a.model)
    meta["load_seconds"] = round(time.time() - t0, 1)
    PK_Eval = pipeline_k.EvalModel
    pipeline_k.EvalModel = lambda name: m  # not reloaded per cell (verify runs per cell -> the first cell writes the reference, later cells compare)

    def cell(name, fn):
        torch.cuda.reset_peak_memory_stats(); s = time.time()
        fn()
        meta["cells"][name] = {"seconds": round(time.time() - s, 1), "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        print(name, meta["cells"][name], flush=True)

    sp = ",".join(map(str, ports))
    cell("C0", lambda: pipeline_k.cmd_c0(argparse.Namespace(model=a.model, n=None, out=str(out / "c0"))))
    c4_ref_k.OUT, c4_ref_k.STATE = out / "c4", {"S1K": "S1"}
    cell("C4", lambda: c4_ref_k.cmd_run(argparse.Namespace(model=a.model, sim_ports=sp)))
    cfg = agent.load_config(allow_provisional=False)

    def p6():
        pipeline_k.STATES = {"S1K": "s1k"}
        pipeline_k.SCHEMAS = {"Q3": pipeline_k.SCHEMAS["Q3"]}
        pipeline_k.ARMS = {"P6": ("Q3", 3, 0)}
        fz = json.load(open(V / "d37/FROZEN_P6_v1.json"))
        assert prompts_p6.sha(pipeline_k.GEN_SYSTEM) == fz["gen_prompt_sha256"]  # same checks as run_evalk.py (before any override)
        pipeline_k.GEN_SYSTEM = prompts_p6.gen_system(pipeline_k.GEN_SYSTEM)
        pipeline_k.cmd_arms(argparse.Namespace(model=a.model, n=None, sim_ports=sp, out=str(out / "p6"), only_det=False))
    cell("P6", p6)

    def oracle():
        from consol_requery import consol_messages
        import prompts_d56 as P
        with open(out / "oracle.jsonl", "w") as fo:
            for it in map(json.loads, open(items)):
                f1, f2 = (x.strip() for x in it["removed_facts"])
                conds = {"L-S0": (it, "S0"), "L-S1K": (it, "S1")}
                for c, facts in (("O2", f"{f1} {f2}"), ("O1a", f1), ("O1b", f2)):
                    conds[c] = ({**it, "pert_question": it["pert_question"] + P.ORACLE_SEP + P.ORACLE_PREFIX + facts}, "S1")
                s = time.time(); rec = {"hash": it["hash"], "id": it["id"]}
                for c, (x, st) in conds.items():
                    r = m.letters(consol_messages(x, st, [], cfg))
                    rec[c] = {"letter": r["letter"], "logits4": r["logits4"], "p4max": r["p4max"], "prompt_sha256": r["prompt_sha256"]}
                rec["seconds"] = round(time.time() - s, 2)
                fo.write(json.dumps(rec) + "\n")
    cell("oracle", oracle)

    def c4f():
        import agent_c4f_v3, abstain_probe
        fz = json.load(open(V / "d36/FROZEN_C4F_v3.json"))
        assert prompts_c4f.prompt_sha(cfg) == fz["system_prompt_sha256"]
        c4_ref_k.OUT, c4_ref_k.STATE = out / "c4f", {"S1K": "S1"}
        c4_ref_k.run_episode = agent_c4f_v3.run_episode_c4f_v3
        abstain_probe.system_prompt = prompts_c4f.c4f_system_prompt
        c4_ref_k.cmd_run(argparse.Namespace(model=a.model, sim_ports=sp))
    cell("C4f", c4f)
    pipeline_k.EvalModel = PK_Eval
    meta["total_seconds"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "DONE").write_text("")
    print("PILOT DONE", a.model, flush=True)


if __name__ == "__main__":
    main()
