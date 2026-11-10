"""D26 v5 evaluation runner (new file; frozen files only imported). On DEV-K (runs/v5/devk/devk.jsonl):

c0   : C0 of the evaluation model (agent.run_episode as is), S0 and S1K. Trajectories for FCR (pf A) and CR.
arms : pipeline arms (D26 E). States S0, S1K, S2K; question generation (greedy, max_new 200, one format re-request for invalid JSON) →
       ASK via simulator v10 (each question independent, same visible item), SEARCH via the existing retrieve (top 3, 300 tokens) → consolidated re-query (D23 template,
       consol_requery.consol_messages) → argmax of A–D restricted logits after FORCED_PREFILL. P1a uses only the first question of an NQ=3, NS=0 generation; P1b uses all 3.
det  : rerun the same 20 items on another GPU and compare records (reproducibility check).
Stored: generated text (questions, queries), simulator labels and kinds, retrieved document ids, letter, logits4, sha, n_in only. Prompts and observations containing SEARCH text are not stored.
Evaluation models: qwen2.5-7b, meerkat-8b, medgemma-4b, olmo2-7b (run_agent.HFModel, checked against instance references), qwen2.5-32b (4-bit, panel_necessity.LetterModel;
it has no registered instance reference, so the consistency_probes output hashes of the first run are recorded in runs/v5/ref_qwen2.5-32b.json and checked in later runs).

CUDA_VISIBLE_DEVICES=4 python src/medgapqa/pipeline_k.py c0 --model qwen2.5-7b
CUDA_VISIBLE_DEVICES=4 python src/medgapqa/pipeline_k.py arms --model qwen2.5-7b --sim-ports 8704,8705,8706,8707
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/medgapqa"))
sys.path.insert(0, str(ROOT / "scripts"))
from agent import FORCED_PREFILL, case_text, load_config, run_episode  # noqa: E402
from forced_answer import logits4_sha  # noqa: E402
from format_check import parse_last_json  # noqa: E402

ITEMS = ROOT / "runs" / "v5" / "devk" / "devk.jsonl"
OUT = ROOT / "runs" / "v5" / "eval_devk"
REGISTERED = ("qwen2.5-7b", "meerkat-8b", "medgemma-4b", "olmo2-7b")
STATES = {"S0": "s0", "S1K": "s1k", "S2K": "s2k"}
GEN_SYSTEM = ("You are a clinical decision agent working on a multiple-choice medical question. Some patient information may be missing "
              "from the question. Before answering, you must ask {NQ} short, specific questions about this patient, each about one clinical "
              "finding, test result, history element, or category, and/or write {NS} search queries for a general medical knowledge base. "
              "Do not repeat information already stated in the question. Output exactly one JSON object and nothing else: {SCHEMA}")
SCHEMAS = {
    "Q3": (3, 0, '{"questions": ["<question 1>", "<question 2>", "<question 3>"]}'),
    "S2": (0, 2, '{"queries": ["<query 1>", "<query 2>"]}'),
    "Q3S2": (3, 2, '{"questions": ["<question 1>", "<question 2>", "<question 3>"], "queries": ["<query 1>", "<query 2>"]}'),
}
ARMS = {"P1a": ("Q3", 1, 0), "P1b": ("Q3", 3, 0), "P3": ("S2", 0, 2), "P4": ("Q3S2", 3, 2)}  # (generation, questions to use, queries to use)
GEN_RETRY = "Your previous output was not valid. Output exactly one JSON object and nothing else: {SCHEMA}"
MAX_NEW_GEN = 200


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


class EvalModel:
    def __init__(self, name):
        self.name = name
        if name in REGISTERED:
            from forced_answer import Scorer
            self.sc = Scorer(name)
            self._chat = self.sc.hf.chat
        else:
            from panel_necessity import LetterModel
            self.lm = LetterModel(name)
            self._chat = self.lm.chat

    def chat(self, msgs, prefill, max_new):
        return self._chat(msgs, prefill, max_new)

    def letters(self, msgs):
        if self.name in REGISTERED:
            r = self.sc(msgs)
            four = np.array(r["logits4"])
            out = {"letter": r["letter"], "logits4": r["logits4"], "n_in": r["n_in"], "prompt_sha256": r["prompt_sha256"]}
        else:
            r = self.lm.letters(msgs)
            four = np.array(r["logits4"])
            out = {"letter": "ABCD"[int(np.argmax(four))], "logits4": r["logits4"], "n_in": r["n_in"], "prompt_sha256": r["prompt_sha256"]}
        p = np.exp(four - four.max())
        p /= p.sum()
        out["p4max"] = float(p.max())
        out["logits4_sha256"] = logits4_sha(out["logits4"])
        return out

    def verify(self):
        from instances import sha as isha, verify_instance
        from run_agent import REF_HASHES, consistency_probes
        if self.name in REGISTERED:
            ok, got = verify_instance(self.chat, consistency_probes(), json.load(open(REF_HASHES))[self.name])
            return ok, "registered"
        got = [isha(self.chat(m, p, n)[0]) for m, p, n in consistency_probes()]
        ref = ROOT / "runs" / "v5" / f"ref_{self.name}.json"
        if not ref.exists():
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(json.dumps({"hashes": got, "instance": inst_id(self.name)}, indent=1))
            return True, "reference created"
        return got == json.loads(ref.read_text())["hashes"], "compared to runs/v5 reference"


def inst_id(name):
    return f"{os.uname().nodename}:gpu{os.environ.get('CUDA_VISIBLE_DEVICES')}:pid{os.getpid()}:{name}"


def load_items(n=None):
    items = [json.loads(x) for x in open(ITEMS)]
    return items[:n] if n else items


def visible_state(state):
    return "S0" if state == "S0" else "S1"


def cmd_c0(a):
    from budget import Budget
    cfg = load_config(allow_provisional=False)
    m = EvalModel(a.model)
    ok, how = m.verify()
    if not ok:
        sys.exit(f"instance check failed for {a.model} ({how})")
    print(f"{a.model} instance ok ({how})", flush=True)
    budget, inst = Budget("dev"), inst_id(a.model)
    for state in ("S0", "S1K"):
        d = Path(a.out) / a.model / "C0" / state
        d.mkdir(parents=True, exist_ok=True)
        f = d / "trajectories.jsonl"
        if f.exists():
            sys.exit(f"{f} exists (no overwrite)")
        with open(f, "w") as fo:
            for it in load_items(a.n):
                t0 = time.time()
                tr = run_episode({k: it[k] for k in ("id", "hash", "question", "pert_question", "removed_fact", "removed_type", "options")},
                                 visible_state(state), "C0", m, cfg)
                tr["seconds"], tr["instance"], tr["state"] = round(time.time() - t0, 2), inst, state
                budget.charge(tr["seconds"], 1.0, note=f"v5_c0/{a.model}/{state}")
                fo.write(json.dumps(tr, ensure_ascii=False) + "\n")
    print("c0 done", flush=True)


def parse_gen(raw, gen):
    nq, ns, _ = SCHEMAS[gen]
    obj = parse_last_json(raw)
    obj = obj[0] if obj else None
    if not isinstance(obj, dict):
        return None
    out = {}
    for key, n in (("questions", nq), ("queries", ns)):
        if n:
            v = obj.get(key)
            if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
                return None
            out[key] = [x.strip() for x in v]
    return out


def generate(m, it, state, gen):
    nq, ns, schema = SCHEMAS[gen]
    visible = it["question"] if state == "S0" else it["pert_question"]
    msgs = [{"role": "system", "content": GEN_SYSTEM.format(NQ=nq, NS=ns, SCHEMA=schema)},
            {"role": "user", "content": case_text(visible, it["options"]) + "\n\nYour output:"}]
    raw, n_in, n_out, trunc = m.chat(msgs, "", MAX_NEW_GEN)
    rec = {"raw": raw, "truncated": trunc, "tokens": {"in": n_in, "out": n_out}, "input_sha256": sha(json.dumps(msgs, sort_keys=True))}
    got = None if trunc else parse_gen(raw, gen)
    if got is None:
        msgs2 = msgs + [{"role": "assistant", "content": raw}, {"role": "user", "content": GEN_RETRY.format(SCHEMA=schema)}]
        raw2, i2, o2, t2 = m.chat(msgs2, "", MAX_NEW_GEN)
        rec.update(retry_raw=raw2, retry_truncated=t2, retry_tokens={"in": i2, "out": o2})
        got = None if t2 else parse_gen(raw2, gen)
    rec["valid"] = got is not None
    rec["parsed"] = got or {"questions": [], "queries": []}
    return rec


def run_arm(m, it, state, arm, gens, sim, retriever, cfg):
    from consol_requery import consol_messages
    from retrieve import format_results
    gen_name, use_q, use_s = ARMS[arm]
    g = gens[gen_name]["parsed"]
    qs, ss = g.get("questions", [])[:use_q], g.get("queries", [])[:use_s]
    visible = it["question"] if state == "S0" else it["pert_question"]
    blocks, asks, searches = [], [], []
    for q in qs:
        text, rec = sim.respond(STATES[state], visible, it["removed_facts"], q)
        blocks.append(f"Question asked: {q}\nResponse: {text}")
        asks.append({"q": q, "label": rec["label"], "kind": rec["kind"], "qtype": rec["qtype"], "sim_instance": rec.get("instance")})
    for s in ss:
        res = retriever.search(s)
        blocks.append(f"Search query: {s}\n{format_results(s, res)}")
        searches.append({"q": s, "doc_ids": [r["id"] for r in res]})
    msgs = consol_messages(it, visible_state(state), blocks, cfg)
    ans = m.letters(msgs)
    return {"asks": asks, "searches": searches, "n_blocks": len(blocks), **ans}


def cmd_arms(a):
    from budget import Budget
    from instances import PooledBackend
    from llm_client import RemoteGenerator
    from retrieve import Retriever
    from run_agent import load_tok
    from simulator_k import SimulatorK
    cfg = load_config(allow_provisional=False)
    m = EvalModel(a.model)
    ok, how = m.verify()
    if not ok:
        sys.exit(f"instance check failed for {a.model} ({how})")
    ports = [int(p) for p in a.sim_ports.split(",")]
    sim = SimulatorK(PooledBackend([RemoteGenerator(p) for p in ports], [f"sim:{p}" for p in ports]))
    retriever = Retriever.load(cfg["search_top_k"], cfg["search_max_tokens"], load_tok(cfg["search_truncation_tokenizer"]))
    items = load_items(a.n)
    if a.only_det:
        items = [items[int(i)] for i in sorted(np.random.default_rng(20261016).permutation(len(items))[:20])]
    out = Path(a.out) / a.model
    out.mkdir(parents=True, exist_ok=True)
    f = out / ("det.jsonl" if a.only_det else "arms.jsonl")
    done = {(r["id"], r["state"]) for r in map(json.loads, open(f))} if f.exists() else set()
    budget, inst = Budget("dev"), inst_id(a.model)
    print(f"{a.model} instance ok ({how}), items {len(items)}", flush=True)
    with open(f, "a") as fo:
        for it in items:
            for state in STATES:
                if (it["id"], state) in done:
                    continue
                t0 = time.time()
                gens = {g: generate(m, it, state, g) for g in SCHEMAS}
                arms = {arm: run_arm(m, it, state, arm, gens, sim, retriever, cfg) for arm in ARMS}
                sec = time.time() - t0
                budget.charge(sec, 1.0, note=f"v5_arms/{a.model}/{state}")
                fo.write(json.dumps({"id": it["id"], "hash": it["hash"], "state": state, "model": a.model, "gens": gens, "arms": arms,
                                     "seconds": round(sec, 2), "instance": inst}, ensure_ascii=False) + "\n")
                fo.flush()
    print(f"wrote {f}", flush=True)


def cmd_det(a):
    x = {(r["id"], r["state"]): r for r in map(json.loads, open(a.a))}
    y = {(r["id"], r["state"]): r for r in map(json.loads, open(a.b))}

    def sig(r):
        return [sha(json.dumps([r["gens"][g]["raw"], r["gens"][g].get("retry_raw")])) for g in sorted(r["gens"])] + \
               [r["arms"][k]["letter"] + r["arms"][k]["logits4_sha256"] for k in sorted(r["arms"])]
    same = [k for k in x if k in y and sig(x[k]) == sig(y[k])]
    rep = {"n_a": len(x), "n_b": len(y), "n_same": len(same), "ok": len(x) == len(y) == len(same) > 0}
    print(json.dumps(rep))
    sys.exit(0 if rep["ok"] else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["c0", "arms", "det"])
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--sim-ports", default="8707")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only-det", action="store_true")
    ap.add_argument("--a")
    ap.add_argument("--b")
    a = ap.parse_args()
    {"c0": cmd_c0, "arms": cmd_arms, "det": cmd_det}[a.cmd](a)


if __name__ == "__main__":
    main()
