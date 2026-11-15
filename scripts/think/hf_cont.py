"""D130-o 2: in the transformers results (Meerkat-70B NF4), episodes that reached the limit (action turn 256) are rerun without the limit (up to the context of 8,192) with transformers greedy decoding.
With greedy decoding this equals 'continuing from the cut' (the tokens before the cut are identical for the same input and model). Targets are read from hf_cap_hits.json (currently C4 S1K and S2K, d31626ab990a3df5).
Results go to a new tree runs/p2/eval_hf_nocap/<model>/ (the original runs/p2/eval and eval_cap1024 are not modified): original entries are linked, and only the changed cell files are copied with those rows replaced.
python scripts/think/hf_cont.py --model meerkat-70b-nf4 --sim-ports 8958
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts"), str(ROOT / "scripts" / "p2"), str(ROOT / "scripts" / "d56"), str(ROOT / "scripts" / "think")]
P2 = ROOT / "runs" / "p2"
CTX = {"meerkat-70b-nf4": 8192}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--sim-ports", required=True); a = ap.parse_args()
    m = a.model
    hits = json.load(open(P2 / "analysis_vllm" / "hf_cap_hits.json"))[m]
    todo = {k: sorted({h[0] for h in v["hits"]}) for k, v in hits.items() if v["hits"] and not k.startswith("P6")}
    print("to rerun:", todo, flush=True)
    src = P2 / "eval_cap1024" / m; dst = P2 / "eval_hf_nocap" / m
    if not dst.exists():
        dst.mkdir(parents=True)
        for e in src.iterdir():
            os.symlink(e.resolve(), dst / e.name)
    if not todo:
        return
    import p2_models
    p2_models.install()
    import pipeline_k, agent, agent_c4f_v3, abstain_probe, prompts_c4f
    from forced_answer import forced_messages
    from retrieve import Retriever
    from run_agent import load_tok
    import veval as V
    from run_pilot import check_hashes
    check_hashes()
    cfg = agent.load_config(allow_provisional=False)
    em = pipeline_k.EvalModel(m)
    ok, how = em.verify(); assert ok, how
    tok = em.lm.tok

    class Uncapped:
        """Up to the context end (8,192 - input length) instead of the action-turn limit. The first-JSON stop rule (STOP_AT_FIRST_JSON) of the wrapper is unchanged."""
        calls = []

        def chat(self, msgs, prefill, max_new):
            n = len(tok(em.lm.render(msgs) + prefill, add_special_tokens=False)["input_ids"])
            out = em.chat(msgs, prefill, max(1, CTX[m] - n))
            self.calls.append({"n_in": n, "n_out": out[2], "truncated": out[3], "cap": CTX[m] - n})
            return out

        def letters(self, msgs):
            return em.letters(msgs)
    simk, *_ = V.sim_setup([int(p) for p in a.sim_ports.split(",")])
    retriever = Retriever.load(cfg["search_top_k"], cfg["search_max_tokens"], load_tok(cfg["search_truncation_tokenizer"]))
    items = {json.loads(l)["hash"]: json.loads(l) for l in open(ROOT / "runs" / "v5" / "evalk" / "evalk.jsonl")}
    report = []
    for cell, hs in todo.items():
        pol, st = cell.split("_")
        sub = {"C0": "c0", "C4": "c4", "C4f": "c4f"}[pol]; cond = "C0" if pol == "C0" else "C4"
        rel = Path(sub) / m / cond / st / "trajectories.jsonl"
        if (dst / sub).is_symlink():  # turn the cell directory into a real directory and replace only the files (other files stay linked)
            real = (dst / sub).resolve(); (dst / sub).unlink(); shutil.copytree(real, dst / sub, symlinks=True)
        f = dst / rel
        rows = [json.loads(l) for l in open(f)]
        runner = agent_c4f_v3.run_episode_c4f_v3 if pol == "C4f" else agent.run_episode
        if pol == "C4f":
            abstain_probe.system_prompt = prompts_c4f.c4f_system_prompt
        for h in hs:
            it = items[h]; ep = {k: it[k] for k in ("id", "hash", "question", "pert_question", "removed_fact", "removed_type", "options")}
            model = Uncapped(); Uncapped.calls = []
            t0 = time.time()
            tr = runner(ep, {"S0": "S0", "S1K": "S1", "S2K": "S2"}[st], cond, model, cfg, V.SimAdapter(simk, it["removed_facts"]), retriever)
            if tr["outcome"] in ("abstain", "unresolved"):
                msgs, sha_ok = forced_messages(tr, ep, cfg); r = em.letters(msgs)
                tr["forced_v5"] = {"letter": r["letter"], "logits4": r["logits4"], "logits4_sha256": r["logits4_sha256"], "sha_ok": sha_ok}
            for x in tr["turns"]:
                if x.get("action") == "SEARCH" and "observation" in x:
                    x["observation"] = None
            old = next(r for r in rows if r["hash"] == h)
            tr.update(seconds=round(time.time() - t0, 2), instance=f"host:{m}:transformers", state_k=st, d130o_nocap={"ctx": CTX[m], "calls": Uncapped.calls,
                      "previous": {"outcome": old["outcome"], "answer": old.get("answer"), "turns": [{k: t.get(k) for k in ("turn", "action", "truncated", "retry_truncated")} for t in old["turns"]]}})
            rows = [tr if r["hash"] == h else r for r in rows]
            report.append({"cell": cell, "hash": h, "before": {"outcome": old["outcome"], "answer": old.get("answer")}, "after": {"outcome": tr["outcome"], "answer": tr.get("answer")}})
            print(cell, h, "before", old["outcome"], old.get("answer"), "after", tr["outcome"], tr.get("answer"), flush=True)
        if f.is_symlink():
            f.unlink()
        with open(f, "w") as fo:
            for r in rows:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    (dst / "D130O_NOCAP.json").write_text(json.dumps({"model": m, "ctx": CTX[m], "rerun": report, "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
