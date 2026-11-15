"""D98 Thinking-mode exploration (meeting material, not in the paper at the time). The frozen C0 episode code (agent.run_episode) and the O2 prompt of the run_eval oracle
are run on a vLLM server (separate venv $PROJECT_ROOT/venv_vllm), Instant and Think on the same engine. The frozen code is only imported, not modified.

The client (this file) runs in the project .venv: chat rendering = p2_models.render (same tokenizer and template as the earlier P2 wrapper); token IDs are sent to the server.
- Instant: enable_thinking=False, greedy, limit max_new (256), first-JSON stop (after generation, cut at the end of the first JSON object of prefill + output.strip() = the same string as the D74-r4 result).
- Think: enable_thinking=True, temperature 0.6 / top_p 0.95 / top_k 20, per-request seed (derived from the sha256 of the input tokens). Thinking stage at most THINK_MAX (3,840) tokens,
  stopping at </think> (one token); if unfinished, recorded as think_truncated and closed with '\\n</think>\\n\\n'. Answer stage = then '\\n\\n' + prefill, same sampling, limit max_new.
  Whole output <= 4,096 tokens.
- O2 letter readout: only the four tokens A-D allowed after prompt + FORCED_PREFILL (allowed_token_ids), one greedy token. Think O2 reads the same way after the thinking stage.
python scripts/think/think_run.py --model qwen3-14b --gpu 3 --out runs/p2/think_pilot --n 20
"""
import argparse, hashlib, json, os, subprocess, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts"), str(ROOT / "scripts" / "p2"), str(ROOT / "scripts" / "d56")]
import p2_models as P2M  # noqa: E402

VENV = Path("$PROJECT_ROOT/venv_vllm")
SUBSET = Path("$PROJECT_ROOT/runs/v5/d42/subset200.jsonl")
SUBSET_SHA = "f13ee9de32403f74cfe8778d06b0d7773611b8491c0876921dc91ae972034805"
EVALK = ROOT / "runs" / "v5" / "evalk" / "evalk.jsonl"
REG = {"C0_system": "0e8748db83d213cc510d49c7ca7974c8dbc541a7a8b6b8c6d1067782c3282803",
       "oracle_prefix": "d42a8b777cce6bdc2a4a4d3bea5d3f51d85a6bc8bb2fed71f20f5d10061e8dfe"}  # release registered_prompts.md
THINK_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
THINK_MAX, TOTAL_MAX = 3840, 4096
PORT = 8731


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


SEED_SALT = os.environ.get("THINK_SEED_SALT", "")  # D126-d: second seed (empty = the same seed as before)


def seed_of(*parts):
    if SEED_SALT:
        parts = parts + (SEED_SALT,)
    return int(sha("|".join(map(str, parts)))[:8], 16) % (2**31)


class Server:
    def __init__(self, model, gpu, log, max_len=8192, port=None):
        s = P2M.SPECS[model]
        self.path = Path(P2M.CACHE) / f"models--{s['repo'].replace('/', '--')}" / "snapshots" / s["revision"]
        assert self.path.exists(), self.path
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "VLLM_CACHE_ROOT": "$PROJECT_ROOT/.cache/vllm", "TMPDIR": "$PROJECT_ROOT/.cache/tmp_vllm",
               "HF_HOME": "$PROJECT_ROOT/.cache/hf_vllm", "XDG_CACHE_HOME": "$PROJECT_ROOT/.cache"}
        self.port = port or PORT
        cmd = [str(VENV / "bin" / "vllm"), "serve", str(self.path), "--served-model-name", "m", "--port", str(self.port), "--dtype", "bfloat16",
               "--max-model-len", str(max_len), "--gpu-memory-utilization", "0.90", "--seed", "0", "--generation-config", "vllm",
               "--logprobs-mode", "processed_logprobs", "--max-logprobs", "20"]
        self.cmd, self.t0 = cmd, time.time()
        self.p = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)

    def wait(self, timeout=1800):
        while time.time() - self.t0 < timeout:
            if self.p.poll() is not None:
                raise SystemExit(f"vllm server exited rc {self.p.returncode}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5)
                return round(time.time() - self.t0, 1)
            except Exception:
                time.sleep(5)
        raise SystemExit("vllm server not ready")

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(120)
        except subprocess.TimeoutExpired:
            self.p.kill()


class ExtServer:
    """D99-r8: uses a vLLM server that is already up (started by r7_seq); it neither starts nor stops it."""

    def __init__(self, port):
        self.port, self.cmd = port, f"external vllm server on port {port}"

    def wait(self, timeout=60):
        urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=timeout)
        return 0.0

    def stop(self):
        pass


HTTP_TIMEOUT = int(os.environ.get("THINK_HTTP_TIMEOUT", "1800"))  # D99-r8e: for long thinking requests (default = as before)


def clean_jsonl(p):
    """D99-r8e resume: drops lines that do not parse (such as a last line cut by an interruption) and rewrites the file. Returns the list of rows."""
    if not p.exists():
        return []
    good = []
    for l in open(p):
        try:
            good.append(json.loads(l))
        except Exception:
            pass
    with open(p, "w") as f:
        for r in good:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return good


def complete(ids, max_tokens, **kw):
    body = {"model": "m", "prompt": ids, "max_tokens": max_tokens, "skip_special_tokens": True, "return_token_ids": True, **kw}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for k in range(5):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=HTTP_TIMEOUT).read())
            return r["choices"][0]
        except Exception:
            if k == 4:
                raise
            time.sleep(5 * (k + 1))


class Engine:
    """Tokenizer and rendering (same as the earlier P2 wrapper) and generation requests. mode: instant | think."""

    def __init__(self, model):
        from transformers import AutoTokenizer
        from forced_answer import letter_ids
        import panel_necessity as PN
        s = P2M.SPECS[model]
        self.model = model
        self.tok = AutoTokenizer.from_pretrained(s["repo"], revision=s["revision"], cache_dir=P2M.CACHE, local_files_only=True)
        self.kw = {m: {**s.get("template_kwargs", {}), "enable_thinking": m == "think"} for m in ("instant", "think")}
        assert s.get("template_kwargs") == P2M.NO_THINK, "Qwen3-family models only (enable_thinking supported)"
        self.merge = {m: not P2M.template_ok(self.tok, self.kw[m]) for m in self.kw}
        self.FORCED = PN.FORCED_PREFILL
        self.letters = letter_ids(self.tok, self.render([{"role": "system", "content": "s"}, {"role": "user", "content": "x"}], "instant") + self.FORCED)
        self.end_think = self.tok.convert_tokens_to_ids("</think>")
        assert isinstance(self.end_think, int) and self.tok.decode([self.end_think]) == "</think>"

    def render(self, msgs, mode):
        return P2M.render(self.tok, msgs, self.merge[mode], self.kw[mode])

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def think(self, ids, tag):
        """Thinking stage. Returns (tokens to append after the thinking, record)."""
        c = complete(ids, THINK_MAX, stop_token_ids=[self.end_think], include_stop_str_in_output=True, seed=seed_of(self.model, "think", tag, sha(json.dumps(ids))), **THINK_SAMPLING)
        gen = list(c.get("token_ids") or self.ids(c["text"]))
        closed = len(gen) > 0 and gen[-1] == self.end_think
        tail = self.ids("\n\n") if closed else self.ids("\n</think>\n\n")
        rec = {"think_tokens": len(gen), "think_truncated": not closed, "finish_reason": c.get("finish_reason"), "think_text": self.tok.decode(gen, skip_special_tokens=False)}
        if not closed and c.get("finish_reason") == "stop":  # ended at EOS (for example, the answer was finished within the thinking stage)
            rec["ended_eos_in_think"] = True
        return gen + tail, rec

    def letter(self, ids):
        c = complete(ids, 1, temperature=0.0, allowed_token_ids=self.letters, logprobs=4)
        tid = (c.get("token_ids") or [None])[0]
        L = "ABCD"[self.letters.index(tid)] if tid in self.letters else c["text"].strip()[:1]
        return {"letter": L, "top_logprobs": (c.get("logprobs") or {}).get("top_logprobs")}


class Chat:
    """model.chat(msgs, prefill, max_new) for agent.run_episode -> (text, n_in, n_out, truncated). Each call is logged in self.calls."""

    def __init__(self, eng, mode, tag):
        self.e, self.mode, self.tag, self.calls = eng, mode, tag, []

    def chat(self, msgs, prefill, max_new):
        e = self.e
        ids = e.ids(e.render(msgs, self.mode))
        rec = {"n_in": len(ids)}
        if self.mode == "think":
            add, trec = e.think(ids, self.tag); rec.update(trec)
            ids2 = ids + add + e.ids(prefill)
            c = complete(ids2, max_new, seed=seed_of(e.model, "answer", self.tag, sha(json.dumps(ids2))), **THINK_SAMPLING)
        else:
            ids2 = ids + e.ids(prefill)
            c = complete(ids2, max_new, temperature=0.0)
        gen = c.get("token_ids") or e.ids(c["text"])
        text = prefill + c["text"].strip()
        end = P2M.first_json_end(text)
        if end is not None:
            text, truncated = text[:end], False
        else:
            truncated = c.get("finish_reason") == "length"
        rec.update(answer_tokens=len(gen), answer_finish=c.get("finish_reason"), stopped_after_json=end is not None)
        self.calls.append(rec)
        return text, len(ids), len(gen), truncated


def load_items(n, which="subset200"):
    """which: subset200 (D42, default) | full (evalk 736 items in order) | rest (full - subset200, evalk order; D99-r6)."""
    assert sha(SUBSET.read_bytes()) == SUBSET_SHA
    sub = [json.loads(l) for l in open(SUBSET)]
    evl = [json.loads(l) for l in open(EVALK)]
    ev = {x["hash"]: x for x in evl}
    assert len(ev) == len(evl) == 736
    for x in sub:
        assert all(ev[x["hash"]][k] == x[k] for k in ("question", "pert_question", "options", "removed_facts", "removed_fact")), x["hash"]
    if which == "subset200":
        out = [ev[x["hash"]] for x in sub]
    elif which == "full":
        out = evl
    elif which == "rest":
        hs = {x["hash"] for x in sub}
        out = [x for x in evl if x["hash"] not in hs]
        assert len(out) == 536
    else:
        raise ValueError(which)
    return out[:n]


def prompt_checks(cfg):
    import agent, prompts_d56 as P
    got = {"C0_system": sha(agent.system_prompt("C0", cfg)), "oracle_prefix": sha(P.ORACLE_PREFIX)}
    assert got == REG, got
    return got


def main():
    global THINK_MAX, PORT
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--gpu", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--modes", default="instant,think"); ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--think-max", type=int, default=3840); ap.add_argument("--max-model-len", type=int, default=8192)  # D99-r5; default = D98 behaviour
    ap.add_argument("--items", default="subset200", choices=["subset200", "full", "rest"])  # D99-r6
    ap.add_argument("--port", type=int, default=PORT); ap.add_argument("--external-server", action="store_true")  # D99-r8
    ap.add_argument("--resume", action="store_true")  # D99-r8e: keep finished rows, add only the missing ones
    a = ap.parse_args()
    THINK_MAX, PORT = a.think_max, a.port
    import agent
    from consol_requery import consol_messages
    import prompts_d56 as P
    out = Path(a.out) / a.model
    if out.exists() and not a.resume:
        sys.exit(f"{out} exists (no resume)")
    out.mkdir(parents=True, exist_ok=True)
    prev = json.loads((out / "meta.json").read_text()) if a.resume and (out / "meta.json").exists() else {}
    cfg = agent.load_config(allow_provisional=False)
    items = load_items(a.n, a.items)
    meta = {"items": a.items, "model": a.model, "spec": P2M.SPECS[a.model], "gpu": a.gpu, "n_items": len(items), "subset_sha256": SUBSET_SHA, "prompt_sha256": prompt_checks(cfg),
            "think_sampling": THINK_SAMPLING, "think_max": THINK_MAX, "max_model_len": a.max_model_len, "total_max": TOTAL_MAX, "max_new": cfg["max_new_tokens"], "workers": a.workers,
            "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    if prev:
        meta.update({k: v for k, v in prev.items() if k.endswith("_seconds")}); meta["start_utc"] = prev.get("start_utc")
        meta["resumed_utc"] = prev.get("resumed_utc", []) + [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())]
    eng = Engine(a.model)
    srv = ExtServer(a.port) if a.external_server else Server(a.model, a.gpu, out / "server.log", a.max_model_len)
    meta["server_cmd"] = srv.cmd
    try:
        meta["server_ready_seconds"] = srv.wait()
        meta["vllm_version"] = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/version", timeout=10).read()).get("version")
        lock = threading.Lock()
        for mode in a.modes.split(","):
            if f"{mode}_seconds" in meta:
                print(mode, "already done (resume)", flush=True); continue
            have = {k: {r["hash"] for r in clean_jsonl(out / f"{mode}_{k}.jsonl")} if a.resume else set() for k in ("C0_S0", "C0_S1K", "O2")}
            fo = {k: open(out / f"{mode}_{k}.jsonl", "a" if a.resume else "w") for k in ("C0_S0", "C0_S1K", "O2")}

            def c0(it, st):
                ch = Chat(eng, mode, f"{it['hash']}|C0|{st}")
                s = time.time()
                tr = agent.run_episode({k: it[k] for k in ("id", "hash", "question", "pert_question", "removed_fact", "removed_type", "options")},
                                       "S0" if st == "S0" else "S1", "C0", ch, cfg)
                tr.update(state=st, mode=mode, engine="vllm", calls=ch.calls, seconds=round(time.time() - s, 2))
                return f"C0_{st}", tr

            def o2(it):
                f1, f2 = (x.strip() for x in it["removed_facts"])
                x = {**it, "pert_question": it["pert_question"] + P.ORACLE_SEP + P.ORACLE_PREFIX + f"{f1} {f2}"}
                s = time.time()
                prompt = eng.render(consol_messages(x, "S1", [], cfg), mode)
                rec = {"hash": it["hash"], "mode": mode, "engine": "vllm"}
                ids = eng.ids(prompt)
                if mode == "think":
                    add, trec = eng.think(ids, f"{it['hash']}|O2"); rec.update(trec)
                    ids = ids + add
                else:
                    rec["prompt_sha256"] = sha(prompt + eng.FORCED)  # must equal the prompt_sha256 of the run_eval oracle
                rec.update(eng.letter(ids + eng.ids(eng.FORCED)), n_in=len(ids), seconds=round(time.time() - s, 2))
                return "O2", rec

            t0 = time.time(); done = 0
            with ThreadPoolExecutor(a.workers) as ex:
                futs = [ex.submit(c0, it, st) for it in items for st in ("S0", "S1K") if it["hash"] not in have[f"C0_{st}"]] + [ex.submit(o2, it) for it in items if it["hash"] not in have["O2"]]
                if a.resume:
                    print(f"{mode} resume: have {({k: len(v) for k, v in have.items()})}, todo {len(futs)}", flush=True)
                for f in as_completed(futs):
                    k, r = f.result()
                    with lock:
                        fo[k].write(json.dumps(r, ensure_ascii=False) + "\n"); fo[k].flush(); done += 1
                    if done % 50 == 0:
                        print(f"{mode} {done}/{len(futs)} {time.time() - t0:.0f}s", flush=True)
            for f in fo.values():
                f.close()
            meta[f"{mode}_seconds"] = round(time.time() - t0, 1)
            (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
            print(mode, "done", meta[f"{mode}_seconds"], flush=True)
    finally:
        srv.stop()
        meta["end_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
