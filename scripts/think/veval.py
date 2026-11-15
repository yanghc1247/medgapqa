"""D130/D130-a: Phase 2 runner that generates with vLLM for the evaluated models (new code; frozen and registered files are only imported and overridden at run time).
Modes: instant (greedy) | think (T 0.6, top_p 0.95, top_k 20, per-request seed = think_run.seed_of(model, stage, tag, input sha[, THINK_SEED_SALT]), thinking budget 32,768).
D130-a: no max_tokens is set for any generation of the evaluated model (targeted-ask question generation and re-request, free-agent and forced-inquiry action turns, C0 answer turn, the answer part after thinking in Think)
(it stops only at EOS, when the first JSON object closes (D74-r4 stop rule: streamed, the request is cut when the first JSON object closes and the string is cut at its closing brace), or at the context end).
pipeline_k.MAX_NEW_GEN and the agent setting max_new_tokens are ignored inside this runner (the registered files configs/agent.json and src/medgapqa/pipeline_k.py are not modified).
Letter readout: only the four tokens A-D are allowed (allowed_token_ids) + logprobs 4 (processed_logprobs) -> it checks and records that all four values are returned. Think reads FORCED_PREFILL after thinking (as in the earlier Think runs).
Forced-inquiry tool choice (agent_c4f_v3.choose_tool): the same definition as before (summed log probability of the ASK and SEARCH continuations after PREFILL), computed with vLLM prompt_logprobs.
  In Think, the thinking of that turn is generated once and used for both the tool choice and the action.
Multi-turn: the history holds only actions (answer parts) and observations (chat() returns the answer without the thinking = the official Qwen3 way).
Simulator: the frozen c1 server (transformers) unchanged, with a disk cache of its classifications (key = simulator setting hash + SimulatorK cache key [visible item, two facts, question text]).
Output: the same layout as run_eval.py (<out>/<model>/{c0,c4,c4f,p6}/..., oracle.jsonl, np_probe.jsonl) -> analysed directly with p2_analysis --root.
python scripts/think/veval.py --model qwen3-14b --mode instant --port 8851 --sim-ports 8861 --out runs/p2/eval_vllm_instant --cells C4,C4f --n 100
"""
import argparse, hashlib, json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts"), str(ROOT / "scripts" / "p2"), str(ROOT / "scripts" / "d56"), str(ROOT / "scripts" / "d58"), str(ROOT / "scripts" / "think")]
import think_run as TR  # noqa: E402
import p2_models as P2M  # noqa: E402

ITEMS = ROOT / "runs" / "v5" / "evalk" / "evalk.jsonl"
ITEMS_SHA = "2da8bef3ed4176ce86f6734bd65877d8e6da884034b9b40c97d149a0a77191bd"
SIM_REV = "495f39366efef23836d0cfae4fbe635880d2be31"
SIM_CACHE = ROOT / "runs" / "p2" / "sim_cache" / "c1_v10.jsonl"
STATS_LOCK = threading.Lock()
MAX_MODEL_LEN = [None]  # filled from the server /v1/models in main()
STATS = {"calls": 0, "answer_finish": {}, "think_capped": 0, "context_end": 0, "json_stop": 0, "letters_missing": 0, "sim_retries": 0}


def stat(k, sub=None):
    with STATS_LOCK:
        if sub is None:
            STATS[k] += 1
        else:
            STATS[k][sub] = STATS[k].get(sub, 0) + 1


def post(body, stream=False, timeout=None):
    req = urllib.request.Request(f"http://127.0.0.1:{TR.PORT}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout or TR.HTTP_TIMEOUT)


def gen_stream(ids, prefill, max_tokens=None, **kw):
    """Streaming generation. max_tokens None = no limit (context end). When the first JSON object closes, the connection is cut to stop the request (the same output as the D74-r4 stop rule).
    Returns (text_without_prefill, n_tokens, finish), finish in {stop, json, length, other}."""
    body = {"model": "m", "prompt": ids, "stream": True, "skip_special_tokens": True, "return_token_ids": True, **kw}
    # In vLLM completions, max_tokens: null is treated as the default 16 (checked). 'No limit' = up to the context end = max_model_len - input length.
    body["max_tokens"] = max_tokens if max_tokens is not None else max(1, MAX_MODEL_LEN[0] - len(ids))
    for k in range(5):
        text, ntok, finish = "", 0, "other"
        try:
            r = post(body, stream=True)
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ch = json.loads(data)["choices"][0]
                text += ch.get("text") or ""
                ntok += len(ch.get("token_ids") or []) or (1 if ch.get("text") else 0)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                if P2M.first_json_end(prefill + text) is not None:
                    finish = "json"; r.close(); break
            return text, ntok, finish
        except Exception:
            if k == 4:
                raise
            time.sleep(5 * (k + 1))


CTX_MARGIN = 64  # position of FORCED_PREFILL and the first answer token after thinking


def think_call(e, ids, tag):
    """D130 r2: thinking budget = min(32,768, remaining context - margin). If the remaining context lowers it, record context_limited (= stopped at the context end)."""
    room = MAX_MODEL_LEN[0] - len(ids) - CTX_MARGIN
    if room >= TR.THINK_MAX:
        return e.think(ids, tag)
    stat("context_end")
    if room < 16:
        tail = [e.end_think] if isinstance(e, DsEngine) else e.ids("\n</think>\n\n")
        return tail, {"think_tokens": 0, "think_truncated": True, "finish_reason": "context", "think_text": "", "context_limited": True}
    c = TR.complete(ids, room, stop_token_ids=[e.end_think], include_stop_str_in_output=True, seed=TR.seed_of(e.model, "think", tag, TR.sha(json.dumps(ids))), **TR.THINK_SAMPLING)
    gen = list(c.get("token_ids") or e.ids(c["text"]))
    closed = len(gen) > 0 and gen[-1] == e.end_think
    if isinstance(e, DsEngine):
        tail = [] if closed else [e.end_think]
    else:
        tail = e.ids("\n\n") if closed else e.ids("\n</think>\n\n")
    return gen + tail, {"think_tokens": len(gen), "think_truncated": not closed, "finish_reason": c.get("finish_reason"), "think_text": e.tok.decode(gen, skip_special_tokens=False),
                        "context_limited": True}


RETRY_ECHO_MAX = 1024  # D130-k
_RETRY_PREFIX = "Your previous output was not valid. Output exactly one JSON object and nothing else:"  # start of the frozen pipeline_k.GEN_RETRY


def clip_retry_echo(e, msgs):
    """D130-k: only for the question-generation format re-request (pipeline_k.GEN_RETRY), the failed output that is echoed back (assistant message) is clipped to its first 1,024 tokens.
    The re-request wording and message structure are unchanged. Returns (msgs, clipped, original token count)."""
    if len(msgs) >= 2 and msgs[-1].get("role") == "user" and str(msgs[-1].get("content", "")).startswith(_RETRY_PREFIX) and msgs[-2].get("role") == "assistant":
        ids = e.ids(msgs[-2]["content"])
        if len(ids) > RETRY_ECHO_MAX:
            clipped = e.tok.decode(ids[:RETRY_ECHO_MAX], skip_special_tokens=False)
            return msgs[:-2] + [{"role": "assistant", "content": clipped}, msgs[-1]], True, len(ids)
        return msgs, False, len(ids)
    return msgs, False, None


class VModel:
    """The model interface called by agent.run_episode, pipeline_k.generate and forced_answer (chat, letters) + tool_logprob for the C4f choose_tool. A new one per item and cell (tag = seed)."""

    def __init__(self, eng, mode, tag, cap=None):
        self.e, self.mode, self.tag, self.cap, self.calls, self._think = eng, mode, tag, cap, [], {}

    def base(self, msgs, kind):
        """Input tokens with the thinking appended. Think thinks only once for the same input (msgs) (the C4f tool choice and the action use the same thinking)."""
        e = self.e
        ids = e.ids(e.render(msgs, self.mode))
        rec = {"n_in": len(ids), "kind": kind}
        if self.mode != "think":
            return ids, rec
        key = TR.sha(json.dumps(ids))
        if key not in self._think:
            add, trec = think_call(e, ids, self.tag)
            if trec.get("think_truncated"):
                stat("think_capped")
            self._think[key] = (add, trec)
        add, trec = self._think[key]
        rec.update({k: v for k, v in trec.items() if k != "think_text"}, think_text=trec["think_text"])
        return ids + add, rec

    def chat(self, msgs, prefill, max_new):
        e = self.e
        msgs, clipped, echo_n = clip_retry_echo(e, msgs)
        ids, rec = self.base(msgs, "chat")
        if echo_n is not None:
            rec.update(retry_echo_tokens=echo_n, retry_echo_clipped=clipped)
        ids2 = ids + e.ids(prefill)
        if len(ids2) >= MAX_MODEL_LEN[0] - 1:  # D130 r2: the input already fills the context -> recorded as cut at the context end without generation
            stat("calls"); stat("answer_finish", "context_prompt"); stat("context_end")
            rec.update(answer_tokens=0, answer_finish="context_prompt", stopped_after_json=False, prefill=prefill); self.calls.append(rec)
            return prefill, rec["n_in"], 0, True
        kw = ({"seed": TR.seed_of(e.model, "answer", self.tag, TR.sha(json.dumps(ids2))), **TR.THINK_SAMPLING} if self.mode == "think" else {"temperature": 0.0})
        kw.update(getattr(e, "gen_kw", {}))
        text, ntok, finish = gen_stream(ids2, prefill, self.cap, **kw)
        stat("calls"); stat("answer_finish", finish)
        if finish == "length":
            stat("context_end")
        full = prefill + text.strip()
        end = P2M.first_json_end(full)
        truncated = False if end is not None else finish == "length"
        if end is not None:
            full = full[:end]
        rec.update(answer_tokens=ntok, answer_finish=finish, stopped_after_json=end is not None, prefill=prefill)
        self.calls.append(rec)
        return full, rec["n_in"], ntok, truncated

    def letters(self, msgs):
        import numpy as np
        from forced_answer import logits4_sha
        e = self.e
        ids, rec = self.base(msgs, "letters")
        r = e.letter(ids + e.ids(e.FORCED))
        tl = r.get("top_logprobs") or [{}]
        top = tl[0] if tl else {}
        lp = []
        for t in e.letters:
            s = e.tok.decode([t])
            v = top.get(s)
            if v is None:  # in case a key is not a decoded string
                v = next((x for k, x in top.items() if k.strip() == s.strip()), None)
            lp.append(v)
        if any(v is None for v in lp):
            stat("letters_missing")
        four = np.array([v if v is not None else -1e9 for v in lp], float)
        p = np.exp(four - four.max()); p /= p.sum()
        rec.update(letter=r["letter"]); self.calls.append(rec)
        prompt = e.render(msgs, self.mode)
        return {"letter": "ABCD"[int(np.argmax(four))] if all(v is not None for v in lp) else r["letter"], "logits4": [float(x) for x in four], "n_in": len(ids),
                "prompt_sha256": TR.sha(prompt + e.FORCED), "p4max": float(p.max()), "logits4_sha256": logits4_sha([float(x) for x in four]),
                "letters_complete": all(v is not None for v in lp), **({"think_tokens": rec.get("think_tokens"), "think_truncated": rec.get("think_truncated")} if self.mode == "think" else {})}

    def tool_logprob(self, msgs, prefill, conts):
        """C4f choose_tool: summed log probability of each continuation after PREFILL (prompt_logprobs)."""
        e = self.e
        ids, rec = self.base(msgs, "choose_tool")
        p_ids = e.ids(prefill)
        out = {}
        for a, c in conts.items():
            f = e.ids(prefill + c)
            assert f[:len(p_ids)] == p_ids, "tokenization boundary mismatch"
            full = ids + f
            body = {"model": "m", "prompt": full, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0}
            for k in range(5):
                try:
                    res = json.loads(post(body).read())["choices"][0]; break
                except Exception:
                    if k == 4:
                        raise
                    time.sleep(5 * (k + 1))
            pl = res["prompt_logprobs"]
            s = 0.0
            for pos in range(len(ids) + len(p_ids), len(full)):
                d = pl[pos]; tid = full[pos]
                v = d.get(str(tid)) if isinstance(d, dict) else None
                s += float(v["logprob"] if isinstance(v, dict) else v)
            out[a] = s
        self.calls.append({**rec, "tool_logprob": out})
        return out


class DsEngine(TR.Engine):
    """D130: DeepSeek-V4-Flash-0731. No Jinja template -> the official encoder (src/medgapqa/vendor/encoding_dsv4.py). Instant = thinking_mode "chat" (as in D82),
    Think = thinking_mode "thinking", default reasoning_effort ("low"; the official default). The encoder drops earlier turns' thinking (drop_thinking=True; no thinking in the history)."""

    def __init__(self, model):
        from transformers import AutoTokenizer
        from forced_answer import letter_ids
        import panel_necessity as PN
        s = P2M.SPECS[model]
        self.model = model
        snap = Path(P2M.CACHE) / f"models--{s['repo'].replace('/', '--')}" / "snapshots" / s["revision"]
        self.tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        sys.path.insert(0, str(ROOT / "src" / "medgapqa" / "vendor"))
        from encoding_dsv4 import encode_messages
        self._enc = encode_messages
        self.FORCED = PN.FORCED_PREFILL
        self.letters = letter_ids(self.tok, self.render([{"role": "system", "content": "s"}, {"role": "user", "content": "x"}], "instant") + self.FORCED)
        self.end_think = self.tok.convert_tokens_to_ids("</think>")
        assert isinstance(self.end_think, int) and self.tok.decode([self.end_think]) == "</think>"

    def render(self, msgs, mode):
        return self._enc([dict(m) for m in msgs], thinking_mode="thinking" if mode == "think" else "chat")

    def think(self, ids, tag):
        """Same as TR.Engine.think, except that the separator after thinking follows the official format ({reasoning}</think>{content}: no extra newline)."""
        c = TR.complete(ids, TR.THINK_MAX, stop_token_ids=[self.end_think], include_stop_str_in_output=True, seed=TR.seed_of(self.model, "think", tag, TR.sha(json.dumps(ids))), **TR.THINK_SAMPLING)
        gen = list(c.get("token_ids") or self.ids(c["text"]))
        closed = len(gen) > 0 and gen[-1] == self.end_think
        tail = [] if closed else [self.end_think]
        rec = {"think_tokens": len(gen), "think_truncated": not closed, "finish_reason": c.get("finish_reason"), "think_text": self.tok.decode(gen, skip_special_tokens=False)}
        if not closed and c.get("finish_reason") == "stop":
            rec["ended_eos_in_think"] = True
        return gen + tail, rec


EXTRA_SPECS = {"medgemma-4b": {"repo": "google/medgemma-4b-it", "revision": "290cda5eeccbee130f987c4ad74a59ae6f196408", "template_kwargs": {}, "extra_eos": ["<end_of_turn>"]}}  # same pinned revision as D15 and parse_check


def spec_of(model):
    return P2M.SPECS.get(model) or EXTRA_SPECS[model]


class GemmaEngine(TR.Engine):
    """D130: MedGemma-4B and 27B (Instant only; the template has no thinking switch). Rendering as in the earlier P2 wrapper (without a system role, the system text goes before the first user turn).
    <end_of_turn> is a stop token in every generation request (the earlier extra_eos); for MedGemma-27B the thought markers <unused94> and <unused95> are blocked (the earlier D74-r10 bad_words)."""

    def __init__(self, model):
        from transformers import AutoTokenizer
        from forced_answer import letter_ids
        import panel_necessity as PN
        s = spec_of(model)
        self.model = model
        snap = Path(P2M.CACHE) / f"models--{s['repo'].replace('/', '--')}" / "snapshots" / s["revision"]
        self.tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        self.kw = {"instant": {}}
        self.merge = {"instant": not P2M.template_ok(self.tok, {})}
        self.FORCED = PN.FORCED_PREFILL
        self.letters = letter_ids(self.tok, self.render([{"role": "system", "content": "s"}, {"role": "user", "content": "x"}], "instant") + self.FORCED)
        stop = [self.tok.convert_tokens_to_ids(t) for t in s.get("extra_eos", [])]
        self.gen_kw = {"stop_token_ids": [t for t in stop if isinstance(t, int)]}
        if s.get("bad_words"):
            self.gen_kw["bad_words"] = list(s["bad_words"])
        self.end_think = None

    def render(self, msgs, mode):
        assert mode == "instant", "MedGemma has no thinking mode"
        return P2M.render(self.tok, msgs, self.merge["instant"], {})


class DiskSimCache:
    """Disk cache of SimulatorK.cache (classifications). Key = setting hash + SimulatorK key. Only rows with the same setting are read."""

    def __init__(self, simk, cfg_hash, path=SIM_CACHE):
        self.simk, self.h, self.path, self.lock = simk, cfg_hash, path, threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        if path.exists():
            for l in open(path):
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                if r["cfg"] == cfg_hash:
                    simk.cache[r["key"]] = r["val"]; n += 1
        self.loaded = n
        orig = simk.classify

        def classify(visible, facts, question, _o=orig):
            f1, f2 = (facts + [None, None])[:2] if facts else (None, None)
            key = hashlib.sha256(json.dumps([visible, f1, f2, question]).encode()).hexdigest()
            c, hit = _o(visible, facts, question)
            if not hit:
                with self.lock, open(self.path, "a") as fo:
                    fo.write(json.dumps({"cfg": self.h, "key": key, "val": c}, ensure_ascii=False) + "\n")
            return c, hit
        simk.classify = classify


def sim_setup(ports):
    from instances import PooledBackend
    from llm_client import Client, RemoteGenerator
    import simulator_k
    from simulator_k import SimulatorK
    health = {}
    for p in ports:
        for k in range(30):  # D130 r3: on a busy single-threaded server, /health times out at 5 s -> retry
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{p}/health", timeout=120) as r_:  # D130 r3: a busy server answers after its queue
                    h = json.loads(r_.read()); break
            except (ConnectionError, TimeoutError, OSError):
                if k == 29:
                    raise
                time.sleep(10)
        assert h.get("model") == "c1" and h.get("revision") == SIM_REV, h; health[p] = h
    cfg_hash = TR.sha(json.dumps({"rev": SIM_REV, "prompt_sha": simulator_k.prompt_sha(), "bundle": health[ports[0]].get("v10_bundle_sha"),
                                  "decoding": health[ports[0]].get("decoding"), "max_new": health[ports[0]].get("max_new_tokens"), "quant": health[ports[0]].get("quant")}, sort_keys=True))
    simk = SimulatorK(PooledBackend([RemoteGenerator(p) for p in ports], [f"sim:{p}" for p in ports]))
    dc = DiskSimCache(simk, cfg_hash)
    return simk, health, cfg_hash, dc


def sim_respond(simk, mode, visible, facts, question, tries=8):
    """D130 r3: the c1 server (single-threaded HTTPServer, backlog 5) refuses or times out when many connections arrive at once. Requests are deterministic (greedy) and free of side effects, so they are resent after an exponential wait."""
    import random as _r
    for k in range(tries):
        try:
            return simk.respond(mode, visible, facts, question)
        except (ConnectionError, TimeoutError, OSError) as ex:
            if k == tries - 1:
                raise
            stat("sim_retries")
            time.sleep(min(60, 2 ** k) + _r.random())


class SimAdapter:
    MODE = {"s0": "s0", "recoverable": "s1k", "unrecoverable": "s2k"}

    def __init__(self, simk, facts):
        self.simk, self.facts = simk, facts

    def respond(self, mode, visible, withheld, question):
        return sim_respond(self.simk, self.MODE[mode], visible, self.facts, question)


def patched_choose_tool(model, msgs):
    from agent_c4f_v3 import CONT
    from agent import PREFILL
    s = model.tool_logprob(msgs, PREFILL, CONT)
    return {"choice": "ASK" if s["ASK"] >= s["SEARCH"] else "SEARCH", "logprob": s}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--mode", choices=["instant", "think"], required=True)
    ap.add_argument("--port", type=int, required=True); ap.add_argument("--sim-ports", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--cells", default="C0,oracle,NP,C4,P6,C4f"); ap.add_argument("--n", type=int, default=736); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--cap", type=int, default=None, help="limit for the answer part (default none = D130-a)"); ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--states", default="S1K,S2K", help="state selection for C4 and C4f (P6 not affected; for checks)")
    ap.add_argument("--shard", default="", help="k/K: only items with i %% K == k in item order (D130-l speed-up)")
    ap.add_argument("--done-root", default="", help="skip items already in the same cell file under this root (no overlap with the main run)")
    a = ap.parse_args()
    TR.PORT, TR.THINK_MAX = a.port, 32768
    TR.HTTP_TIMEOUT = max(TR.HTTP_TIMEOUT, 10800)
    assert hashlib.sha256(ITEMS.read_bytes()).hexdigest() == ITEMS_SHA
    import agent, pipeline_k, prompts_c4f, prompts_p6, agent_c4f_v3, abstain_probe
    from run_pilot import check_hashes
    from forced_answer import forced_messages
    from consol_requery import consol_messages
    check_hashes()
    P2M.install()
    cfg = agent.load_config(allow_provisional=False)
    items = [json.loads(l) for l in open(ITEMS)][:a.n]
    if a.shard:
        k, K = map(int, a.shard.split("/")); items = [it for i, it in enumerate(items) if i % K == k]
    sp = spec_of(a.model)
    eng = DsEngine(a.model) if sp.get("encoder") == "dsv4_chat" else (GemmaEngine(a.model) if "medgemma" in a.model else TR.Engine(a.model))
    out = Path(a.out) / a.model; out.mkdir(parents=True, exist_ok=True)
    vinfo = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/version", timeout=10).read())
    minfo = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=10).read())
    MAX_MODEL_LEN[0] = int((minfo.get("data") or [{}])[0]["max_model_len"])
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"model": a.model, "mode": a.mode, "engine": "vllm", "cells": {}}
    meta.update(vllm=vinfo, max_model_len=(minfo.get("data") or [{}])[0].get("max_model_len"), cap=a.cap, think_max=32768 if a.mode == "think" else None,
                think_sampling=TR.THINK_SAMPLING if a.mode == "think" else None, seed_salt=TR.SEED_SALT, items_sha256=ITEMS_SHA, n=len(items),
                batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT_SERVER", "unknown"))
    ports = [int(p) for p in a.sim_ports.split(",") if p]
    simk = None
    if ports:
        simk, health, cfg_hash, dc = sim_setup(ports)
        meta.update(sim_ports=ports, sim_health=health, sim_cfg_hash=cfg_hash, sim_cache_loaded=dc.loaded)
    retriever = None
    lock = threading.Lock()

    def write_meta():
        meta["stats"] = dict(STATS); meta_p.write_text(json.dumps(meta, indent=1, default=str))

    def ep_of(it):
        return {k: it[k] for k in ("id", "hash", "question", "pert_question", "removed_fact", "removed_type", "options")}

    def run_cell(name, path, keys, fn):
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {r["hash"] if "state_k" not in r else (r["hash"]) for r in TR.clean_jsonl(path)} if path.exists() else set()
        if a.done_root:  # D130-l: items already completed in the same cell file of the main run
            rel = path.relative_to(Path(a.out)); other = Path(a.done_root) / rel
            if other.exists():
                done |= {r["hash"] for r in TR.clean_jsonl(other)}
        todo = [it for it in items if it["hash"] not in done]
        t0 = time.time()
        with open(path, "a") as fo, ThreadPoolExecutor(a.workers) as ex:
            for f in as_completed([ex.submit(fn, it) for it in todo]):
                try:
                    r = f.result()
                except Exception as ex2:  # D130 r2: errors of a cell function are recorded as well, and the run continues
                    with open(out / "errors.jsonl", "a") as fe:
                        fe.write(json.dumps({"cell": name, "error": repr(ex2)[:500], "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}) + "\n")
                    continue
                if r is None:
                    continue
                with lock:
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
        meta["cells"][name] = {**meta["cells"].get(name, {}), "seconds": round(time.time() - t0, 1) + meta["cells"].get(name, {}).get("seconds", 0),
                               "todo": len(todo), "end_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
        write_meta(); print(name, meta["cells"][name], flush=True)

    def episode(it, state, cond, sub, runner, st_name):
        m = VModel(eng, a.mode, f"{it['hash']}|{cond}|{st_name}{a.tag_suffix}", a.cap)
        sim = SimAdapter(simk, it["removed_facts"]) if simk else None
        s = time.time()
        try:
            tr = runner(ep_of(it), state, "C4" if cond in ("C4", "C4f") else cond, m, cfg, sim, retriever)
        except Exception as ex:  # D130 r2: an error on one item does not stop the whole cell; no row is written, so the item is retried on resume
            with lock, open(out / "errors.jsonl", "a") as fe:
                fe.write(json.dumps({"hash": it["hash"], "cell": f"{cond}_{st_name}", "error": repr(ex)[:500], "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}) + "\n")
            return None
        if tr["outcome"] in ("abstain", "unresolved"):
            msgs, sha_ok = forced_messages(tr, ep_of(it), cfg)
            r = m.letters(msgs)
            tr["forced_v5"] = {"letter": r["letter"], "logits4": r["logits4"], "logits4_sha256": r["logits4_sha256"], "sha_ok": sha_ok, "letters_complete": r["letters_complete"]}
        for x in tr.get("turns", []):
            if x.get("action") == "SEARCH" and "observation" in x:
                x["observation"] = None
        tr.update(seconds=round(time.time() - s, 2), instance=f"vllm:{a.model}:{a.mode}", state_k=st_name, engine="vllm", mode=a.mode, vcalls=m.calls)
        return tr

    cells = a.cells.split(",")
    if "C0" in cells:
        for st, rs in (("S0", "S0"), ("S1K", "S1")):
            run_cell(f"C0_{st}", out / "c0" / a.model / "C0" / st / "trajectories.jsonl", None, lambda it, st=st, rs=rs: episode(it, rs, "C0", "c0", agent.run_episode, st))
        # C0 S2K = C0 S1K (same prompt, D99-r3): copy the S1K trajectories with only the state changed
        p1, p2 = out / "c0" / a.model / "C0" / "S1K" / "trajectories.jsonl", out / "c0" / a.model / "C0" / "S2K" / "trajectories.jsonl"
        p2.parent.mkdir(parents=True, exist_ok=True)
        with open(p2, "w") as fo:
            for r in TR.clean_jsonl(p1):
                fo.write(json.dumps({**r, "state": "S2", "state_k": "S2K", "copied_from": "S1K (identical prompt)"}, ensure_ascii=False) + "\n")
    if "oracle" in cells:
        import prompts_d56 as P

        def orc(it):
            m = VModel(eng, a.mode, f"{it['hash']}|oracle{a.tag_suffix}", a.cap)
            f1, f2 = (x.strip() for x in it["removed_facts"])
            conds = {"L-S0": (it, "S0"), "L-S1K": (it, "S1")}
            for c, facts in (("O2", f"{f1} {f2}"), ("O1a", f1), ("O1b", f2)):
                conds[c] = ({**it, "pert_question": it["pert_question"] + P.ORACLE_SEP + P.ORACLE_PREFIX + facts}, "S1")
            rec = {"hash": it["hash"], "row_id": it["row_id"], "model": a.model}; s = time.time()
            for c, (x, st) in conds.items():
                mm = VModel(eng, a.mode, f"{it['hash']}|oracle|{c}{a.tag_suffix}", a.cap)
                r = mm.letters(consol_messages(x, st, [], cfg))
                rec[c] = {k: r[k] for k in ("letter", "logits4", "p4max", "prompt_sha256", "letters_complete")}
            rec["seconds"] = round(time.time() - s, 2)
            return rec
        run_cell("oracle", out / "oracle.jsonl", None, orc)
    if "NP" in cells and a.mode == "instant":
        from agent import case_text
        import prompts_d58 as P
        assert P.sha(P.NP_QUESTION) == "1f9431e06753326cda524193b724ec5c2f7f0de4c78fdf24f9140d451e0b15c8"
        base = eng.render([{"role": "user", "content": "x"}], "instant"); ids = {}
        for w in ("Yes", "No"):
            b = eng.ids(base); x = eng.ids(base + w)
            assert x[:len(b)] == b and len(x) == len(b) + 1
            ids[w] = x[-1]

        def np_(it):
            import math
            rec = {"hash": it["hash"], "row_id": it["row_id"], "model": a.model, "token_ids": ids}; s = time.time()
            for st, q in (("S0", it["question"]), ("S1", it["pert_question"])):
                prompt = eng.render([{"role": "user", "content": case_text(q, it["options"]) + P.NP_SEP + P.NP_QUESTION}], "instant")
                c = TR.complete(eng.ids(prompt), 1, temperature=0.0, allowed_token_ids=[ids["Yes"], ids["No"]], logprobs=2)
                top = (c.get("logprobs") or {}).get("top_logprobs", [{}])[0]
                ly, ln = top.get("Yes"), top.get("No")
                py, pn = (math.exp(ly) if ly is not None else None), (math.exp(ln) if ln is not None else None)
                rec[st] = {"logit_yes": ly, "logit_no": ln, "p_yes": py, "p_no": pn, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                           "note": "restricted log-probabilities over {Yes, No}; logit_no - logit_yes equals the raw logit difference"}
            rec["seconds"] = round(time.time() - s, 2)
            return rec
        run_cell("NP", out / "np_probe.jsonl", None, np_)
    if any(c in cells for c in ("C4", "C4f", "P6")):
        from retrieve import Retriever
        from run_agent import load_tok
        retriever = Retriever.load(cfg["search_top_k"], cfg["search_max_tokens"], load_tok(cfg["search_truncation_tokenizer"]))
    if "C4" in cells:
        for st, rs in [x for x in (("S1K", "S1"), ("S2K", "S2")) if x[0] in a.states.split(",")]:
            run_cell(f"C4_{st}", out / "c4" / a.model / "C4" / st / "trajectories.jsonl", None, lambda it, st=st, rs=rs: episode(it, rs, "C4", "c4", agent.run_episode, st))
    if "P6" in cells:
        import targeted_run as T
        pk, gsha = T.p6_setup()
        pk.MAX_NEW_GEN = a.cap  # None = no limit (VModel sets no max_tokens)
        meta["p6_gen_prompt_sha256"] = gsha

        def p6(it):
            m = VModel(eng, a.mode, f"{it['hash']}|P6gen{a.tag_suffix}", a.cap)
            s = time.time()
            g = pk.generate(m, it, "S1K", "Q3")
            qs = (g.get("parsed") or {}).get("questions", [])[:3] if g.get("valid") else []
            rows = []
            for st in ("S1K", "S2K"):
                blocks, asks = [], []
                for q in qs:
                    text, rec = sim_respond(simk, pk.STATES[st] if hasattr(pk, "STATES") and st in pk.STATES else st.lower(), it["pert_question"], it["removed_facts"], q)
                    blocks.append(f"Question asked: {q}\nResponse: {text}")
                    asks.append({"q": q, "label": rec["label"], "kind": rec["kind"], "qtype": rec["qtype"], "cached": rec["cached"]})
                mm = VModel(eng, a.mode, f"{it['hash']}|P6|{st}{a.tag_suffix}", a.cap)
                r = mm.letters(consol_messages(it, "S1", blocks, cfg))
                rows.append({"id": it["id"], "hash": it["hash"], "state": st, "model": a.model, "gens": {"Q3": g},
                             "arms": {"P6": {"asks": asks, "searches": [], "n_blocks": len(blocks), **{k: r[k] for k in ("letter", "logits4", "n_in", "prompt_sha256", "p4max", "logits4_sha256", "letters_complete")}}},
                             "seconds": round(time.time() - s, 2), "instance": f"vllm:{a.model}:{a.mode}", "engine": "vllm", "mode": a.mode, "vcalls": m.calls + mm.calls})
            return rows
        path = out / "p6" / a.model / "arms.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
        done = {r["hash"] for r in TR.clean_jsonl(path)} if path.exists() else set()
        t0 = time.time()
        with open(path, "a") as fo, ThreadPoolExecutor(a.workers) as ex:
            for f in as_completed([ex.submit(p6, it) for it in items if it["hash"] not in done]):
                try:
                    rows = f.result()
                except Exception as ex2:  # D130 r2
                    with open(out / "errors.jsonl", "a") as fe:
                        fe.write(json.dumps({"cell": "P6", "error": repr(ex2)[:500], "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}) + "\n")
                    continue
                with lock:
                    for r in rows:
                        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fo.flush()
        meta["cells"]["P6"] = {"seconds": round(time.time() - t0, 1), "end_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}; write_meta()
    if "C4f" in cells:
        fz = json.load(open(ROOT / "runs" / "v5" / "d36" / "FROZEN_C4F_v3.json"))
        assert prompts_c4f.prompt_sha(cfg) == fz["system_prompt_sha256"]
        agent_c4f_v3.choose_tool = patched_choose_tool
        abstain_probe.system_prompt = prompts_c4f.c4f_system_prompt
        for st, rs in [x for x in (("S1K", "S1"), ("S2K", "S2")) if x[0] in a.states.split(",")]:
            run_cell(f"C4f_{st}", out / "c4f" / a.model / "C4" / st / "trajectories.jsonl", None,
                     lambda it, st=st, rs=rs: episode(it, rs, "C4f", "c4f", agent_c4f_v3.run_episode_c4f_v3, st))
    write_meta()
    print("done", json.dumps(STATS), flush=True)


if __name__ == "__main__":
    main()
