"""D32 resident server for candidate simulators (new file; llm_server.py and perturb.py unchanged). Same HTTP interface, logging and decoding as llm_server.py (greedy, max_new 768, batch 1).

The only differences from v10 (Meerkat-70B) are the per-model chat template and stop tokens (the prompt string is simulator_k's, from the caller, unchanged).
- Chat template: each model tokenizer's apply_chat_template(add_generation_prompt=True) + prefill (the prefix is placed before the assistant reply and included in the returned text) — same as perturb.Generator.chat.
- Stop tokens: perturb.Generator uses [eos_token_id, convert_tokens_to_ids("<|eot_id|>")], but for Qwen the second is None.
  Here the MODEL_EOS[model] tokens and tokenizer.eos_token_id are combined and None removed (C1 Qwen: <|im_end|>, C2 Llama: <|eot_id|>). pad = eos[0].
- Weights: the NF4 saved with save_pretrained (nf4, compute bf16, double_quant False; same settings as Meerkat-70B NF4) is loaded as is.
Usage: CUDA_VISIBLE_DEVICES=N python src/medgapqa/llm_server_c.py --model c1 --port 8714 --log logs/server/c1_gpuN.jsonl
"""
import argparse
import hashlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perturb import Generator  # noqa: E402
import simulator_k  # noqa: E402

MODELS = {"c1": {"path": "$PROJECT_ROOT/models/qwen2.5-72b-nf4", "repo": "Qwen/Qwen2.5-72B-Instruct",
                 "revision": "495f39366efef23836d0cfae4fbe635880d2be31", "eos_tokens": ["<|im_end|>"]},
          "c2": {"path": "$PROJECT_ROOT/models/llama-3.3-70b-nf4", "repo": "meta-llama/Llama-3.3-70B-Instruct",
                 "revision": "6f6073b423013f6a7d4d9f39144961bfbfbc386b", "eos_tokens": ["<|eot_id|>"]}}
MAX_NEW_TOKENS = 768
V10_BUNDLE_SHA = "68f99cd48cf56d7f2533404e1f913a7ee82d7a23025eb41d392c68bd3445e002"


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def eos_ids(tok, tokens):
    ids = [tok.eos_token_id] + [tok.convert_tokens_to_ids(t) for t in tokens]
    out = []
    for i in ids:
        if i is not None and i != tok.unk_token_id and i not in out:
            out.append(i)
    return out


class GeneratorC(Generator):
    def __init__(self, model, device="cuda:0"):
        spec = MODELS[model]
        super().__init__(spec["path"], device=device, max_new_tokens=MAX_NEW_TOKENS, prequantized=True)
        self.eos = eos_ids(self.tok, spec["eos_tokens"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--gpu-label", default="")
    args = ap.parse_args()
    assert simulator_k.prompt_sha() == V10_BUNDLE_SHA
    t0 = time.time()
    gen = GeneratorC(args.model)
    spec = MODELS[args.model]
    info = {"model": args.model, "model_path": spec["path"], "repo": spec["repo"], "revision": spec["revision"], "quant": "saved nf4",
            "decoding": "greedy", "max_new_tokens": MAX_NEW_TOKENS, "batch_size": 1, "gpu": args.gpu_label, "load_seconds": round(time.time() - t0, 1),
            "eos_ids": gen.eos, "eos_tokens": [gen.tok.convert_ids_to_tokens(i) for i in gen.eos], "v10_bundle_sha": V10_BUNDLE_SHA}
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logf = open(args.log, "a")
    logf.write(json.dumps({"event": "start", "time": time.time(), **info}) + "\n")
    logf.flush()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, info)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/generate":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                msgs, prefill = req["messages"], req.get("prefill", "")
                assert isinstance(msgs, list) and all({"role", "content"} <= set(m) for m in msgs)
            except Exception as e:
                return self._send(400, {"error": f"bad request: {e!r}"})
            s = time.time()
            text, n_in, n_out = gen.chat(msgs, prefill)
            res = {"text": text, "in_tokens": n_in, "out_tokens": n_out, "seconds": round(time.time() - s, 3),
                   "req_sha": sha({"messages": msgs, "prefill": prefill}), "resp_sha": sha(text)}
            logf.write(json.dumps({"event": "generate", "time": time.time(),
                                   **{k: res[k] for k in ["req_sha", "resp_sha", "in_tokens", "out_tokens", "seconds"]}}) + "\n")
            logf.flush()
            self._send(200, res)

        def log_message(self, *a):
            pass

    print(json.dumps({"ready": True, "port": args.port, **info}), flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
