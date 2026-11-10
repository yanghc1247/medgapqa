"""Resident-server client; instance startup, checking and shutdown.

python src/medgapqa/llm_client.py start --gpus 0,1,2,3   # start instances after confirming the GPUs are free; self-check when ready
python src/medgapqa/llm_client.py check                  # check output agreement across instances with a fixed request
python src/medgapqa/llm_client.py stop                   # kills only the recorded PIDs
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "runs" / "servers.json"
AVOID_GPUS = {4, 5}  # in use by another user (user instruction)
BASE_PORT = 8700

SELF_CHECK = [
    [{"role": "system", "content": "You are a careful medical exam editor."},
     {"role": "user", "content": "In one sentence, what does an elevated serum lipase suggest?"}],
    [{"role": "user", "content": "List three causes of hyperkalemia, comma-separated, nothing else."}],
    [{"role": "user", "content": "Reply with exactly: OK"}],
]


class Client:
    def __init__(self, port, timeout=600):
        from infra_retry import InfraRetryLog
        self.url = f"http://127.0.0.1:{port}"
        self.timeout = timeout
        self.retry_log = InfraRetryLog()

    def health(self):
        with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
            return json.loads(r.read())

    def chat(self, messages, prefill=""):
        """Only infrastructure failures (connection errors, timeouts) are retried, at most 2 times (src/medgapqa/infra_retry.py)."""
        from infra_retry import call_with_infra_retry
        body = json.dumps({"messages": messages, "prefill": prefill}).encode()

        def once():
            req = urllib.request.Request(self.url + "/generate", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        return call_with_infra_retry(once, log=self.retry_log)


class RemoteGenerator:
    """Same interface as perturb.Generator (__call__(user) -> text, in, out)."""

    def __init__(self, port):
        self.c = Client(port)

    def chat(self, msgs, prefill=""):
        r = self.c.chat(msgs, prefill)
        return r["text"], r["in_tokens"], r["out_tokens"]

    def __call__(self, user):
        from perturb import SYSTEM
        return self.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])


def gpu_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"instances": []}


def check():
    inst = load_state()["instances"]
    results = []
    for msgs in SELF_CHECK:
        shas = {i["port"]: Client(i["port"]).chat(msgs)["resp_sha"] for i in inst}
        results.append({"n_instances": len(shas), "identical": len(set(shas.values())) == 1, "resp_sha": shas})
    ok = all(r["identical"] for r in results)
    print(json.dumps({"self_check_identical": ok, "requests": len(results), "detail": results}, indent=1))
    return ok


def start(gpus):
    used = gpu_used()
    bad = [g for g in gpus if g in AVOID_GPUS or used.get(g, 1) > 1024]
    if bad:
        sys.exit(f"GPU {bad} is busy or reserved; aborting")
    state = load_state()
    py = str(ROOT / ".venv" / "bin" / "python")
    for k, g in enumerate(gpus):
        port = BASE_PORT + g
        log = ROOT / "logs" / "server" / f"gpu{g}.jsonl"
        out = open(ROOT / "logs" / "server" / f"gpu{g}.out", "a")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), HF_HUB_OFFLINE="1")
        p = subprocess.Popen([py, "-u", str(ROOT / "src/medgapqa" / "llm_server.py"), "--port", str(port), "--log", str(log),
                              "--gpu-label", str(g)], stdout=out, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
        state["instances"].append({"gpu": g, "port": port, "pid": p.pid, "started": time.time()})
        STATE.write_text(json.dumps(state, indent=1))
        # the first instance is loaded alone to time disk loading separately
        if k == 0:
            wait_ready([state["instances"][-1]])
    wait_ready(state["instances"])
    for i in state["instances"]:
        i["health"] = Client(i["port"]).health()
    STATE.write_text(json.dumps(state, indent=1))
    print(json.dumps([{"gpu": i["gpu"], "port": i["port"], "load_seconds": i["health"]["load_seconds"]}
                      for i in state["instances"]]))
    check()


def wait_ready(instances, timeout=3600):
    t = time.time()
    pending = list(instances)
    while pending and time.time() - t < timeout:
        for i in list(pending):
            try:
                Client(i["port"]).health()
                pending.remove(i)
            except Exception:
                try:
                    os.kill(i["pid"], 0)
                except ProcessLookupError:
                    sys.exit(f"instance on GPU {i['gpu']} died; see logs/server/gpu{i['gpu']}.out")
        time.sleep(5)
    if pending:
        sys.exit(f"timeout waiting for {pending}")


def stop():
    state = load_state()
    for i in state["instances"]:
        try:
            os.kill(i["pid"], signal.SIGTERM)
            print(f"stopped pid {i['pid']} (GPU {i['gpu']})")
        except ProcessLookupError:
            print(f"pid {i['pid']} already gone")
    STATE.write_text(json.dumps({"instances": []}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["start", "check", "stop"])
    ap.add_argument("--gpus", default="")
    a = ap.parse_args()
    if a.cmd == "start":
        start([int(x) for x in a.gpus.split(",")])
    elif a.cmd == "check":
        sys.exit(0 if check() else 1)
    else:
        stop()
