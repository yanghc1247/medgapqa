"""Retry rule (user decision 2026-10-16, plan §9): only infrastructure failures (server connection errors, timeouts etc.) are retried, at most 2 times.
Retrying based on results (correctness, parse_fail or other output content) is forbidden.

- call_with_infra_retry: retries only exceptions in INFRA_ERRORS and passes other exceptions and normal return values through unchanged.
  It takes no argument that inspects the return value, so retrying based on output content is structurally impossible.
- write_episodes: never overwrites result files. An interrupted run is only continued with --resume, skipping finished items.
"""
import http.client
import json
import socket
import time
import urllib.error
from pathlib import Path

MAX_INFRA_RETRIES = 2
INFRA_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError, socket.timeout, http.client.HTTPException)
try:  # GPU out-of-memory also counts as an infrastructure failure (rerun the same input with the same settings)
    import torch
    INFRA_ERRORS = INFRA_ERRORS + (torch.cuda.OutOfMemoryError,)
except Exception:  # noqa: BLE001
    pass


class InfraRetryLog:
    def __init__(self):
        self.events = []


def call_with_infra_retry(fn, *args, log=None, sleep=2.0, **kwargs):
    for attempt in range(MAX_INFRA_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except INFRA_ERRORS as e:
            if log is not None:
                log.events.append({"attempt": attempt, "error": type(e).__name__, "msg": str(e)[:200]})
            if attempt == MAX_INFRA_RETRIES:
                raise
            time.sleep(sleep * (attempt + 1))


def write_episodes(path, items, episode_fn, resume=False):
    """Runs items once each in order and writes jsonl. If the file exists, only with resume are unfinished items appended.
    There is no path that reruns an item depending on its result."""
    path = Path(path)
    done = set()
    if path.exists():
        if not resume:
            raise SystemExit(f"{path} exists; refusing to overwrite (use --resume to continue an interrupted run)")
        done = {json.loads(line)["id"] for line in open(path) if line.strip()}
    n_new = 0
    with open(path, "a") as f:
        for it in items:
            if it["id"] in done:
                continue
            f.write(json.dumps(episode_fn(it), ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
    return {"skipped_done": len(done), "written": n_new}
