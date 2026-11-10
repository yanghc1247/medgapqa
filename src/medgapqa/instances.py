"""Variable-instance rule (user decision 2026-10-16): results must not depend on the number of instances or on which instance handled an item.

- Item order is fixed to manifest order; splitting is shard(items, k, n) = items[k::n]; merging sorts by manifest order.
- Each episode depends only on the item and settings (greedy, batch 1, no seed). The simulator classification cache affects only speed.
- An instance is used only if its replies to fixed inputs hash to the reference (verify_instance). The instance id is logged.
"""
import hashlib
import json
import os
import socket


def instance_id(extra=""):
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    return f"{socket.gethostname()}:gpu{gpu}:pid{os.getpid()}{(':' + extra) if extra else ''}"


def shard(items, k, n):
    return items[k::n]


def merge(rows, order_ids):
    pos = {i: k for k, i in enumerate(order_ids)}
    return sorted(rows, key=lambda r: pos[r["id"]])


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def verify_instance(chat, probes, ref_hashes):
    """probes: [(msgs, prefill, max_new)], ref_hashes: sha256 in the same order. True if all match."""
    got = [sha(chat(m, p, n)[0]) for m, p, n in probes]
    return got == list(ref_hashes), got


class PooledBackend:
    """Uses several verified servers as one. The instance is chosen from the hash of the request content (deterministic) and the handling instance is recorded."""

    def __init__(self, backends, names):
        assert backends and len(backends) == len(names)
        self.backends, self.names = backends, names
        self.last_instance = None

    def chat(self, msgs, prefill=""):
        k = int(sha(json.dumps([msgs, prefill], sort_keys=True)), 16) % len(self.backends)
        self.last_instance = self.names[k]
        return self.backends[k].chat(msgs, prefill)
