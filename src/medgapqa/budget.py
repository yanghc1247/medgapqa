"""GPU-hour budget enforcement (user decision 2026-10-16, plan §9).

- EVAL: warning line 55 GPU-hours (recorded and continued past it, reported to the user), hard stop at 120 (stop safely and resume with --resume).
- DEV (prompt tuning, judge checks, noise floor): limit 15, warning recorded at 12 (80%).
The ledger is a jsonl shared by several processes (file lock), and whether the total crossed a line is decided by re-reading it after every record.
GPU-hour record = episode wall-clock seconds × gpu_weight (evaluation-model GPU 1 + this process's share of simulator GPUs).
"""
import fcntl
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIMITS = {"eval": {"warn": 55.0, "hard": 120.0}, "dev": {"warn": 12.0, "hard": 15.0}}
LEDGER_DIR = ROOT / "runs" / "budget"


class BudgetStop(Exception):
    """Hard stop line reached. The run stops after the current episode is recorded (resumable)."""


class Budget:
    def __init__(self, category, ledger=None, limits=None):
        self.category = category
        self.limits = limits or LIMITS[category]
        self.ledger = Path(ledger) if ledger else LEDGER_DIR / f"{category}_ledger.jsonl"
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.events = self.ledger.with_name(self.ledger.stem + "_events.jsonl")

    def total(self):
        if not self.ledger.exists():
            return 0.0
        return sum(json.loads(line)["gpu_hours"] for line in open(self.ledger) if line.strip())

    def _append(self, path, rec):
        with open(path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(rec) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

    def _event_logged(self, kind):
        return self.events.exists() and any(json.loads(line)["kind"] == kind for line in open(self.events) if line.strip())

    def check(self):
        """Call before the next job: BudgetStop if at or above the hard stop line."""
        t = self.total()
        if t >= self.limits["hard"]:
            if not self._event_logged("hard"):
                self._append(self.events, {"kind": "hard", "time": time.time(), "total_gpu_hours": t})
            raise BudgetStop(f"{self.category} GPU-hour {t:.2f} >= hard limit {self.limits['hard']}")
        return t

    def charge(self, seconds, gpu_weight, note=""):
        """Records usage and checks the lines. The warning line is recorded only once; the hard stop line raises BudgetStop."""
        self._append(self.ledger, {"time": time.time(), "gpu_hours": seconds * gpu_weight / 3600, "note": note})
        t = self.total()
        if t >= self.limits["warn"] and not self._event_logged("warn"):
            self._append(self.events, {"kind": "warn", "time": time.time(), "total_gpu_hours": t})
        if t >= self.limits["hard"]:
            if not self._event_logged("hard"):
                self._append(self.events, {"kind": "hard", "time": time.time(), "total_gpu_hours": t})
            raise BudgetStop(f"{self.category} GPU-hour {t:.2f} >= hard limit {self.limits['hard']}")
        return t
