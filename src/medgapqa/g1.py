"""D74 G1 parse_fail rule (the same rule for every model). Applied to stored trajectories at the scoring step only (no model is re-run).
If the outcome is parse_fail, the invalid reason of the final attempt of the last turn (retry_raw, else raw) is bad_letter, and the first object of
parse_last_json has action ANSWER and content starting with RX, return that letter. Otherwise None (A, as before).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from format_check import parse_last_json  # noqa: E402

RX = re.compile(r"^\s*[(\[]?([A-D])[)\].:]?\s")


def g1_letter(tr):
    if tr.get("outcome") != "parse_fail" or not tr.get("turns"):
        return None
    last = tr["turns"][-1]
    if "retry_raw" in last:
        raw, why = last.get("retry_raw"), last.get("retry_invalid_reason")
    else:
        raw, why = last.get("raw"), last.get("invalid_reason")
    if why != "bad_letter" or not raw:
        return None
    found = parse_last_json(raw)
    if not found or found[0].get("action") != "ANSWER" or not isinstance(found[0].get("content"), str):
        return None
    m = RX.match(found[0]["content"])
    return m.group(1) if m else None
