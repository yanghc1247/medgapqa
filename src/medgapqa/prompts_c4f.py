"""D36 correction: C4f system-prompt variant (new file; prompts.py and agent.py unchanged). Only two sentences of agent.system_prompt("C4", cfg) change; the rest is byte-identical.
- Removed: the sentence urging an immediate answer, DROP (must occur exactly once, verbatim). The preceding sentence on the same line, "Most questions contain all the information needed to answer.", stays.
- Replaced: the per-tool limit sentence OLD_LIMITS → NEW_RULE. "You can take at most 5 actions in total." stays.
"""
import hashlib
from agent import system_prompt

DROP = " Each ASK or SEARCH costs time and delays care, so ANSWER immediately unless a specific missing fact would change your answer."
OLD_LIMITS = "You can use ASK at most 2 times. You can use SEARCH at most 2 times."
NEW_RULE = ("In your first 4 actions you must use a tool (ASK or SEARCH). You may use each tool any number of times (0–4 in total across the 4 actions). "
            "After your 4th tool action, your next and final action must be ANSWER or ABSTAIN.")


def c4f_system_prompt(cond, cfg):
    assert cond == "C4"
    p = system_prompt(cond, cfg)
    assert p.count(DROP) == 1 and p.count(OLD_LIMITS) == 1, "C4 prompt sentence differs from the expected text"
    return p.replace(DROP, "").replace(OLD_LIMITS, NEW_RULE)


def prompt_sha(cfg):
    return hashlib.sha256(c4f_system_prompt("C4", cfg).encode()).hexdigest()
