"""D26 simulator v10 (2 facts, new file; frozen simulator.py only imported). Meerkat-70B NF4 resident server, greedy.

Input: visible item (VISIBLE), removed facts F1 and F2 verbatim, question. Output JSON {"qtype", "rationale" (before label, at most 25 words), "label"}.
label ∈ F1 | F2 | F1F2 | IN_CASE | NOT_AVAILABLE. Interpretation questions and open questions without a category are NOT_AVAILABLE.
Reply (D2 kept, no generated text): F1/F2 → that fact verbatim, stripped; F1F2 → the two facts joined by one space; IN_CASE and NOT_AVAILABLE → fixed sentences.
In S2K, F1, F2 and F1F2 all get the NOT_AVAILABLE sentence. The agent's observation is built by the caller as "Response: " + sentence.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/medgapqa"))
from simulator import IN_CASE_TEXT, NOT_AVAILABLE_TEXT  # noqa: E402

VERSION = "v10"
LABELS = ["F1", "F2", "F1F2", "IN_CASE", "NOT_AVAILABLE"]
SYSTEM = "You are a careful assistant that classifies a doctor's question about a patient record. Follow the output format exactly."
PROMPT_V10 = """A doctor is working on a patient case. The doctor can see the CASE TEXT. Two facts (F1 and F2) were removed from the case; the doctor cannot see them.

[CASE TEXT] (the doctor can see this)
{visible}

[F1] (removed; the doctor cannot see this)
{f1}

[F2] (removed; the doctor cannot see this)
{f2}

[DOCTOR'S QUESTION]
{question}

Decide what the question asks, following these steps:
1. qtype: "interpretation" if the question asks for an interpretation, diagnosis, explanation, or medical knowledge; otherwise "patient_fact".
2. For F1 and for F2 separately, decide: does this question ask for the information contained in this fact? A question that names a clinical category, finding, test, or history item (for example "What did the tests show?") counts as a specific question about the facts in that category. An open question with no category (for example "Is there anything else?" or "Tell me more") does not ask for any fact.
3. label:
   - "F1" if the question asks for the information in F1 but not F2;
   - "F2" if it asks for the information in F2 but not F1;
   - "F1F2" if it asks for the information in both;
   - otherwise "IN_CASE" if the CASE TEXT already answers the question;
   - otherwise "NOT_AVAILABLE".
   If qtype is "interpretation", or the question is an open question with no category, the label is "NOT_AVAILABLE".

Output exactly one JSON object and nothing else, with the rationale written before the label:
{{"qtype": "patient_fact" | "interpretation", "rationale": "<at most 25 words>", "label": "F1" | "F2" | "F1F2" | "IN_CASE" | "NOT_AVAILABLE"}}"""
PREFILL = '{"qtype": "'
MODES = ("s0", "s1k", "s2k")


def prompt_sha():
    return hashlib.sha256((SYSTEM + PROMPT_V10 + PREFILL).encode()).hexdigest()


def parse(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        obj = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        obj = None
    if not isinstance(obj, dict):
        return None, None
    qt, lab = obj.get("qtype"), obj.get("label")
    return (qt if qt in ("patient_fact", "interpretation") else None), (lab if lab in LABELS else None)


def combine(qtype, label):
    if qtype is None or label is None:
        return None
    return "NOT_AVAILABLE" if qtype == "interpretation" else label


class SimulatorK:
    def __init__(self, backend):
        self.backend, self.cache = backend, {}

    def classify(self, visible, facts, question):
        f1, f2 = (facts + [None, None])[:2] if facts else (None, None)
        key = hashlib.sha256(json.dumps([visible, f1, f2, question]).encode()).hexdigest()
        hit = key in self.cache
        if not hit:
            user = PROMPT_V10.format(visible=visible.strip(), f1=(f1 or "(none)").strip(), f2=(f2 or "(none)").strip(),
                                     question=question.strip())
            raw, i, o = self.backend.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], PREFILL)
            qt, lab_raw = parse(raw)
            self.cache[key] = {"label": combine(qt, lab_raw), "qtype": qt, "label_raw": lab_raw, "raw": raw, "tokens": {"in": i, "out": o},
                               "prompt_version": VERSION, "prompt_sha": hashlib.sha256(user.encode()).hexdigest()[:16],
                               "instance": getattr(self.backend, "last_instance", None)}
        return self.cache[key], hit

    def respond(self, mode, visible, facts, question):
        """facts: [F1, F2] verbatim (None/empty list at S0). Returns (sentence, record)."""
        assert mode in MODES
        c, hit = self.classify(visible, None if mode == "s0" else facts, question)
        lab = c["label"]
        if lab is None:
            kind, text = "sim_parse_fail", NOT_AVAILABLE_TEXT
        elif lab in ("F1", "F2", "F1F2") and mode == "s1k" and facts:
            parts = {"F1": [facts[0]], "F2": [facts[1]], "F1F2": [facts[0], facts[1]]}[lab]
            kind, text = "provided", " ".join(p.strip() for p in parts)
        elif lab in ("F1", "F2", "F1F2"):
            kind, text = "withheld_denied", NOT_AVAILABLE_TEXT
        elif lab == "IN_CASE":
            kind, text = "in_case", IN_CASE_TEXT
        else:
            kind, text = "not_available", NOT_AVAILABLE_TEXT
        return text, {**c, "kind": kind, "cached": hit}
