"""ASK simulator (plan §4.1).

Design: the simulator LLM sees the hidden original item (= the item the agent sees + the removed fact) and the question,
and only classifies the question into three types. The sentence returned to the agent is fixed by the class.
  WITHHELD      : the question specifically asks for the removed fact → recoverable (S1): the removed fact verbatim; unrecoverable (S2): the NOT_AVAILABLE sentence
  IN_CASE       : asks for information already in the item → the IN_CASE sentence (rule ③)
  NOT_AVAILABLE : everything else (information not in the original, broad questions, requests for the answer, diagnosis or advice) → the NOT_AVAILABLE sentence (rules ②④)
This way
  - no free-form text from the simulator LLM reaches the agent, so there is no path for leaking the answer or diagnosis (the classification prompt contains neither options nor the answer).
  - classification uses the same mode-independent input and is cached, so S1 and S2 differ only in replies to WITHHELD questions.
  - the S2 reply to a WITHHELD question is character-for-character the reply to information that never existed, so the policy is not revealed.
"""
import hashlib
import json
import re

LABELS = ["WITHHELD", "IN_CASE", "NOT_AVAILABLE"]
MODES = ["s0", "recoverable", "unrecoverable"]
NOT_AVAILABLE_TEXT = "That information is not available."
IN_CASE_TEXT = "That information is already given in the case."

SIM_PROMPT_VERSION = "v6"
# v1 → v2: separated inputs, narrowed to "does it ask for the content of the removed fact", boundary examples
# v2 → v3: rule on diagnosis and cause questions moved to the front → more diagnosis questions went to IN_CASE, discarded
# v3 → v4: three judgments (true/false) as JSON → nearly all false (WITHHELD 0/90), discarded
# v4 → v5: quote the answering sentence → bias toward "none" and quote mismatches, discarded
# v5 → v6: two calls. (1) judge only the question type (patient_fact/interpretation) (2) v2 label prompt.
#          NOT_AVAILABLE if interpretation, otherwise the v2 label.
# (all 2026-10-16, tuned on 90 DEV check questions. runs/sim_tuning/)
SYSTEM = "You classify a doctor's question about a patient case. Follow the rules exactly."
INPUT = """[CASE TEXT] (the doctor can see this)
{visible}

[REMOVED FACT] (removed from the case text; the doctor cannot see this)
{withheld}

[DOCTOR'S QUESTION]
{question}
"""
TYPE_PROMPT = """[DOCTOR'S QUESTION]
{question}

Is this question asking for a fact about the patient, or for an interpretation?
- "patient_fact": history, symptoms, medications, exposures, exam findings, or test results of this patient.
- "interpretation": the diagnosis or cause of the presentation, a mechanism or defect, a risk factor, management or treatment choice, which option is correct, or general medical knowledge.

Examples:
- "Does she smoke?" -> patient_fact
- "What did the chest x-ray show?" -> patient_fact
- "Has she been treated for this before?" -> patient_fact
- "What is the most likely diagnosis?" -> interpretation
- "What is the underlying mechanism?" -> interpretation
- "Which drug should be given?" -> interpretation

Output exactly one JSON object and nothing else: {{"type": "patient_fact" | "interpretation"}}"""
TYPE_PREFILL = '{"type": "'
PROMPT = INPUT + """
Decide:
1. WITHHELD: the question asks for the content of the REMOVED FACT, that is, the answer to the question would be the removed fact itself (fully or partly). A question about a different finding, test, or history item is not WITHHELD, even if it is related to the same organ or disease.
2. IN_CASE: the question asks for information that is explicitly written in the CASE TEXT.
3. NOT_AVAILABLE: everything else. This includes history, exam findings, or test results that are written in neither the CASE TEXT nor the REMOVED FACT, and broad requests such as "any other findings?".

Examples (for a different case):
CASE TEXT: A 30-year-old woman has had a cough for 2 weeks. Her temperature is 38.5°C (101.3°F).
REMOVED FACT: A chest x-ray shows a right lower lobe infiltrate.
- "What did the chest x-ray show?" -> WITHHELD
- "Were any imaging studies done, and what did they show?" -> WITHHELD
- "What is her temperature?" -> IN_CASE
- "How long has she had the cough?" -> IN_CASE
- "Does she smoke?" -> NOT_AVAILABLE
- "What is her oxygen saturation?" -> NOT_AVAILABLE
- "What did the lung examination show?" -> NOT_AVAILABLE
- "Does she have a history of lung disease?" -> NOT_AVAILABLE
- "Are there any other findings?" -> NOT_AVAILABLE

Output exactly one JSON object and nothing else: {{"label": "WITHHELD" | "IN_CASE" | "NOT_AVAILABLE"}}"""
PREFILL = '{"label": "'


def parse_field(text, key, allowed):
    m = re.search(r'"%s"\s*:\s*"([A-Za-z_]+)"' % key, text)
    return m.group(1) if m and m.group(1) in allowed else None


def combine(qtype, label):
    """NOT_AVAILABLE if interpretation, otherwise the label. None if either fails to parse."""
    if qtype is None or label is None:
        return None
    return "NOT_AVAILABLE" if qtype == "interpretation" else label


# v7 (2026-10-16): WITHHELD definition confirmed by the user (asks specifically for the category, and an honest, complete answer would include the removed fact → WITHHELD).
#     examples are only invented cases that do not overlap the 50 human-checked items (to avoid overfitting).
PROMPT_V7 = INPUT + """
Decide:
1. WITHHELD: the question specifically asks about a category that the REMOVED FACT belongs to (for example, current medications, symptoms or findings of a specific organ or body system, or a specific test), and an honest and complete answer from the full record (CASE TEXT plus REMOVED FACT) would include the REMOVED FACT. If the question has several parts, it is WITHHELD when any part meets this rule. Check this rule first: a question can be WITHHELD even when part of the answer is already in the CASE TEXT.
2. IN_CASE: otherwise, if the information the question asks for is explicitly written in the CASE TEXT.
3. NOT_AVAILABLE: everything else. This includes details written in neither text (even if the CASE TEXT mentions the topic), questions that do not name a category (such as "the full history" or "any other findings?"), and questions about the diagnosis, the correct option, or advice.

Example 1 (a different case):
CASE TEXT: A 30-year-old woman has had a cough for 2 weeks. Her temperature is 38.5°C (101.3°F).
REMOVED FACT: A chest x-ray shows a right lower lobe infiltrate.
- "What did the chest x-ray show?" -> WITHHELD
- "Were any imaging studies done?" -> WITHHELD
- "Are there any abnormalities in her lungs?" -> WITHHELD
- "Does she have any lung problems besides the cough?" -> WITHHELD
- "What is her temperature, and what did her chest imaging show?" -> WITHHELD
- "What is her temperature?" -> IN_CASE
- "How long has she had the cough?" -> IN_CASE
- "How often does she cough each day?" -> NOT_AVAILABLE
- "What did the lung examination show?" -> NOT_AVAILABLE
- "Does she smoke?" -> NOT_AVAILABLE
- "Tell me her full medical history." -> NOT_AVAILABLE
- "Are there any other findings?" -> NOT_AVAILABLE
- "Which treatment should she receive?" -> NOT_AVAILABLE

Example 2 (a different case):
CASE TEXT: A 70-year-old man has a nosebleed that will not stop. He has atrial fibrillation.
REMOVED FACT: He takes warfarin.
- "Is he on any anticoagulants?" -> WITHHELD
- "What medications does he take?" -> WITHHELD
- "Does he have any heart rhythm problems?" -> IN_CASE
- "What is his INR?" -> NOT_AVAILABLE

Output exactly one JSON object and nothing else: {{"label": "WITHHELD" | "IN_CASE" | "NOT_AVAILABLE"}}"""

# v7 second pass (same day): added the sentence "rule 1 first; WITHHELD is possible even if part of the answer is in the item" and one example.
# v7 third pass (same day): the overfitting check found an example nearly identical to a question in the 50 human-checked cases ("What is the most likely diagnosis?"), which was
#     replaced with "Which treatment should she receive?".
# v8 (trial): v7 + a separate IN_CASE judgment. For patient-fact questions that are not WITHHELD, the item is asked to quote its evidence, and
#     IN_CASE if the quote is actually in the item; NOT_AVAILABLE if it is "none" or absent. Separate from the WITHHELD decision (avoids v5's skew to "none").
QUOTE_PROMPT = """[CASE TEXT]
{visible}

[DOCTOR'S QUESTION]
{question}

Copy word-for-word the sentence or phrase from the CASE TEXT that states the specific information the question asks for. If the CASE TEXT does not state it (for example, it mentions the topic but not the requested detail), write "none".

Output exactly one JSON object and nothing else: {{"quote": "<exact phrase from the CASE TEXT, or none>"}}"""
QUOTE_PREFILL = '{"quote": "'


def _toks(t):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", t.lower())


def quote_in_case(raw, visible):
    """Is the quote in the item: True if it is not 'none' and 80% or more of the quoted tokens are in the item. None on parse failure."""
    m = re.search(r'"quote"\s*:\s*"(.*?)"\s*\}', raw, re.S)
    if not m:
        return None
    q = _toks(m.group(1))
    if not q or q == ["none"]:
        return False
    vt = set(_toks(visible))
    return sum(t in vt for t in q) / len(q) >= 0.8


# v9 (the one change allowed after the probe, checked on the first 10 items and validated on the last 10): definitions and examples as in v7; only the names change.
#     in the probe, direct questions such as "What is the patient's BMI?" were also classified NOT_AVAILABLE → "REMOVED FACT(the doctor cannot see this)" and
#     reading the label "WITHHELD" as "information that cannot be given", the input becomes CHART NOTE and the label CHART_NOTE (mapped to WITHHELD in code).
_V9_INPUT = """[CASE TEXT] (the doctor can see this)
{visible}

[CHART NOTE] (part of this patient's record that is not shown in the case text; the doctor has not read it yet)
{withheld}

[DOCTOR'S QUESTION]
{question}
"""
PROMPT_V9 = _V9_INPUT + PROMPT_V7[len(INPUT):].replace("REMOVED FACT", "CHART NOTE").replace("WITHHELD", "CHART_NOTE")
V9_LABEL_MAP = {"CHART_NOTE": "WITHHELD", "IN_CASE": "IN_CASE", "NOT_AVAILABLE": "NOT_AVAILABLE"}

VERSIONS = ("v6", "v7", "v8", "v9")


class Simulator:
    """backend.chat(messages, prefill) -> (text, n_in, n_out). The same (visible item, removed fact, question) is classified only once."""

    def __init__(self, backend, version="v6"):
        assert version in VERSIONS
        self.backend = backend
        self.version = version
        self.cache = {}

    def prompt(self, visible, withheld, question):
        tpl = {"v6": PROMPT, "v9": PROMPT_V9}.get(self.version, PROMPT_V7)
        return tpl.format(visible=visible.strip(), withheld=(withheld or "(none)").strip(), question=question.strip())

    def classify(self, visible, withheld, question):
        key = hashlib.sha256(json.dumps([visible, withheld, question]).encode()).hexdigest()
        hit = key in self.cache
        if not hit:
            m1 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TYPE_PROMPT.format(question=question.strip())}]
            m2 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": self.prompt(visible, withheld, question)}]
            raw1, i1, o1 = self.backend.chat(m1, TYPE_PREFILL)
            raw2, i2, o2 = self.backend.chat(m2, PREFILL)
            qtype = parse_field(raw1, "type", ("patient_fact", "interpretation"))
            if self.version == "v9":
                label_raw = V9_LABEL_MAP.get(parse_field(raw2, "label", tuple(V9_LABEL_MAP)))
            else:
                label_raw = parse_field(raw2, "label", LABELS)
            label = combine(qtype, label_raw)
            raws, tin, tout, quote = {"type": raw1, "label": raw2}, i1 + i2, o1 + o2, None
            if self.version == "v8" and label in ("IN_CASE", "NOT_AVAILABLE") and qtype == "patient_fact":
                m3 = [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": QUOTE_PROMPT.format(visible=visible.strip(), question=question.strip())}]
                raw3, i3, o3 = self.backend.chat(m3, QUOTE_PREFILL)
                raws["quote"], tin, tout = raw3, tin + i3, tout + o3
                quote = quote_in_case(raw3, visible)
                label = None if quote is None else ("IN_CASE" if quote else "NOT_AVAILABLE")
            self.cache[key] = {"label": label, "qtype": qtype, "label_raw": label_raw, "quote_in_case": quote,
                               "raw": raws, "tokens": {"in": tin, "out": tout},
                               "prompt_version": self.version,
                               "instance": getattr(self.backend, "last_instance", None),
                               "prompt_sha": hashlib.sha256((m1[1]["content"] + m2[1]["content"]).encode()).hexdigest()[:16]}
        return self.cache[key], hit

    def respond(self, mode, visible, withheld, question):
        """Reply for the agent and the log record. mode: s0 / recoverable / unrecoverable."""
        assert mode in MODES
        if mode == "s0":
            withheld = None
        c, hit = self.classify(visible, withheld, question)
        label = c["label"]
        if label is None:
            kind, text = "sim_parse_fail", NOT_AVAILABLE_TEXT
        elif label == "WITHHELD" and withheld and mode == "recoverable":
            kind, text = "provided", withheld.strip()
        elif label == "WITHHELD":
            kind, text = "withheld_denied", NOT_AVAILABLE_TEXT
        elif label == "IN_CASE":
            kind, text = "in_case", IN_CASE_TEXT
        else:
            kind, text = "not_available", NOT_AVAILABLE_TEXT
        return text, {"label": label, "kind": kind, "qtype": c["qtype"], "label_raw": c["label_raw"], "raw": c["raw"], "tokens": c["tokens"],
                      "prompt_sha": c["prompt_sha"], "prompt_version": c["prompt_version"], "cached": hit,
                      "instance": c.get("instance")}
