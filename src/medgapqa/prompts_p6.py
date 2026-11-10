"""D37 P6 (targeted questioning) question-generation prompt (new file; pipeline_k.py unchanged). Appends one space and SUFFIX to the end of the P1b generation system prompt
(pipeline_k.GEN_SYSTEM, NQ = 3, NS = 0, SCHEMA Q3). Consolidated re-query, simulator, judge, FCR and number of questions are the same as P1b.
"""
import hashlib

VERSION = "v1"
SUFFIX = ("Before writing your questions, identify which pieces of patient-specific information (history, examination findings, laboratory or imaging "
          "results, medications, timing) are missing from the case and would decide between the two most plausible answer options. Then write each "
          "question so that it asks directly for one such missing fact about this patient. Do not ask general medical knowledge questions. Do not ask "
          "about information already stated in the case. Do not repeat a question.")


def gen_system(base):
    return base + " " + SUFFIX


def sha(base):
    from pipeline_k import SCHEMAS
    nq, ns, schema = SCHEMAS["Q3"]
    return hashlib.sha256(gen_system(base).format(NQ=nq, NS=ns, SCHEMA=schema).encode()).hexdigest()
