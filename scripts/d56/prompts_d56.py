"""D56 Part B fixed texts (registered with their sha256 before any run)."""
import hashlib

# B1 oracle facts: appended to the S1K vignette; facts verbatim as the simulator returns them (each strip(), F1F2 joined by one space)
ORACLE_SEP = "\n\n"
ORACLE_PREFIX = "Additional information from the patient: "

# B2 C4n: the only change to the C4 system prompt (p3 guidance sentence for C4, tools "ASK or SEARCH")
C4_OLD = ("Most questions contain all the information needed to answer. Each ASK or SEARCH costs time and delays care, "
          "so ANSWER immediately unless a specific missing fact would change your answer.")
C4N_NEW = ("Decide whether the case contains enough information to answer. If something you need is missing, you may ask the patient or search; "
           "otherwise you may answer.")

# B3 blind asker: prompt to be used by the user with a new Claude Opus session (no gold, no removed facts, no labels, no model outputs)
BLIND_ASKER_PROMPT = (
    "You will see clinical multiple-choice questions in a CSV file (columns item_id, case_s1k, options). In each case, some patient "
    "information that would help choose the answer may be missing. For each item, write exactly three short questions you would ask "
    "this patient (or the patient's record) to obtain the missing information most useful for choosing among the options. Each "
    "question must ask for one finding, test result, history element or category about this patient; do not ask general medical "
    "knowledge questions and do not ask about information already stated in the case. Return a CSV file named "
    "blind_asker_questions.csv with columns item_id,q1,q2,q3 and one row per item, in any order.")


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


if __name__ == "__main__":
    for k in ("ORACLE_PREFIX", "C4_OLD", "C4N_NEW", "BLIND_ASKER_PROMPT"):
        print(k, sha(globals()[k]))
