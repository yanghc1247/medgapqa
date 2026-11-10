"""Agent prompt sentences (same for every model; only the per-model chat template differs). Change history in logs/prompt_tuning.md.

v4 (2026-10-16): in the format_check v3 sentences, the ANSWER instruction became "a single letter (A–D) only", examples became neutral placeholders,
and the invalid reason was added to the re-request sentence.
"""
PROMPT_VERSION = "v4"
PREFILL = '{"action": "'
HEADER = "You are a clinical decision agent working on a multiple-choice medical question. At each turn you choose exactly one action."
DEFS = {
    "ANSWER": 'ANSWER: commit to one option. "content" is a single letter (A, B, C, or D) and nothing else. Ends the case.',
    "ASK": 'ASK: ask one question about this patient. "content" is the question.',
    "SEARCH": 'SEARCH: search a general medical knowledge corpus. "content" is the query.',
    "ABSTAIN": 'ABSTAIN: decline and hand the case to a human. "content" is a brief reason. Ends the case.',
}
EXAMPLES = {
    "ANSWER": '{"action": "ANSWER", "content": "<A, B, C, or D>"}',
    "ASK": '{"action": "ASK", "content": "<one question to ask>"}',
    "SEARCH": '{"action": "SEARCH", "content": "<one search query>"}',
    "ABSTAIN": '{"action": "ABSTAIN", "content": "<one brief reason>"}',
}
COST = "Gathering information costs time, so use ASK or SEARCH only when needed."
FORMAT = 'Output exactly one JSON object and nothing else. Keep "content" to at most 30 words.'

RETRY_BASE = "Your previous output was not a valid action: {why}. Output exactly one JSON object with the keys \"action\" and \"content\"."
RETRY_WHY = {
    "truncated": "it was too long",
    "no_json": "it was not a JSON object",
    "bad_schema": 'it must have exactly the keys "action" and "content"',
    "disallowed_action": "the action must be one of the actions listed above",
    "bad_letter": "content must be one letter A–D",
    "empty_content": "content must not be empty",
}


def retry_message(reason):
    return RETRY_BASE.format(why=RETRY_WHY[reason])


# DEV prompt-tuning candidates (2026-10-16, user instruction: problem of S0 first action ANSWER 0/10, at most 3 candidates).
# goes in place of the COST sentence in conditions allowing tools (ASK/SEARCH). v4 keeps the existing COST sentence.
GUIDANCE = {
    "v4": [COST],
    "p1": [COST, "If the question already gives enough information to choose an answer, ANSWER right away."],
    "p2": ["Before acting, decide whether the question as written is enough to choose one option. If it is, ANSWER now.",
           "Use ASK only when a specific fact about this patient that you need is missing.",
           "Use SEARCH only when you need general medical knowledge that you do not have.",
           "Use ABSTAIN only when you cannot choose an answer."],
    "p3": ["Most questions contain all the information needed to answer. Each {tools} costs time and delays care, "
           "so ANSWER immediately unless a specific missing fact would change your answer."],
}
# {tools} of p3 is filled with the allowed tools: C2 "ASK", C3 "SEARCH", C4 "ASK or SEARCH" (user instruction 2026-10-16, deviations.md)
GUIDANCE_ACTION = {1: "ASK", 2: "SEARCH", 3: "ABSTAIN"}  # per-action sentences of p2 → included only for allowed actions
