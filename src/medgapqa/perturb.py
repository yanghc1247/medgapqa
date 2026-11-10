"""PERT (cue removal) generation and automatic checks (plan §3.2).

Step 1: the generator LLM picks "the single patient-specific fact without which the answer would be hardest to determine" and its type.
Step 2: the same model rewrites the item text without only that fact. Options and answer stay the same.
If the automatic checks fail, regenerate up to MAX_RETRIES times with the failure reason included, then exclude (reason recorded).
"""
import json
import re

REMOVED_TYPES = ["history", "physical_exam", "lab_imaging", "demographic"]
TYPE_KO = {"history": "history", "physical_exam": "physical exam", "lab_imaging": "labs/imaging", "demographic": "demographics"}
MAX_RETRIES = 2
NO_PATIENT_FACT = "no patient-specific fact (model returned null)"

SYSTEM = "You are a careful medical exam editor. Follow the output format exactly."

SELECT_PROMPT = """Below is a USMLE-style question with its correct answer.

Identify the ONE patient-specific fact in the vignette (e.g., a history item, a physical exam finding, a lab/imaging result, or a demographic detail of THIS patient) whose absence would make the correct answer MOST difficult to determine. Do not choose general medical knowledge, and do not choose the final question sentence.

Choose a DECISIVE fact, not merely a relevant one: without it, the remaining information should no longer single out the correct answer, and at least one other option should become a reasonable choice. Avoid facts whose information is repeated or strongly implied by other findings that would remain in the vignette.

Copy the fact exactly as it appears in the vignette (a contiguous span, as short as possible while complete).

Question:
{question}

Options:
{options}

Correct answer: {answer_idx}

If the question is not about a specific patient (e.g., a population, epidemiology, or pure knowledge question), set removed_fact to null.

Respond with only a JSON object:
{{"removed_fact": "<exact span copied from the vignette, or null>", "removed_type": "<one of: history, physical_exam, lab_imaging, demographic, or null>"}}"""

REWRITE_PROMPT = """Rewrite the vignette below so that the following fact is completely removed. Remove the fact and anything that directly restates or implies it, but change nothing else: keep every other sentence, number and the final question sentence word-for-word. Do not add any new information, hints, or placeholders such as "unknown" or "not reported".

Fact to remove: {removed_fact}

Vignette:
{question}

Respond with only the rewritten vignette text."""

RETRY_SUFFIX = "\n\nYour previous attempt was rejected for this reason: {reason}. Fix it."

STOP = set("""a an the of and or in on at to for with by from is was were are be been has had have this that these
those it its as his her he she they their patient patients year years old man woman boy girl which following most
likely what who after before during into over under than then there no not""".split())


def content_tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in STOP]


def norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()]


def split_stem(question):
    """Splits off the last sentence (the question sentence)."""
    sents = sentences(question)
    return " ".join(sents[:-1]), sents[-1]


# common generic words in question sentences: excluded from the dangling-reference check
GENERIC_Q = set("""following best next step appropriate management diagnosis finding findings expected explain explains
cause caused condition presentation pathophysiology mechanism properties likely seen would will be should most this
these patient patients her his him he she their which what why how does do is are""".split())

# for verb detection in edited sentences (list of finite verbs + -ed forms)
VERB_RE = re.compile(
    r"(is|are|was|were|be|been|being|has|have|had|does|do|did|can|could|may|might|must|shall|should|will|would|"
    r"shows?|showed|shown|reveals?|revealed|demonstrates?|demonstrated|presents?|presented|reports?|reported|"
    r"denies|denied|complains?|complained|appears?|appeared|notes?|noted|takes?|took|includes?|included|"
    r"undergoes|underwent|remains?|becomes?|became|describes?|described|says?|said|learns?|learned|finds?|found|"
    r"gets?|got|goes|went|comes?|came|feels?|felt|states?|stated|lives?|lived|works?|worked|uses?|used|drinks?|"
    r"drank|smokes?|smoked|began|begins?|wakes?|woke|blames?|refuses?|refused|returns?|returned|snores?|"
    r"experiences?|experienced|needs?|needed|wants?|wanted|asks?|asked|tells?|told|gives?|gave|starts?|started|"
    r"\w+ed)")


# extended dangling check (user instruction 2026-10-16)
# (1) sentence ending without an object ("Physical examination shows.")
EMPTY_OBJECT_RE = re.compile(r"\b(shows?|showed|reveals?|revealed|demonstrates?|demonstrated|includes?|included|"
                             r"notable for|significant for|remarkable for|consistent with|positive for|negative for)\s*[.:]?$",
                             re.I)
# (2) fragment left with only a relative clause ("A 27-year old male who works on an organic farm.")
REL_PRON_RE = re.compile(r"\b(who|that|which)\b", re.I)
# (3) generic noun that lost its antecedent ("these parasites", "the mass")
GENERIC_NOUNS = set("""lesion mass tumor nodule lump growth cyst polyp parasite organism bacterium bacteria pathogen
infection rash wound injury fracture ulcer murmur effusion abnormality specimen drug medication agent substance
procedure surgery operation device implant""".split())
DET_PHRASE_RE = re.compile(r"\b(the|this|these|those)\s+((?:[a-z][a-z-]*\s+){0,2}[a-z][a-z-]*)\b")
# 'the' only for the generic-noun list; demonstratives (this/these/those) also check the words of the removed fact (avoids false hits such as 'the blood count' using the same word in another sense)


def _stem(w):
    return w[:-2] if w.endswith("es") and len(w) > 4 else (w[:-1] if w.endswith("s") and len(w) > 3 else w)


def _words(text):
    return [_stem(w) for w in re.findall(r"[a-z][a-z-]*", text.lower())]


def is_fragment(sentence):
    """Sentence left with only a relative clause: starts with 'A/An ...', no verb before the relative pronoun and at most one verb after it."""
    st = sentence.strip()
    m = REL_PRON_RE.search(st)
    if not re.match(r"(A|An)\s", st) or not m:
        return False
    pre = re.findall(r"[a-z]+", st[:m.start()].lower())
    post = re.findall(r"[a-z]+", st[m.end():].lower())
    return not any(VERB_RE.fullmatch(w) for w in pre) and sum(bool(VERB_RE.fullmatch(w)) for w in post) <= 1


def lost_antecedents(pert, question, fact):
    """'the/this/these/those + noun' whose antecedent existed in the original but disappeared in PERT."""
    fact_stems = {w for w in _words(fact) if w not in STOP and w not in GENERIC_Q and len(w) > 3}
    generic = {_stem(w) for w in GENERIC_NOUNS}
    pl, ql = pert.lower(), question.lower()
    out = []
    for m in DET_PHRASE_RE.finditer(pl):
        cand = generic if m.group(1) == "the" else generic | fact_stems
        heads = [w for w in _words(m.group(2)) if w in cand]
        if not heads:
            continue
        head = heads[-1]
        if head in _words(pl[:m.start()]):
            continue  # the antecedent remains in PERT
        pos = ql.find(m.group(0))
        if pos > 0 and head in _words(ql[:pos]):
            out.append(m.group(0))  # the original had an antecedent → lost by the removal
    return out


def fmt_options(options):
    return "\n".join(f"{k}. {v}" for k, v in options.items())


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def check_selection(sel, question):
    if sel and "removed_fact" in sel and sel["removed_fact"] is None:
        return NO_PATIENT_FACT
    if not sel or not isinstance(sel.get("removed_fact"), str) or not sel["removed_fact"].strip():
        return "selection output is not valid JSON with a removed_fact"
    if sel.get("removed_type") not in REMOVED_TYPES:
        return f"removed_type must be one of {REMOVED_TYPES}"
    fact = sel["removed_fact"]
    _, last = split_stem(question)
    if norm(fact) not in norm(question):
        ct = content_tokens(fact)
        orig = set(content_tokens(question))
        if not ct or sum(t in orig for t in ct) / len(ct) < 0.8:
            return "removed_fact is not a span copied from the vignette"
    if norm(fact) in norm(last) or norm(last) in norm(fact):
        return "removed_fact must not be the final question sentence"
    return None


# if the sentence right after the removed fact starts with a demonstrative pronoun, flag for human review (not an automatic exclusion, logs/decisions.md 2026-10-16)
REVIEW_PRONOUNS = {"it", "this", "these", "they", "those", "that"}
PERSON_NOUNS = {"patient", "patients", "man", "woman", "boy", "girl", "child", "infant", "newborn", "neonate",
                "individual", "person", "mother", "father", "baby", "toddler", "adolescent", "student"}


def pronoun_after_fact(pert, question, fact):
    """If the sentence right after the removed fact in the original starts with a pronoun and remains as is in PERT, returns that sentence."""
    m = re.search(r"\s+".join(map(re.escape, fact.lower().split())), question.lower())
    if not m:
        return None
    pos = 0
    for s in sentences(question):
        start = question.find(s, pos)
        pos = start + len(s)
        if start < m.end():
            continue
        words = re.findall(r"[a-z]+", s.lower())
        if words and words[0] in REVIEW_PRONOUNS and not (len(words) > 1 and words[1] in PERSON_NOUNS) \
                and norm(s) in norm(pert):
            return s
        return None
    return None


def check_rewrite(pert, question, fact):
    """Automatic checks. Failure reason (string) or None. Results are also returned as a dict."""
    checks = {}
    pert = pert.strip()
    _, last = split_stem(question)
    checks["nonempty_and_shorter"] = 0 < len(pert) < len(question)
    # the stray trailing quote (?") of the MedQA original is ignored in the comparison
    checks["question_sentence_kept"] = norm(last).strip('"\'') in norm(pert)
    checks["fact_span_absent"] = norm(fact) not in norm(pert)
    # whether content tokens found only in the fact (not in the rest of the original) remain in PERT
    rest = norm(question).replace(norm(fact), " ")
    rest_tokens = set(content_tokens(rest))
    unique = [t for t in set(content_tokens(fact)) if t not in rest_tokens]
    left = [t for t in unique if t in set(content_tokens(pert))]
    checks["fact_unique_tokens_left"] = left
    checks["fact_content_removed"] = len(left) <= max(0, len(unique) // 5)
    # whether new content tokens or numbers not in the original were introduced
    orig_tokens = set(content_tokens(question))
    new = sorted({t for t in content_tokens(pert) if t not in orig_tokens})
    checks["new_tokens"] = new
    checks["no_new_numbers"] = not any(re.fullmatch(r"[0-9.]+", t) for t in new)
    checks["no_new_info"] = len(new) <= 2 and checks["no_new_numbers"]
    # whether the final question sentence refers to a target that existed only in the removed fact (dangling reference)
    pert_body, _ = split_stem(pert)
    body_tokens = set(content_tokens(pert_body))
    fact_tokens = set(content_tokens(fact))
    dangling = sorted(t for t in set(content_tokens(last)) - GENERIC_Q
                      if t in fact_tokens and t not in body_tokens)
    checks["question_dangling_terms"] = dangling
    checks["no_dangling_reference"] = not dangling
    # whether a verb remains in the edited sentences (sentences not found verbatim in the original)
    orig_sents = {norm(s) for s in sentences(question)}
    modified = [s for s in sentences(pert) if norm(s) not in orig_sents]
    verbless = [s for s in modified if not any(VERB_RE.fullmatch(w) for w in re.findall(r"[a-z]+", s.lower()))]
    empty_obj = [x for x in modified if EMPTY_OBJECT_RE.search(x.strip().rstrip("."))
                 or EMPTY_OBJECT_RE.search(x.strip())]
    fragments = [x for x in modified if is_fragment(x)]
    incomplete = verbless + [x for x in empty_obj + fragments if x not in verbless]
    checks["verbless_modified_sentences"] = verbless
    checks["empty_object_sentences"] = empty_obj
    checks["fragment_sentences"] = fragments
    checks["no_incomplete_sentence"] = not incomplete
    lost = lost_antecedents(pert, question, fact)
    checks["lost_antecedents"] = lost
    checks["no_lost_antecedent"] = not lost
    reasons = {
        "nonempty_and_shorter": "the rewrite must be non-empty and shorter than the original",
        "question_sentence_kept": "the final question sentence must be kept word-for-word",
        "fact_span_absent": "the removed fact still appears verbatim",
        "fact_content_removed": f"content of the removed fact remains ({', '.join(left)})",
        "no_new_info": f"new information was added ({', '.join(new)})",
        "no_dangling_reference": f"the question sentence refers to the removed fact ({', '.join(dangling)})",
        "no_incomplete_sentence": f"the rewrite left an incomplete sentence ({' | '.join(incomplete)})",
        "no_lost_antecedent": f"a noun phrase lost its antecedent ({', '.join(lost)})",
    }
    # human-review flag that does not exclude automatically
    review = []
    pron = pronoun_after_fact(pert, question, fact)
    if pron:
        review.append({"flag": "pronoun_after_removed_fact", "sentence": pron})
    checks["needs_review"] = review
    for k, why in reasons.items():
        if not checks[k]:
            return why, checks
    return None, checks


class Generator:
    def __init__(self, model_path, device="cuda:0", max_new_tokens=768, prequantized=False):
        """prequantized=True: loads the NF4 weights saved with save_pretrained as is (the quantization config is in the saved config)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        kw = {} if prequantized else {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)}
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": device},
                                                          dtype=torch.bfloat16, **kw)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.eos = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|eot_id|>")]

    def chat(self, msgs, prefill=""):
        """Fixed greedy. If there is a prefill, it is placed before the assistant reply so generation continues from it, and the returned text includes the prefill."""
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + prefill
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(ids, attention_mask=self.torch.ones_like(ids), do_sample=False,
                                      max_new_tokens=self.max_new_tokens, eos_token_id=self.eos,
                                      pad_token_id=self.eos[0])
        gen = out[0, ids.shape[1]:]
        text = (prefill + self.tok.decode(gen, skip_special_tokens=True)).strip()
        return text, int(ids.shape[1]), int(gen.shape[0])

    def __call__(self, user):
        return self.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])


NO_PATIENT_RULE = "no patient in vignette (text rule, data/filter_policy.md)"
IMAGE_RULE = "image reference in vignette (text rule, data/filter_policy.md)"


def perturb_item(gen, item):
    """Processes one item and returns the full record of attempts. Items without a patient are excluded without a model call."""
    from data import has_patient, image_reference
    q = item["question"]
    rule = NO_PATIENT_RULE if not has_patient(q) else (IMAGE_RULE if image_reference(q) else None)
    if rule:
        return {"id": item["id"], "hash": item["hash"], "status": "excluded", "exclude_reason": rule,
                "removed_fact": None, "removed_type": None, "pert_question": None, "n_attempts": 0, "attempts": []}
    attempts = []
    reason = None
    for attempt in range(MAX_RETRIES + 1):
        suffix = RETRY_SUFFIX.format(reason=reason) if reason else ""
        sel_raw, in1, out1 = gen(SELECT_PROMPT.format(question=q, options=fmt_options(item["options"]),
                                                      answer_idx=item["answer_idx"]) + suffix)
        sel = parse_json(sel_raw)
        rec = {"attempt": attempt, "select_raw": sel_raw, "tokens": {"in": in1, "out": out1}}
        reason = check_selection(sel, q)
        if reason is None:
            pert, in2, out2 = gen(REWRITE_PROMPT.format(removed_fact=sel["removed_fact"], question=q) + suffix)
            rec.update(removed_fact=sel["removed_fact"], removed_type=sel["removed_type"], pert=pert)
            rec["tokens"] = {"in": in1 + in2, "out": out1 + out2}
            reason, checks = check_rewrite(pert, q, sel["removed_fact"])
            rec["checks"] = checks
        rec["fail_reason"] = reason
        attempts.append(rec)
        if reason is None or reason == NO_PATIENT_FACT:
            break
    final = attempts[-1]
    return {
        "id": item["id"], "hash": item["hash"],
        "status": "pass" if final["fail_reason"] is None else "excluded",
        "exclude_reason": final["fail_reason"],
        "removed_fact": final.get("removed_fact"), "removed_type": final.get("removed_type"),
        "pert_question": final.get("pert") if final["fail_reason"] is None else None,
        "needs_review": final.get("checks", {}).get("needs_review", []) if final["fail_reason"] is None else [],
        "n_attempts": len(attempts), "attempts": attempts,
    }
