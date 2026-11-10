"""MedQA-USMLE 4-option loader and EVAL/DEV manifest builder (plan §3.1)."""
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "GBaker/MedQA-USMLE-4-options"
REVISION = "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18"
DEV_SEED = 20261016
DEV_N = 300


def _snapshot_file(name):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION,
                           cache_dir=str(ROOT / "hf_cache" / "hub"))


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def item_hash(q, options, answer_idx):
    blob = json.dumps({"q": q, "o": options, "a": answer_idx}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_split(split):
    """split: 'train' | 'test'. Returns a list of dicts (id, question, options, answer_idx, hash)."""
    path = _snapshot_file(f"phrases_no_exclude_{split}.jsonl")
    items = []
    for i, line in enumerate(open(path)):
        r = json.loads(line)
        items.append({
            "id": f"medqa_{split}_{i:05d}",
            "question": r["question"],
            "options": r["options"],
            "answer_idx": r["answer_idx"],
            "meta_info": r.get("meta_info"),
            "hash": item_hash(r["question"], r["options"], r["answer_idx"]),
        })
    return items


def ngrams(text, n=13):
    toks = norm(text).split()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def build_manifests():
    test = load_split("test")
    train = load_split("train")

    # exclude train items overlapping EVAL: normalized question match or a shared 13-gram
    test_norm = {norm(x["question"]) for x in test}
    test_grams = set().union(*(ngrams(x["question"]) for x in test))
    excluded = []
    pool = []
    for x in train:
        if norm(x["question"]) in test_norm:
            excluded.append((x["id"], "exact_question_match_with_test"))
        elif ngrams(x["question"]) & test_grams:
            excluded.append((x["id"], "shares_13gram_with_test"))
        else:
            pool.append(x)

    dev = random.Random(DEV_SEED).sample(pool, DEV_N)
    dev_ids = {x["id"] for x in dev}
    assert not dev_ids & {x["id"] for x in test}
    assert not {x["hash"] for x in dev} & {x["hash"] for x in test}

    out = ROOT / "data"
    for name, rows in [("eval_manifest.jsonl", test), ("dev_manifest.jsonl", dev)]:
        with open(out / name, "w") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(out / "dev_pool_exclusions.jsonl", "w") as f:
        for i, why in excluded:
            f.write(json.dumps({"id": i, "reason": why}) + "\n")
    return {"eval_n": len(test), "train_n": len(train), "train_excluded_overlap": len(excluded),
            "dev_pool_n": len(pool), "dev_n": len(dev), "dev_seed": DEV_SEED, "revision": REVISION}


def load_manifest(name):
    return [json.loads(l) for l in open(ROOT / "data" / f"{name}_manifest.jsonl")]



# pre-filter for items without a patient (fixed 2026-10-16, data/filter_policy.md). Excluded if there is no singular patient reference or "N-year-old" form at all.
# plurals (patients, infants) are deliberately left out: to filter population and study items.
PATIENT_RE = re.compile(
    r"\b\d+[- ](year|month|week|day|hour)s?[- ]old\b"
    r"|\b(man|woman|boy|girl|infant|newborn|neonate|patient|child|toddler|adolescent|gravida|primigravida)\b",
    re.I)


def has_patient(question):
    return PATIENT_RE.search(question) is not None


if __name__ == "__main__":
    print(json.dumps(build_manifests(), indent=2))


# pre-filter for items referring to an image (user instruction 2026-10-16, data/filter_policy.md). Decided from the original text only.
# words pointing to an image, photo or figure, or "X is shown" (excluding cases followed by a text table and uses such as "studies have shown").
IMAGE_WORD_RE = re.compile(r"\b(images?|pictures?|photographs?|photos?|photomicrographs?|micrographs?|exhibits?|"
                           r"figures?|diagrams?|graphs?|arrows?|arrowheads?)\b", re.I)
SHOWN_RE = re.compile(r"\bshown\b", re.I)
SHOWN_NOT_IMAGE_RE = re.compile(r"\b(has|have|had)\s+shown\b|\bshown\s+(by|that)\b", re.I)
TEXT_TABLE_AFTER_RE = re.compile(r"^[^\n.?]{0,40}[.:]?[ \t]*\n\s*[^\n?]+\n")


def image_reference(question):
    """The matching string if there is an image reference, otherwise None."""
    m = IMAGE_WORD_RE.search(question)
    if m:
        return m.group(0)
    for m in SHOWN_RE.finditer(question):
        start = max(question.rfind(".", 0, m.start()), question.rfind("\n", 0, m.start())) + 1
        ctx = question[start:m.end()]
        if SHOWN_NOT_IMAGE_RE.search(question[start:m.end() + 8]):
            continue
        if TEXT_TABLE_AFTER_RE.match(question[m.end():m.end() + 120]):
            continue  # followed by a text table (line break + values/items) → the content is in the item
        return ctx.strip()
    return None
