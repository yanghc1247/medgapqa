"""StatPearls snippets → leakage filter → bm25s index (plan §4.2).

Filter order:
1. Exclude "Review Questions" snippets that contain only a notice (a link to an external quiz, no question text)
2. Exclude snippets sharing any normalized 13-gram with an EVAL (MedQA test) question or option (number excluded is reported)
BM25 settings (fixed): bm25s defaults (method="lucene", k1=1.5, b=0.75), English stop words removed, no stemming, indexed field contents (title + text).
Usage: python -I src/medgapqa/build_index.py <chunks_jsonl> <out_dir>
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_manifest, ngrams  # noqa: E402

BOILERPLATE = re.compile(r"^\s*Access free multiple choice questions on this topic\.", re.I)
K1, B, METHOD = 1.5, 0.75, "lucene"


def main():
    import bm25s
    chunks, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    eval_grams = set()
    for x in load_manifest("eval"):
        eval_grams |= ngrams(x["question"])
        for o in x["options"].values():
            eval_grams |= ngrams(o)

    kept, n_total, n_boiler, leaked = [], 0, 0, []
    for line in open(chunks):
        s = json.loads(line)
        n_total += 1
        if s["title"].endswith("-- Review Questions") and BOILERPLATE.match(s["content"]):
            n_boiler += 1
            continue
        hit = ngrams(s["contents"]) & eval_grams
        if hit:
            leaked.append({"id": s["id"], "n_shared_13grams": len(hit), "example": sorted(hit)[0]})
            continue
        kept.append(s)

    with open(out / "corpus.jsonl", "w") as f:
        for s in kept:
            f.write(json.dumps({"id": s["id"], "title": s["title"], "content": s["content"]}, ensure_ascii=False) + "\n")
    with open(out / "leak_excluded.jsonl", "w") as f:
        for x in leaked:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    t = time.time()
    tokens = bm25s.tokenize([s["contents"] for s in kept], stopwords="en", show_progress=False)
    retriever = bm25s.BM25(method=METHOD, k1=K1, b=B)
    retriever.index(tokens, show_progress=False)
    retriever.save(str(out / "bm25s"))
    stats = {"snippets_total": n_total, "excluded_review_boilerplate": n_boiler, "excluded_eval_13gram": len(leaked),
             "indexed": len(kept), "bm25": {"impl": f"bm25s {bm25s.__version__}", "method": METHOD, "k1": K1, "b": B,
                                           "stopwords": "en", "stemmer": None},
             "index_seconds": round(time.time() - t, 1)}
    json.dump(stats, open(out / "index_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
