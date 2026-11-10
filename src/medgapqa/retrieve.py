"""SEARCH: StatPearls BM25 (plan §4.2). Settings as in build_index.py (bm25s lucene, k1 1.5, b 0.75, English stop words, no stemming).

- The query is the model's content as is.
- Documents with score 0 (no query token overlaps) are not returned.
- Each document is truncated to max_tokens with one fixed tokenizer (so every evaluation model receives the same observation string).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "corpus" / "statpearls" / "index"


class Retriever:
    def __init__(self, bm25, docs, top_k, max_tokens, encode, decode):
        """docs[i] = {"id", "title", "content"} (index order). encode/decode: fixed tokenizer for truncation."""
        self.bm25, self.docs, self.top_k, self.max_tokens = bm25, docs, top_k, max_tokens
        self.encode, self.decode = encode, decode

    @classmethod
    def load(cls, top_k, max_tokens, tokenizer, index_dir=INDEX):
        import bm25s
        bm25 = bm25s.BM25.load(str(Path(index_dir) / "bm25s"))
        docs = [json.loads(line) for line in open(Path(index_dir) / "corpus.jsonl")]
        return cls(bm25, docs, top_k, max_tokens,
                   lambda t: tokenizer.encode(t, add_special_tokens=False), tokenizer.decode)

    def truncate(self, text):
        ids = self.encode(text)
        if len(ids) <= self.max_tokens:
            return text, False
        return self.decode(ids[: self.max_tokens]).strip() + " ...", True

    def search(self, query):
        import bm25s
        toks = bm25s.tokenize([query], stopwords="en", show_progress=False, return_ids=False)
        if not toks[0]:
            return []
        res, scores = self.bm25.retrieve(toks, k=min(self.top_k, len(self.docs)), show_progress=False)
        out = []
        for i, s in zip(res[0].tolist(), scores[0].tolist()):
            if s <= 0:
                continue
            d = self.docs[i]
            text, cut = self.truncate(d["content"])
            out.append({"idx": i, "id": d["id"], "title": d["title"], "text": text, "truncated": cut,
                        "score": round(float(s), 4)})
        return out


def format_results(query, results):
    if not results:
        return f'Search results for "{query}":\nNo results found.'
    parts = [f"[{k}] {r['title']}\n{r['text']}" for k, r in enumerate(results, 1)]
    return f'Search results for "{query}":\n' + "\n\n".join(parts)
