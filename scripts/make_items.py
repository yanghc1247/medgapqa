"""Write the item file read by scripts/p2/run_eval.py (runs/v5/evalk/evalk.jsonl) from the released data/medgapqa.jsonl and print its sha256.
Set ITEMS_SHA in scripts/p2/run_eval.py to the printed value before running.
Fields used by the runs: id, hash, question, options, answer_idx, pert_question, removed_facts, removed_fact (the two facts joined by a space, as in the
file of the reported runs). removed_type / removed_types (the author's fact types) are not released and are written as null; the runs only copy them
into the output records. row_id is written as MG0001, MG0002, ... in file order; it appears only in the output records.
python scripts/make_items.py [--src data/medgapqa.jsonl] [--out runs/v5/evalk/evalk.jsonl]
"""
import argparse, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def convert(rec, n):
    return {"id": f"medqa_test_{rec['medqa_test_index']:05d}", "hash": rec["medqa_item_hash"], "question": rec["question_s0"], "options": rec["options"],
            "answer_idx": rec["answer_idx"], "pert_question": rec["question_fact_removed"], "removed_facts": rec["removed_facts"],
            "removed_types": [None, None], "removed_fact": " ".join(f.strip() for f in rec["removed_facts"]), "removed_type": None, "row_id": f"MG{n:04d}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "medgapqa.jsonl")); ap.add_argument("--out", default=str(ROOT / "runs" / "v5" / "evalk" / "evalk.jsonl"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fo:
        for n, line in enumerate(open(a.src, encoding="utf-8"), 1):
            fo.write(json.dumps(convert(json.loads(line), n), ensure_ascii=False) + "\n")
    print(f"wrote {out}\nITEMS_SHA = \"{hashlib.sha256(out.read_bytes()).hexdigest()}\"")


if __name__ == "__main__":
    main()
