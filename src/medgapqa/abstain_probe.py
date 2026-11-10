"""Best-guess probe after abstention (post hoc exploration, deviations D14). Frozen files are only read.

Targets: trajectories with outcome abstain (C1, C4; S0, S1, S2). After the same conversation record up to the abstaining turn (system prompt, item, previous actions and
observations, ABSTAIN output), PROBE is appended as a user message and one greedy generation is made. No tools or simulator; only the ANSWER action format
is allowed; format errors get one re-request with the reason. The generation step does not read gold labels; they are used only for scoring (cell_metrics).
Results are reported as "exploratory", separately from the main results, and are not used to change design, prompts or judgments.

python src/medgapqa/abstain_probe.py run --split dev --runs runs/prompt_tune/proto2/p3 --items runs/prompt_tune/items_dev50.jsonl \
    --model olmo2-7b --conds C4 --states S0 S1 --budget dev --out runs/abstain_probe_dev
python src/medgapqa/abstain_probe.py report --runs <runs> --c0 <c0 runs> --items <items> --probe <out>
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import (FIRST_TRAILER, NEXT_TRAILER, STATES, case_text, last_turn_sentence, load_config,  # noqa: E402
                   protocol, system_prompt, tools_of, turns_for, validate)
from prompts import PREFILL, retry_message  # noqa: E402

PROBE = ("You chose to ABSTAIN. Even so, give your single best guess among the options with the information you have. "
         "Reply with the option letter only.")
LETTERS = ["A", "B", "C", "D"]
OUTCOMES = ["correct", "wrong", "abstain", "unresolved", "parse_fail"]


def select_abstain(trs):
    return [t for t in trs if t["outcome"] == "abstain"]


def _trailer(turn, cond, cfg):
    if protocol(cfg) == "v2" and tools_of(cond) and turn == turns_for(cond, cfg):
        return last_turn_sentence(cond) + "\n\n" + NEXT_TRAILER
    return FIRST_TRAILER if turn == 1 else NEXT_TRAILER


def history_before_abstain(t, item, cond, cfg):
    """Conversation record just before the abstaining turn's input, and that turn's instruction (verifiable with the recorded input_sha)."""
    visible = item["question"] if t["state"] == "S0" else item["pert_question"]
    hist = [{"role": "system", "content": system_prompt(cond, cfg)},
            {"role": "user", "content": case_text(visible, item["options"])}]
    for rec in t["turns"][:-1]:
        hist.append({"role": "assistant", "content": rec.get("retry_raw", rec["raw"])})
        hist.append({"role": "user", "content": rec["observation"]})
    return hist, _trailer(t["turns"][-1]["turn"], cond, cfg)


def probe_messages(t, item, cond, cfg):
    hist, _ = history_before_abstain(t, item, cond, cfg)
    last = t["turns"][-1]
    return hist + [{"role": "assistant", "content": last.get("retry_raw", last["raw"])},
                   {"role": "user", "content": PROBE}]


def probe(t, item, cond, cfg, model):
    msgs = probe_messages(t, item, cond, cfg)
    raw, n_in, n_out, trunc = model.chat(msgs, PREFILL, cfg["max_new_tokens"])
    act, why = validate(raw, trunc, ["ANSWER"], item["options"], cfg["answer_normalization"])
    rec = {"id": t["id"], "hash": t["hash"], "cond": cond, "state": t["state"], "raw": raw, "invalid_reason": why,
           "retried": False, "tokens": {"in": n_in, "out": n_out}}
    if act is None:
        retry = msgs + [{"role": "assistant", "content": raw}, {"role": "user", "content": retry_message(why)}]
        raw2, i2, o2, tr2 = model.chat(retry, PREFILL, cfg["max_new_tokens"])
        act, why2 = validate(raw2, tr2, ["ANSWER"], item["options"], cfg["answer_normalization"])
        rec.update(retried=True, retry_raw=raw2, retry_invalid_reason=why2, retry_tokens={"in": i2, "out": o2})
    rec["guess"] = act["content"] if act else None
    rec["status"] = "ok" if act else "parse_fail"
    return rec


def _grade(t, gold):
    if t["outcome"] == "answer":
        return "correct" if t["answer"] == gold[t["hash"]] else "wrong"
    return t["outcome"]


def cell_metrics(cell, probes, c0, gold):
    """cell: trajectories of one model, condition and state; probes: {hash: probe result}; c0: C0 trajectories of the same model and state."""
    n = len(cell)
    ab = [t for t in cell if t["outcome"] == "abstain"]
    got = [probes[t["hash"]] for t in ab if t["hash"] in probes]
    ok = [p for p in got if p["status"] == "ok"]
    gc = {t["hash"] for t in ab if probes.get(t["hash"], {}).get("guess") == gold[t["hash"]]}
    correct = sum(_grade(t, gold) == "correct" for t in cell)
    letters = Counter(p["guess"] for p in ok)
    c0m = {t["hash"]: _grade(t, gold) for t in c0}
    trans = {o: {"guess_correct": 0, "guess_wrong": 0, "guess_invalid": 0} for o in OUTCOMES}
    for t in ab:
        p = probes.get(t["hash"])
        k = "guess_invalid" if not p or p["status"] != "ok" else ("guess_correct" if t["hash"] in gc else "guess_wrong")
        if t["hash"] in c0m:
            trans[c0m[t["hash"]]][k] += 1
    return {"label": "exploratory", "n": n, "n_abstain": len(ab), "n_probed": len(got), "probe_parse_fail": len(got) - len(ok),
            "guess_accuracy": len(gc) / len(ok) if ok else None, "chance": 0.25,
            "correct_resolution": correct / n if n else None,
            "abstain_allowed_accuracy": (correct + len(gc)) / n if n else None,
            "c0_correct_resolution": (sum(v == "correct" for v in c0m.values()) / len(c0m)) if c0m else None,
            "guess_letters": {k: (letters[k] / len(ok) if ok else None) for k in LETTERS},
            "c0_x_guess": trans}


def check_instance(name, chat, probes=None, refs=None):
    """Evaluation-model instance hash check (same fixed inputs and reference as run_agent). Does not run on mismatch."""
    from instances import verify_instance
    if probes is None or refs is None:
        from run_agent import REF_HASHES, consistency_probes
        probes = consistency_probes() if probes is None else probes
        refs = json.load(open(REF_HASHES)) if refs is None else refs
    ok, got = verify_instance(chat, probes, refs[name])
    if not ok:
        sys.exit(f"instance verification failed for {name}: {got}")
    return got


def load(p):
    return [json.loads(x) for x in open(p)]


def run(a):
    from budget import Budget
    from run_agent import HFModel, eval_guard
    if a.split == "eval":
        problems = eval_guard()
        if problems:
            sys.exit("EVAL probe refused: " + "; ".join(problems))
    cfg = load_config(allow_provisional=a.split == "dev")
    items = {x["hash"]: x for x in load(a.items)}
    out = Path(a.out) / a.model
    out.mkdir(parents=True, exist_ok=True)
    f = out / "probe.jsonl"
    done = {(r["cond"], r["state"], r["id"]) for r in load(f)} if f.exists() else set()
    if f.exists() and not a.resume:
        sys.exit(f"{f} exists (no overwrite; use --resume)")
    model, budget = HFModel(a.model), Budget(a.budget)
    check_instance(a.model, model.chat)
    print(f"instance verified: {a.model}", flush=True)
    with open(f, "a") as fo:
        for c in a.conds:
            for s in a.states:
                src = Path(a.runs) / a.model / c / s / "trajectories.jsonl"
                if not src.exists():
                    continue
                for t in select_abstain(load(src)):
                    if (c, s, t["id"]) in done:
                        continue
                    budget.check()
                    t0 = time.time()
                    r = probe(t, items[t["hash"]], c, cfg, model)
                    budget.charge(time.time() - t0, 1.0, note=f"abstain_probe/{a.model}/{c}/{s}")
                    fo.write(json.dumps({**r, "model": a.model}, ensure_ascii=False) + "\n")
                    fo.flush()


def report(a):
    gold = {x["hash"]: x["answer_idx"] for x in load(a.items)}
    rep = {}
    for pf in sorted(Path(a.probe).glob("*/probe.jsonl")):
        m = pf.parent.name
        rows = load(pf)
        for c in a.conds:
            for s in a.states:
                cell_p = Path(a.runs) / m / c / s / "trajectories.jsonl"
                c0_p = Path(a.c0) / m / "C0" / ("S1" if s == "S2" else s) / "trajectories.jsonl"
                if not cell_p.exists():
                    continue
                probes = {r["hash"]: r for r in rows if r["cond"] == c and r["state"] == s}
                rep[f"{m}/{c}/{s}"] = cell_metrics(load(cell_p), probes, load(c0_p) if c0_p.exists() else [], gold)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--split", choices=["dev", "eval"], default="dev")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--c0", default=None)
    ap.add_argument("--items", required=True)
    ap.add_argument("--model")
    ap.add_argument("--conds", nargs="+", default=["C1", "C4"])
    ap.add_argument("--states", nargs="+", default=list(STATES))
    ap.add_argument("--budget", choices=["dev", "eval"], default="dev")
    ap.add_argument("--out", default=None)
    ap.add_argument("--probe", default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    main()
