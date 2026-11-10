"""Runs the agent loop with an evaluation model. Currently for DEV checks only (EVAL is refused before the freeze).

Usage: CUDA_VISIBLE_DEVICES=1 python src/medgapqa/run_agent.py --model qwen2.5-7b --n 10 --runs C4:S0,S1,S2 C0:S0,S1 \
        --sim-port 8700 --out runs/dev_smoke
Output: {out}/{model}/{cond}/{state}/trajectories.jsonl, meta.json. Correctness and accuracy are not computed (src/medgapqa/grade.py).
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import CONDITIONS, load_config, run_episode  # noqa: E402
from llm_client import RemoteGenerator  # noqa: E402
from parse_check import CACHE, MODELS, REVISIONS  # noqa: E402
from retrieve import Retriever  # noqa: E402
from budget import Budget, BudgetStop  # noqa: E402
from infra_retry import call_with_infra_retry, write_episodes  # noqa: E402
from instances import PooledBackend, instance_id, shard, verify_instance  # noqa: E402
from simulator import Simulator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_tok(name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODELS[name], cache_dir=CACHE.get(name, str(ROOT / "hf_cache" / "hub")),
                                         revision=REVISIONS[name])


# deviations D15 (MedGemma-4B). The vocabularies of the three existing models have no <end_of_turn>, and they are not in TEXT_ONLY or NO_SYSTEM_ROLE either, so
# rendering and stop tokens are unchanged (tests/test_render_d15.py, reference hashes from freeze-v0.4.1).
EOS_TOKENS = ("<|eot_id|>", "<|im_end|>", "<end_of_turn>")
NO_SYSTEM_ROLE = {"medgemma-4b"}  # no system role: the system text is prepended to the first user turn (same string as the Gemma template's own handling)
TEXT_ONLY = {"medgemma-4b"}  # load only the text model from the multimodal checkpoint (vision tower and projector excluded)
GEMMA3_TEXT_KEYS = {r"^language_model\.model\.": "model.", r"^language_model\.lm_head\.": "lm_head."}
VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.")


def template_messages(name, msgs):
    if name not in NO_SYSTEM_ROLE or not msgs or msgs[0]["role"] != "system":
        return msgs
    assert msgs[1]["role"] == "user"
    return [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}] + msgs[2:]


def load_text_only(name, kw):
    """Loads only the language-model weights of a Gemma3ForConditionalGeneration checkpoint into Gemma3ForCausalLM.
    Does not run if any language-model weight is missing or left over (other than vision and projector)."""
    from transformers import AutoConfig, Gemma3ForCausalLM
    cfg = AutoConfig.from_pretrained(MODELS[name], cache_dir=kw["cache_dir"], revision=kw["revision"])
    model, info = Gemma3ForCausalLM.from_pretrained(MODELS[name], config=cfg.text_config, key_mapping=GEMMA3_TEXT_KEYS,
                                                    output_loading_info=True, **kw)
    bad = (list(info["missing_keys"]) + list(info["mismatched_keys"])
           + [k for k in info["unexpected_keys"] if not k.startswith(VISION_PREFIXES)])
    if bad:
        raise RuntimeError(f"text-only load of {name} incomplete: {bad[:5]}")
    return model


class HFModel:
    """greedy, batch 1. chat(msgs, prefill, max_new) -> (text, n_in, n_out, truncated)."""

    def __init__(self, name):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.name = name
        self.tok = load_tok(name)
        kw = dict(cache_dir=CACHE.get(name, str(ROOT / "hf_cache" / "hub")), revision=REVISIONS[name],
                  dtype=torch.bfloat16, device_map={"": "cuda:0"})
        self.model = load_text_only(name, kw) if name in TEXT_ONLY else AutoModelForCausalLM.from_pretrained(MODELS[name], **kw)
        self.model.eval()
        v = self.tok.get_vocab()
        self.eos = [self.tok.eos_token_id] + [v[t] for t in EOS_TOKENS if t in v and v[t] != self.tok.eos_token_id]

    def chat(self, msgs, prefill, max_new):
        return call_with_infra_retry(self._chat, msgs, prefill, max_new)

    def _chat(self, msgs, prefill, max_new):
        prompt = self.tok.apply_chat_template(template_messages(self.name, msgs), add_generation_prompt=True,
                                              tokenize=False) + prefill
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(ids, attention_mask=self.torch.ones_like(ids), do_sample=False,
                                      max_new_tokens=max_new, eos_token_id=self.eos,
                                      pad_token_id=self.tok.pad_token_id or self.eos[0])
        new = out[0, ids.shape[1]:]
        truncated = len(new) >= max_new and int(new[-1]) not in self.eos
        text = prefill + self.tok.decode(new, skip_special_tokens=True).strip()
        return text, int(ids.shape[1]), int(len(new)), truncated


REF_HASHES = ROOT / "metadata" / "instance_ref_hashes.json"
EVAL_TAG = "freeze-v0.4.2"  # deviations D13, D15
MANIFEST_REL = "metadata/freeze2_manifest.json"


def _git_tag_message(tag, root=ROOT):
    r = subprocess.run(["git", "-C", str(root), "tag", "-l", "--format=%(contents)", tag], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 and r.stdout.strip() else None


def eval_guard(root=ROOT, tag=EVAL_TAG, tag_message=_git_tag_message):
    """EVAL permission conditions (D13). Returns a list of problems (empty list = allowed).
    (i) tag exists (ii) frozen file and data hashes match the manifest (iii) the manifest file hash matches the value in the tag message."""
    root = Path(root)
    msg = tag_message(tag)
    if msg is None:
        return [f"tag {tag} missing"]
    problems = []
    man_path = root / MANIFEST_REL
    m = re.search(re.escape(MANIFEST_REL) + r" sha256 ([0-9a-f]{64})", msg)
    if not m:
        problems.append(f"tag {tag} message has no {MANIFEST_REL} sha256")
    elif hashlib.sha256(man_path.read_bytes()).hexdigest() != m.group(1):
        problems.append(f"{MANIFEST_REL} hash differs from tag {tag}")
    man = json.loads(man_path.read_text())
    for f, h in {**man.get("code_hashes", {}), **man.get("data_hashes", {})}.items():
        p = root / f
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            problems.append(f"frozen file changed or missing: {f}")
    return problems


def consistency_probes(n=8):
    """Fixed inputs for instance verification: the first C4 S0 turn of the first 8 DEV items (default prompt v4, greedy)."""
    from agent import FIRST_TRAILER, _with_trailer, case_text, system_prompt
    from prompts import PREFILL
    # fixed to the exact inputs used to make the reference hashes (protocol v1, prompt v4). Instance-check inputs do not change even if settings change
    cfg = dict(load_config(), protocol="v1", prompt_variant="v4")
    items = [json.loads(line) for line in open(ROOT / "data" / "perturbed_dev.jsonl")][:n]
    return [(_with_trailer([{"role": "system", "content": system_prompt("C4", cfg)},
                            {"role": "user", "content": case_text(it["question"], it["options"])}], FIRST_TRAILER),
             PREFILL, cfg["max_new_tokens"]) for it in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "eval"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--runs", nargs="+", required=True, help="COND:STATE,STATE ...")
    ap.add_argument("--sim-ports", default="8700", help="verified simulator server ports (comma-separated); results do not depend on the number of ports")
    ap.add_argument("--shard", default="0/1", help="k/n: run only items[k::n] (merge in manifest order)")
    ap.add_argument("--budget", choices=["eval", "dev"], required=True)
    ap.add_argument("--gpu-weight", type=float, default=1.0,
                    help="GPUs per episode-second (1 for the evaluated model + this process's share of simulator GPUs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="after an infrastructure stop, run only the unfinished items")
    ap.add_argument("--prompt-variant", default=None, help="DEV prompt tuning candidate (v4/p1/p2/p3), DEV only")
    ap.add_argument("--items", default=None, help="DEV item file (default data/perturbed_{split}.jsonl)")
    a = ap.parse_args()
    if a.split == "eval":
        problems = eval_guard()
        if problems:
            sys.exit("EVAL run refused: " + "; ".join(problems))
        if a.prompt_variant or a.items:
            sys.exit("EVAL does not use --prompt-variant/--items (frozen configuration and items only)")
        cfg = load_config(allow_provisional=False)
    else:
        cfg = load_config(allow_provisional=True)
    if a.prompt_variant:
        cfg["prompt_variant"] = a.prompt_variant
    src = Path(a.items) if a.items else ROOT / "data" / f"perturbed_{a.split}.jsonl"
    items = [json.loads(line) for line in open(src)][: a.n]
    k, n = map(int, a.shard.split("/"))
    items = shard(items, k, n)
    budget = Budget(a.budget)
    budget.check()
    model = HFModel(a.model)
    ok, got = verify_instance(model.chat, consistency_probes(), json.load(open(REF_HASHES))[a.model])
    if not ok:
        sys.exit(f"instance verification failed for {a.model} on {instance_id()}: {got}")
    inst = instance_id(a.model)
    retriever = Retriever.load(cfg["search_top_k"], cfg["search_max_tokens"], load_tok(cfg["search_truncation_tokenizer"]))
    ports = [int(x) for x in a.sim_ports.split(",")]
    pool = PooledBackend([RemoteGenerator(p) for p in ports], [f"sim:{p}" for p in ports])
    simulator = Simulator(pool, cfg["simulator_version"])  # S1 and S2 share the classification cache within one model
    for spec in a.runs:
        cond, states = spec.split(":")
        assert cond in CONDITIONS
        for state in states.split(","):
            d = Path(a.out) / a.model / cond / state
            d.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            def episode(it, state=state, cond=cond):
                budget.check()  # at the hard stop line, this item is not started and the run stops (resume with --resume)
                s = time.time()
                tr = run_episode(it, state, cond, model, cfg, simulator, retriever)
                tr["seconds"] = round(time.time() - s, 2)
                tr["instance"] = inst
                try:
                    budget.charge(tr["seconds"], a.gpu_weight, note=f"{a.model}/{cond}/{state}")
                except BudgetStop:
                    pass  # this episode is recorded and the run stops at the next check
                return tr
            # no overwriting; after interruption, --resume continues only unfinished items (no result-based reruns)
            out_name = "trajectories.jsonl" if n == 1 else f"trajectories.shard{k}of{n}.jsonl"
            try:
                progress = write_episodes(d / out_name, items, episode, resume=a.resume)
            except BudgetStop as e:
                print(f"BUDGET STOP: {e}", flush=True)
                sys.exit(3)
            trs = [json.loads(line) for line in open(d / out_name)]
            turns = [t for x in trs for t in x["turns"]]
            metrics = {"turns": len(turns), "first_try_invalid": sum(t["invalid_reason"] is not None for t in turns),
                       "retries": sum("retry_raw" in t for t in turns),
                       "parse_fail": sum(x["outcome"] == "parse_fail" for x in trs),
                       "unresolved_cause": dict(Counter(x["unresolved_cause"] for x in trs if x.get("unresolved_cause"))),
                       "budget_exhausted": sum(bool(x.get("budget_exhausted")) for x in trs)}
            metrics["first_try_invalid_rate"] = round(metrics["first_try_invalid"] / len(turns), 3) if turns else None
            meta = {"metrics": metrics, "progress": progress,
                    "infra_retries_simulator": sum(len(b.c.retry_log.events) for b in pool.backends),
                    "shard": a.shard, "instance": inst, "sim_instances": pool.names, "budget": a.budget,
                    "budget_total_gpu_hours": round(budget.total(), 3), "model": a.model, "repo": MODELS[a.model], "revision": REVISIONS[a.model], "cond": cond,
                    "state": state, "split": a.split, "n": len(items), "config": cfg,
                    "items_sha": hashlib.sha256("".join(i["hash"] for i in items).encode()).hexdigest()[:16],
                    "wall_seconds": round(time.time() - t0, 1), "simulator": "llm_server ports %s" % a.sim_ports}
            json.dump(meta, open(d / ("meta.json" if n == 1 else f"meta.shard{k}of{n}.json"), "w"), indent=1, ensure_ascii=False)
            print(f"{a.model} {cond} {state} done {meta['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
