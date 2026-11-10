"""Phase 2 (D74) registration of the new evaluated models. Registered-hash files (panel_necessity.py, pipeline_k.py, agent.py, etc.) are not modified.

What install() does (all run-time overrides, applied only to P2 model names):
- adds the P2 model specs to panel_necessity.SPECS.
- turns panel_necessity.LetterModel into a factory that returns P2LetterModel for P2 names and the original class for any other name
  (pipeline_k.EvalModel does `from panel_necessity import LetterModel` inside __init__, so it reads this name).
- P2LetterModel uses the same generation and logit code as LetterModel and differs only in:
  (1) render passes the spec's template_kwargs to apply_chat_template (Qwen3 family: enable_thinking=False).
  (2) weight location (cache_dir), precision (dtype, bnb4) and device_map come from the spec.
  (3) D74-r4 stop rule (STOP_AT_FIRST_JSON): chat generation stops when the first JSON object closes. First JSON object = the dict completed from the
      first '{' of (prefill + generated text) by json.JSONDecoder.raw_decode (the same decoder as the frozen parser format_check.parse_last_json, including braces inside strings).
      After it closes, one more token is inspected: if it is EOS, generation ends as it would have (models that already stop give identical outputs and token counts);
      otherwise that token is dropped and generation stops early (truncated = False). The text is cut after the closing brace.
- changes pipeline_k.EvalModel.verify, for P2 names only, to write and compare the instance reference under runs/p2/refs/ (existing runs/v5/ref_*.json unchanged).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts")]
REFS = ROOT / "runs" / "p2" / ("refs_r4" if True else "refs")  # D74-r4: instance reference for the stop-rule version (pilot references under runs/p2/refs kept)
STOP_AT_FIRST_JSON = True
CACHE = "$PROJECT_ROOT/model_cache"  # SSD HF cache (approved in D74-r1; the data disk had no free inodes)
NO_THINK = {"enable_thinking": False}
SPECS = {
    "qwen3-8b": {"repo": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "template_kwargs": NO_THINK},
    "qwen3-14b": {"repo": "Qwen/Qwen3-14B", "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18", "template_kwargs": NO_THINK},
    "meerkat2-14b": {"repo": "aigensciences/Meerkat2-14B", "revision": "25bdbc341e01d968ffc95b041334bb9f82af524b", "template_kwargs": NO_THINK},
    "qwen3-30b-a3b": {"repo": "Qwen/Qwen3-30B-A3B", "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39", "template_kwargs": NO_THINK},
    "meerkat2-30b-a3b": {"repo": "aigensciences/Meerkat2-30B-A3B", "revision": "35b5206280d690f3eb6ddc1d0d3ef9071294a2eb", "template_kwargs": NO_THINK},
    "qwen3-32b": {"repo": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137", "template_kwargs": NO_THINK},
    "meerkat2-32b": {"repo": "aigensciences/Meerkat2-32B", "revision": "97c25748bc7a9452c6e44df9b90032457db5dd8c", "template_kwargs": NO_THINK},  # D94
    # D74-r6: Gemma family. The template has no thinking toggle (empty kwargs). For multimodal checkpoints only the text model (same key mapping as run_agent.load_text_only),
    # adds the end-of-turn token <end_of_turn> to the stop tokens (generation_config eos [1, 106]). Specs without the flag below (Qwen3, Meerkat2) are unchanged.
    "medgemma-1.5-4b": {"repo": "google/medgemma-1.5-4b-it", "revision": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b", "template_kwargs": {},
                        "text_only": True, "extra_eos": ["<end_of_turn>"]},
    "medgemma-27b": {"repo": "google/medgemma-27b-text-it", "revision": "5b667cf2ddcf064085bc90952edb35a0edbfb79c", "template_kwargs": {},
                     "extra_eos": ["<end_of_turn>"], "bad_words": ["<unused94>", "<unused95>"]},  # D74-r10: block the Gemma thought marker (all generation)
    # D74-r7: large. Meerkat-70B is loaded from the D41 NF4 checkpoint (SSD copy, model.safetensors sha256 29a10dfe...; quantization settings in its config).
    # Qwen3-235B-A22B: original BF16 weights (local cache) spread over all visible GPUs with device_map="auto" (D74-r5; bnb NF4 cannot quantize the experts).
    "meerkat-70b-nf4": {"path": "$PROJECT_ROOT/model_cache/nf4/meerkat-70b-nf4", "template_kwargs": {}},
    "qwen3-235b-a22b": {"repo": "Qwen/Qwen3-235B-A22B", "revision": "8efa61729e24bd65b1d152b5ab5409052aa80e65", "template_kwargs": NO_THINK,
                        "cache": "$PROJECT_ROOT/hf_cache/hub", "device_map": "auto", "max_memory_gib": 84},  # D80: 6 GPUs x 84 GiB
    # D82: DeepSeek-V4-Flash-0731. No Jinja chat template; rendered with the official encoding/encoding_dsv4.py (copied unchanged to src/medgapqa/vendor/),
    # Instant = official non-thinking mode thinking_mode="chat" (</think> after <｜Assistant｜>). FP8 checkpoint, device_map auto over the visible GPUs.
    # running needs kernels 0.17.2 (PYTHONPATH=$PROJECT_ROOT/kernels_p2) and the FP8 kernel cache (HF_HOME=$PROJECT_ROOT/hf_home) (D76 (6)).
    "dsv4-0731": {"repo": "deepseek-ai/DeepSeek-V4-Flash-0731", "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062", "template_kwargs": {},
                  "encoder": "dsv4_chat", "device_map": "auto",
                  "fp8_kernel_revision": "0e5764aa5c27d0f005117a92645156481540950a"},  # D82: pinned to the cached kernels-community/finegrained-fp8 (v4) commit (offline load)
}
GEMMA3_TEXT_KEYS = {r"^language_model\.model\.": "model.", r"^language_model\.lm_head\.": "lm_head."}  # same as run_agent.py
VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.")
PRECISION = {k: "bf16" for k in SPECS}  # for the result record (D74 decision 6)
PRECISION["meerkat-70b-nf4"] = "nf4"
PRECISION["dsv4-0731"] = "fp8 (checkpoint)"


def render_dsv4(msgs):
    """D82: official encoder, non-thinking mode. The generation prompt ends with '<｜Assistant｜></think>' after the last user turn."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
    from encoding_dsv4 import encode_messages
    return encode_messages([dict(m) for m in msgs], thinking_mode="chat")


def render(tok, msgs, merge_system, template_kwargs):
    """Same as panel_necessity.LetterModel.render but also passes template_kwargs (an empty dict gives the original string)."""
    if merge_system and msgs and msgs[0]["role"] == "system":
        msgs = [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}] + msgs[2:]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **template_kwargs)


def template_ok(tok, template_kwargs):
    try:
        out = tok.apply_chat_template([{"role": "system", "content": "SYSMARK"}, {"role": "user", "content": "x"}],
                                      add_generation_prompt=True, tokenize=False, **template_kwargs)
        return "SYSMARK" in out
    except Exception:
        return False


def first_json_end(text):
    """End position if a JSON dict starting at the first '{' of text is complete, else None (same raw_decode as the frozen parser)."""
    i = text.find("{")
    if i == -1:
        return None
    try:
        obj, end = json.JSONDecoder().raw_decode(text, i)
    except json.JSONDecodeError:
        return None
    return end if isinstance(obj, dict) else None


def install():
    import panel_necessity as PN
    import pipeline_k as PK
    from forced_answer import letter_ids
    if getattr(PN, "_p2_installed", False):
        return
    Orig = PN.LetterModel

    class P2LetterModel(Orig):
        def __init__(self, name):
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            assert CACHE, "CACHE (weight location) is not set"
            self.torch, self.name, s = torch, name, SPECS[name]
            self.template_kwargs = dict(s.get("template_kwargs", {}))
            src = s.get("path") or s["repo"]
            kw = {} if "path" in s else {"revision": s["revision"], "cache_dir": s.get("cache", CACHE)}
            self.tok = AutoTokenizer.from_pretrained(src, local_files_only=True, **kw)
            mk = dict(dtype=torch.bfloat16, device_map=s.get("device_map", {"": "cuda:0"}), local_files_only=True, **kw)
            if s.get("fp8_kernel_revision"):  # D82: pinned to the cached commit instead of an online version lookup
                import transformers.integrations.hub_kernels as HK
                HK._HUB_KERNEL_MAPPING["finegrained-fp8"] = {"repo_id": "kernels-community/finegrained-fp8", "revision": s["fp8_kernel_revision"]}
            if s.get("max_memory_gib"):  # D80: split evenly with the same cap on every visible GPU (first 235B attempt ran out of memory on GPU 0)
                mk["max_memory"] = {i: f"{s['max_memory_gib']}GiB" for i in range(torch.cuda.device_count())}
            if s.get("bnb4"):
                mk["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
            if s.get("text_only"):  # same way as run_agent.load_text_only; stops if language-model weights are missing or left over
                from transformers import AutoConfig, Gemma3ForCausalLM
                cfg = AutoConfig.from_pretrained(s["repo"], local_files_only=True, **kw)
                self.model, info = Gemma3ForCausalLM.from_pretrained(s["repo"], config=cfg.text_config, key_mapping=GEMMA3_TEXT_KEYS,
                                                                     output_loading_info=True, **mk)
                bad = (list(info["missing_keys"]) + list(info["mismatched_keys"])
                       + [k for k in info["unexpected_keys"] if not k.startswith(VISION_PREFIXES)])
                if bad:
                    raise RuntimeError(f"text-only load of {name} incomplete: {bad[:5]}")
                self.model.eval()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(src, **mk).eval()
            self.encoder = s.get("encoder")
            self.merge_system = False if self.encoder else not template_ok(self.tok, self.template_kwargs)
            self.ids = letter_ids(self.tok, self.render([{"role": "system", "content": "s"}, {"role": "user", "content": "x"}]) + PN.FORCED_PREFILL)
            v = self.tok.get_vocab()
            self.eos = sorted({self.tok.eos_token_id, *[v[t] for t in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>", "<|end|>") if t in v],
                               *[v[t] for t in s.get("extra_eos", []) if t in v]} - {None})
            self.bad_words_ids = [[v[t]] for t in s.get("bad_words", [])] or None  # D74-r10: stops with KeyError if the token does not exist

        def render(self, msgs):
            if self.encoder == "dsv4_chat":
                return render_dsv4(msgs)
            return render(self.tok, msgs, self.merge_system, self.template_kwargs)

        def chat(self, msgs, prefill, max_new):
            """Same format as LetterModel.chat (text, n_in, n_out, truncated), greedy. With STOP_AT_FIRST_JSON, the D74-r4 stop rule."""
            if not STOP_AT_FIRST_JSON:
                return Orig.chat(self, msgs, prefill, max_new)
            from transformers import StoppingCriteria, StoppingCriteriaList
            prompt = self.render(msgs) + prefill
            ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.model.device)
            n0, tok = ids.shape[1], self.tok

            class AfterFirstJSON(StoppingCriteria):
                def __call__(self, input_ids, scores, **kw):  # stop if the first JSON already closed in the text without the last token
                    prev = input_ids[0, n0:-1]
                    done = len(prev) > 0 and first_json_end(prefill + tok.decode(prev, skip_special_tokens=True)) is not None
                    return self_torch.full((input_ids.shape[0],), done, dtype=self_torch.bool, device=input_ids.device)
            self_torch = self.torch
            with self.torch.no_grad():
                out = self.model.generate(ids, attention_mask=self.torch.ones_like(ids), do_sample=False, max_new_tokens=max_new,
                                          eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id or self.eos[0],
                                          stopping_criteria=StoppingCriteriaList([AfterFirstJSON()]), bad_words_ids=self.bad_words_ids)
            new = out[0, n0:]
            stopped = False
            if len(new) > 1 and int(new[-1]) not in self.eos and first_json_end(prefill + tok.decode(new[:-1], skip_special_tokens=True)) is not None:
                new, stopped = new[:-1], True  # a non-EOS token generated after the object closed is dropped
            truncated = (not stopped) and len(new) >= max_new and int(new[-1]) not in self.eos
            text = prefill + tok.decode(new, skip_special_tokens=True).strip()
            if stopped:
                end = first_json_end(text)
                text = text[:end] if end is not None else text
            self.last_stop = {"stopped_after_json": stopped}
            return text, int(n0), int(len(new)), truncated

    def factory(name):
        return P2LetterModel(name) if name in SPECS else Orig(name)

    PN.SPECS.update({k: ({"path": v["path"]} if "path" in v else {"repo": v["repo"], "revision": v["revision"]}) for k, v in SPECS.items()})  # D74-r8: path specs
    PN.LetterModel, PN.OrigLetterModel, PN.P2LetterModel = factory, Orig, P2LetterModel
    orig_verify = PK.EvalModel.verify

    def verify(self):
        if self.name not in SPECS:
            return orig_verify(self)
        from instances import sha as isha
        from run_agent import consistency_probes
        got = [isha(self.chat(m, p, n)[0]) for m, p, n in consistency_probes()]
        ref = REFS / f"ref_{self.name}.json"
        if not ref.exists():
            REFS.mkdir(parents=True, exist_ok=True)
            ref.write_text(json.dumps({"hashes": got, "instance": PK.inst_id(self.name)}, indent=1))
            return True, "reference created (runs/p2/refs)"
        return got == json.loads(ref.read_text())["hashes"], "compared to runs/p2/refs"

    PK.EvalModel.verify = verify
    PN._p2_installed = True
