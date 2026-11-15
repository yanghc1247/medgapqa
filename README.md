# MedGapQA

Benchmark items, code, prompts, labels and aggregated results for "Ignorance Is Not Bliss: Do Language Models Ask for Missing Clinical Information?" (Heechul Yang and Jaewoo Kang, Korea University; corresponding author: Jaewoo Kang, kangj@korea.ac.kr).

MedGapQA (Medical information-Gap QA) has 736 MedQA questions. In each, an LLM author removed two decisive patient facts, which can be recovered only by asking a patient/chart simulator. Each item comes in three states: intact (S0), fact-removed with the facts available on request (S1), and fact-removed with the facts withheld (S2). Each item is also labeled for whether the removed facts are necessary.

Benchmark items, code, prompts, labels and aggregated results are released here; trajectories are not, because they contain StatPearls passages (CC BY-NC-ND). The items (`data/medgapqa.jsonl`) are derived from MedQA (Jin et al., 2021), released under the MIT License; the MedQA copyright and permission notice is in `data/LICENSE_MedQA`. Our code, edits and labels are released under the repository `LICENSE` (MIT).

Supplementary material for the paper (`supplement.pdf`).

## Results in brief

Numbers are copied from the paper; intervals are 95% item-clustered paired bootstrap intervals.

- Over the 9 main-pool models (Instant mode), removing two facts lowers accuracy by 0.207 [0.185, 0.229] on average (0.121 to 0.255 per model, large models included).
- A probe asking whether the information suffices separates fact-removed from intact cases with an AUROC of only 0.61–0.76 across all 11 models.
- In the targeted ask pipeline, 11.3%–29.3% of questions in the main pool elicit a removed fact (forced inquiry: 9.0%–36.1%).
- On the Instant main mean, the targeted ask pipeline gains +0.041 [+0.032, +0.051], forced inquiry +0.023 [+0.012, +0.035] and the free agent +0.002; neither improvement policy met the criterion fixed before the runs.
- Handing over both removed facts recovers most of the lost accuracy (recovery ratio 0.85–1.10 per model); the best Instant mean gain is about 20% of this both-facts reference (+0.205).
- On the same items, facts received through targeted questions yield 0.43 [0.37, 0.49] of the gain from handing the same facts over (use efficiency, Instant main mean; post hoc), suggesting that both acquiring and using the facts limit the models in Instant mode; in thinking mode the hybrid models' efficiency rises from 0.46 to 0.86, leaving acquisition as the main remaining limit.
- In thinking mode, the 7 hybrid models answer at once more often (91.8% against 50.6% of free-agent episodes) and gain no more from the free agent or forced inquiry, but the targeted ask pipeline's mean gain rises from +0.047 to +0.128 (difference +0.081 [+0.064, +0.099]; about 48% of the both-facts gain).

## Pre-specified and exploratory analyses

Fixed before the runs: forced inquiry or the targeted ask pipeline counts as an improvement if, on the main mean, its gain over answering immediately (S1) is at least +0.050 with the lower bound of the 97.5% interval above zero, and its change in S2 accuracy is at least −0.020.
The comparison of Instant with Think was specified in the plan without a decision rule; it uses the 7 hybrid models, both modes on the same models with differences paired over items.
Exploratory or post hoc, with no multiplicity correction: the necessity strata, the recovery ratio by stratum, the blind asker against the targeted ask pipeline (B3), the correlation between models, the probe AUROC by necessity stratum, and the probe AUROC difference within fine-tune/base pairs.
Details are in the supplementary material (`supplement.pdf`).

## Models

All evaluated models run in Instant mode (thinking off, greedy decoding); the hybrid Qwen3 and Meerkat2 models also run in thinking mode (temperature 0.6, top-p 0.95, top-k 20, a fixed seed per request, at most 32,768 thinking tokens). DeepSeek-V4-Flash also runs in thinking mode. Generation uses vLLM 0.32.0, except Meerkat-70B (transformers, Instant only). Main-pool models run in BF16.

| Model | Role | Hugging Face repository | Revision |
|---|---|---|---|
| Qwen3-8B | main pool | `Qwen/Qwen3-8B` | `b968826d…` |
| Qwen3-14B | main pool | `Qwen/Qwen3-14B` | `40c06982…` |
| Meerkat2-14B | main pool | `aigensciences/Meerkat2-14B` | `25bdbc34…` |
| MedGemma-4B | main pool (outputs of an earlier run of the same frozen pipeline) | `google/medgemma-4b-it` | `290cda5e…` |
| Qwen3-30B-A3B | main pool | `Qwen/Qwen3-30B-A3B` | `ad44e777…` |
| Meerkat2-30B-A3B | main pool | `aigensciences/Meerkat2-30B-A3B` | `35b52062…` |
| Qwen3-32B | main pool | `Qwen/Qwen3-32B` | `9216db57…` |
| MedGemma-27B | main pool | `google/medgemma-27b-text-it` | `5b667cf2…` |
| Meerkat2-32B | main pool | `aigensciences/Meerkat2-32B` | `97c25748…` |
| Meerkat-70B | large (NF4) | `dmis-lab/llama-3-meerkat-70b-v1.0` | `6f6af4f1…` |
| DeepSeek-V4-Flash | large (native FP4/FP8) | `deepseek-ai/DeepSeek-V4-Flash-0731` | `7872f01b…` |
| Qwen2.5-72B-Instruct | patient/chart simulator (NF4) | `Qwen/Qwen2.5-72B-Instruct` | `495f3936…` |

## Layout

| Path | Contents |
|---|---|
| `src/medgapqa/` | Model wrapper (`p2_models.py`, with the DeepSeek-V4-Flash encoder in `vendor/`), agent loop and policies (free agent, forced inquiry), targeted ask pipeline, simulator client and server, scoring (`g1.py` letter rule, first-JSON stop rule in `p2_models.py`, `format_check.py`), BM25 search over StatPearls |
| `scripts/think/veval.py` | vLLM runner of the reported results: every cell for one model in Instant or Think mode (no generation length limit; re-request echo clipped); `vserve.py` starts the vLLM server, `think_run.py` holds the shared client code |
| `scripts/think/hf_cont.py` | Continues the transformers episodes of Meerkat-70B that had reached the earlier action-turn limit, without a limit |
| `scripts/p2/run_eval.py` | Earlier transformers runner (every cell for one model; items written by `scripts/make_items.py`) |
| `scripts/p2/p2_*.py`, `scripts/p2/analysis_vllm.py` | Analysis scripts; `analysis_vllm.py` builds the Instant and Think evaluation roots and runs them |
| `scripts/p2/vllm_final.py`, `scripts/p2/engine_agreement.py` | Final tables (Instant, Think, 7-model paired means, strata, pairs, seeds) and the engine check against the transformers runs |
| `scripts/paper/d131_rerun_extra.py`, `scripts/paper/d131_share.py` | Extra analyses on the vLLM root (probe strata and pairs, blind asker, earlier pool, 7-model Instant mean, Think minus Instant cost) and the share of the both-facts gain |
| `scripts/` (other) | Modules that `run_eval.py` imports: hash checks (`p2/run_pilot.py`), the C0 S2 cell (`run_devk2_explore.py`), oracle prompts (`consol_requery.py`, `d56/prompts_d56.py`) and the noticing-probe prompt (`d58/prompts_d58.py`) |
| `configs/agent.json` | Agent configuration |
| `metadata/instance_ref_hashes.json` | Reference output hashes for the model-instance check of MedGemma-4B and of the models of an earlier run |
| `prompts/` | Frozen manifests of the two improvement policies and `registered_prompts.md` (every pre-specified prompt text with its sha256) |
| `data/` | `medgapqa.jsonl` (the 736 benchmark items with their labels), `README.md` (fields), `LICENSE_MedQA` (MedQA license notice) |
| `results/` | Aggregate results used in the paper: `analysis_vllm/instant/` and `analysis_vllm/think/` (p2_analysis, strata, recovery, correlation, pair accuracy per mode), `analysis_vllm/final_tables.json` (7-model paired means, strata, pairs, seeds, large models), `analysis_vllm/engine_agreement.json` (engine check), `analysis_vllm_paper/` (extra analyses and the share of the both-facts gain) |
| `supplement.pdf` | Supplementary material |
| `tests/test_smoke.py` | CPU smoke test: compile, prompt hashes, hash guards |
| `docs/deviations.md` | Changes from the pre-specified plan (same content as the supplement section) |

Some module names reflect the project's development history; the files listed under scripts/ and src/ are the ones used for the reported runs. Some modules also contain entry points from earlier runs that need files not released here (for example the report command of c4_ref_k.py); only the commands under Running and Reproducing are supported. D-numbers in code comments and result metadata refer to entries of the internal decision log, which is not released; the changes from the pre-specified plan are in `docs/deviations.md`.

Internal names in the code map to the paper as follows: S1K, S2K = S1, S2; C0 = answer only; C4 = free agent; C4f = forced inquiry; P6 = targeted ask pipeline; O2 = both removed facts handed over; O1a, O1b = one fact handed over; NP = noticing probe.

## Installation

- Python 3.13 and `pip install -r requirements.txt`; Qwen3, Meerkat2 and MedGemma need a `transformers` version that supports them.
- vLLM 0.32.0 in its own environment at `$PROJECT_ROOT/venv_vllm` (`scripts/think/vserve.py` starts `venv_vllm/bin/vllm serve`).
- DeepSeek-V4-Flash in vLLM on our GPUs (RTX PRO 6000, SM120) also needed the FlashInfer DSV4 kernels (`flashinfer-cubin` and `flashinfer-jit-cache` 0.7.0.post1) and tensor parallelism over four GPUs.
- Inference needs CUDA GPUs. DeepSeek-V4-Flash also needs the `kernels` package and its FP8 kernel (see the comments in `src/medgapqa/p2_models.py`).
- The smoke test and the analysis scripts run on CPU.

Paths that pointed to local disks are masked as `$PROJECT_ROOT`, `$HF_CACHE`, `$DATA_ROOT` and `$HOME`; set them before running (for example `CACHE` in `src/medgapqa/p2_models.py`).

## Running the models

The runners read the items from `runs/v5/evalk/evalk.jsonl` and check its sha256 against `ITEMS_SHA` in `scripts/think/veval.py` (and `scripts/p2/run_eval.py`). `python scripts/make_items.py` writes that file from `data/medgapqa.jsonl` and prints its sha256; set `ITEMS_SHA` to this value before running. The internal file of the reported runs also held authoring fields that are not used by the runs and not released (the author's fact types, recovery questions and runner-up option, and internal row ids), so its hash differs; `make_items.py` writes the author's fact types as null and row ids `MG0001`, `MG0002`, …, which appear only in the output records.

```bash
# 1. patient/chart simulator server (transformers, greedy; one GPU)
CUDA_VISIBLE_DEVICES=0 python src/medgapqa/llm_server_c.py --model c1 --port 8714 --log runs/v5/sim_c1.log
# 2. frozen manifests where the runners expect them
mkdir -p runs/v5/d36 runs/v5/d37
cp prompts/FROZEN_C4F_v3.json runs/v5/d36/ && cp prompts/FROZEN_P6_v1.json runs/v5/d37/
# 3. vLLM server for one model (max length 40960 for Qwen3 and Meerkat2, 131072 for MedGemma and DeepSeek-V4-Flash, which needs 4 GPUs with --gpus 0,1,2,3);
#    --batch-invariant for Instant (deterministic), without it for Think
python scripts/think/vserve.py --model qwen3-14b --gpus 1 --port 8851 --max-len 40960 --batch-invariant --log runs/srv_qwen3-14b.log
# 4. all cells for one model and mode: C0 (S0, S1, S2), oracle, noticing probe, free agent, targeted ask, forced inquiry
python scripts/think/veval.py --model qwen3-14b --mode instant --port 8851 --sim-ports 8714 --out runs/p2/eval_vllm_instant
python scripts/think/veval.py --model qwen3-14b --mode think --port 8851 --sim-ports 8714 --out runs/p2/eval_vllm_think
```

`veval.py` imports the pre-specified code unchanged and checks its hashes; it sets no generation length limit (stop at the first complete JSON object, the end-of-sequence token or the context end), echoes only the first 1,024 tokens of a failed output in a format re-request, and caches the simulator's classifications on disk. Meerkat-70B (NF4) was run in Instant mode with the earlier transformers runner `scripts/p2/run_eval.py`; Meerkat-70B episodes that had reached the earlier action-turn limit were continued without a limit with `scripts/think/hf_cont.py`. The BM25 index over StatPearls is rebuilt from the NCBI archive with `src/medgapqa/statpearls_chunk.py` and `src/medgapqa/build_index.py`; the corpus and index are not redistributed.

## Changes from the pre-specified plan

The engine (vLLM instead of transformers), the removal of the generation length limits, the clipped re-request echo and the added thinking runs are described in `docs/deviations.md`, with the same content as the supplement section "Changes from the pre-specified plan". The pre-specified files are unchanged.

## Reproducing the results

1. `python tests/test_smoke.py` (CPU, no data needed) compiles every module and checks the pre-specified prompt hashes and hash guards.
2. With the run outputs under `runs/p2/eval_vllm_instant/` and `runs/p2/eval_vllm_think/` (and the transformers outputs of Meerkat-70B):

```bash
python scripts/p2/analysis_vllm.py                       # Instant and Think roots; p2_analysis, strata, recovery, correlation, pair accuracy
python scripts/p2/vllm_final.py --out runs/p2/analysis_vllm/final_tables.json --md final_tables.md
python scripts/p2/engine_agreement.py --out runs/p2/analysis_vllm/engine_agreement.json --md engine_agreement.md
for j in probe_strata b3 probe_pairs pre32b think7 c0; do python scripts/paper/d131_rerun_extra.py $j; done
python scripts/paper/d131_share.py
python scripts/paper/d136_conditional_use.py              # use of received facts (post hoc)
```

Statistics: 95% item-clustered paired bootstrap, B = 10,000, seed 20261016.

## Benchmark items and labels

`data/medgapqa.jsonl` has one record per item (fields in `data/README.md`): the original MedQA question (S0), the fact-removed question given to the models in S1 and S2, the two removed facts, the options and the answer, and the labels. `medqa_test_index` is the 0-based line number in `phrases_no_exclude_test.jsonl` of the Hugging Face copy of the MedQA US four-option test set (`GBaker/MedQA-USMLE-4-options`, revision `0fb93dd2`); `medqa_item_hash` is the first 16 hex digits of the sha256 of the original question, options and answer index (`src/medgapqa/data.py`, `item_hash`). The removed facts were selected and the questions rewritten by Claude Opus; all labels were produced by Claude models with no clinician review (necessity labels by Claude Sonnet, fact-type labels by Claude Opus).

## Not included

- Trajectories and run logs, because they contain StatPearls passages (CC BY-NC-ND).
- Model weights, caches and virtual environments.
