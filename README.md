# MedGapQA

Benchmark items, code, prompts, labels and aggregated results for "Ignorance Is Not Bliss: Do Language Models Ask for Missing Clinical Information?" (Heechul Yang and Jaewoo Kang, Korea University; corresponding author: Jaewoo Kang, kangj@korea.ac.kr).

MedGapQA (Medical information-Gap QA) has 736 MedQA questions. In each, an LLM author removed two decisive patient facts, which can be recovered only by asking a patient/chart simulator. Each item comes in three states: intact (S0), fact-removed with the facts available on request (S1), and fact-removed with the facts withheld (S2). Each item is also labeled for whether the removed facts are necessary.

Benchmark items, code, prompts, labels and aggregated results are released here; trajectories are not, because they contain StatPearls passages (CC BY-NC-ND). The items (`data/medgapqa.jsonl`) are derived from MedQA (Jin et al., 2021), released under the MIT License; the MedQA copyright and permission notice is in `data/LICENSE_MedQA`. Our code, edits and labels are released under the repository `LICENSE` (MIT).

The supplementary material (`supplement.pdf`) is provided in this repository only; it is not uploaded to the submission system (EasyChair) with the paper.

## Results in brief

Numbers are copied from the paper; intervals are 95% item-clustered paired bootstrap intervals.

- Over the 9 main-pool models, removing two facts lowers accuracy by +0.209 [+0.187, +0.231] on average (+0.124 to +0.251 per model, large models included).
- A probe asking whether the information suffices separates fact-removed from intact cases with an AUROC of only 0.64–0.77.
- In the targeted ask pipeline, 11.2%–30.7% of questions elicit a removed fact (forced inquiry: 9.2%–41.7%).
- On the main mean, the targeted ask pipeline gains +0.038 [+0.029, +0.047], forced inquiry +0.027 [+0.016, +0.038] and the free agent +0.003; neither improvement policy met the criterion fixed before the runs.
- Handing over both removed facts recovers most of the lost accuracy (recovery ratio 0.83–1.14 per model); the best mean gain is about 18% of this both-facts reference (+0.209).
- The two large models answer at once in 97.7%–99.3% of free-agent episodes but gain more than the main-pool mean when made to ask (targeted ask pipeline: +0.053 for Meerkat-70B and +0.110 for DeepSeek-V4-Flash).

## Pre-specified and exploratory analyses

Fixed before the runs: forced inquiry or the targeted ask pipeline counts as an improvement if, on the main mean, its gain over answering immediately (S1) is at least +0.050 with the lower bound of the 97.5% interval above zero, and its change in S2 accuracy is at least −0.020.
Exploratory or post hoc, with no multiplicity correction: the necessity strata, the recovery ratio by stratum, the blind asker against the targeted ask pipeline (B3), the correlation between models, and the probe AUROC by necessity stratum.
Details are in the supplementary material (`supplement.pdf`).

## Models

All evaluated models run in Instant mode (thinking off), greedy decoding. Main-pool models run in BF16.

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
| `scripts/p2/run_eval.py` | Runs every cell for one model (items written by `scripts/make_items.py`) |
| `scripts/p2/p2_*.py` | Analysis scripts that write the JSON files in `results/` |
| `scripts/` (other) | Modules that `run_eval.py` imports: hash checks (`p2/run_pilot.py`), the C0 S2 cell (`run_devk2_explore.py`), oracle prompts (`consol_requery.py`, `d56/prompts_d56.py`) and the noticing-probe prompt (`d58/prompts_d58.py`) |
| `configs/`, `metadata/` | Agent configuration, reference hashes for model-instance checks, function-word list |
| `prompts/` | Frozen manifests of the two improvement policies and `registered_prompts.md` (every registered prompt text with its sha256) |
| `data/` | `medgapqa.jsonl` (the 736 benchmark items with their labels), `README.md` (fields), `LICENSE_MedQA` (MedQA license notice) |
| `results/` | Aggregate results used in the paper (`p2_analysis.json` and the exploratory analyses) |
| `supplement.pdf` | Supplementary material |
| `tests/test_smoke.py` | CPU smoke test: compile, prompt hashes, hash guards |

Some module names reflect the project's development history; the files listed under scripts/ and src/ are the ones used for the reported runs.

Internal names in the code map to the paper as follows: S1K, S2K = S1, S2; C0 = answer only; C4 = free agent; C4f = forced inquiry; P6 = targeted ask pipeline; O2 = both removed facts handed over; O1a, O1b = one fact handed over; NP = noticing probe.

## Installation

- Python 3.13 and `pip install -r requirements.txt`; Qwen3, Meerkat2 and MedGemma need a `transformers` version that supports them.
- Inference needs CUDA GPUs. DeepSeek-V4-Flash also needs the `kernels` package and its FP8 kernel (see the comments in `src/medgapqa/p2_models.py`).
- The smoke test and the analysis scripts run on CPU.

Paths that pointed to local disks are masked as `$PROJECT_ROOT`, `$HF_CACHE`, `$DATA_ROOT` and `$HOME`; set them before running (for example `CACHE` in `src/medgapqa/p2_models.py`).

## Running the models

The run wrapper reads the items from `runs/v5/evalk/evalk.jsonl` and checks its sha256 against `ITEMS_SHA` in `scripts/p2/run_eval.py`. `python scripts/make_items.py` writes that file from `data/medgapqa.jsonl` and prints its sha256; set `ITEMS_SHA` to this value before running. The internal file of the reported runs also held authoring fields that are not used by the runs and not released (the author's fact types, recovery questions and runner-up option, and internal row ids), so its hash differs; `make_items.py` writes the author's fact types as null and row ids `MG0001`, `MG0002`, …, which appear only in the output records.

```bash
# 1. patient/chart simulator server (one GPU)
CUDA_VISIBLE_DEVICES=0 python src/medgapqa/llm_server_c.py --model c1 --port 8714 --log runs/v5/sim_c1.log
# 2. frozen manifests where the wrapper expects them
mkdir -p runs/v5/d36 runs/v5/d37
cp prompts/FROZEN_C4F_v3.json runs/v5/d36/ && cp prompts/FROZEN_P6_v1.json runs/v5/d37/
# 3. all cells for one model: C0 (S0, S1, S2), oracle, noticing probe, free agent, targeted ask, forced inquiry
CUDA_VISIBLE_DEVICES=1 python scripts/p2/run_eval.py --model qwen3-14b --sim-ports 8714
```

`run_eval.py` checks the sha256 of the registered code and prompts, the simulator `/health`, and a reference output hash of each model instance before every cell, and refuses to overwrite existing output. The BM25 index over StatPearls is rebuilt from the NCBI archive with `src/medgapqa/statpearls_chunk.py` and `src/medgapqa/build_index.py`; the corpus and index are not redistributed.

## Reproducing the results

1. `python tests/test_smoke.py` (CPU, no data needed) compiles every module and checks the registered prompt hashes and hash guards.
2. With the run outputs under `runs/p2/eval/`: `scripts/p2/p2_analysis.py`, then `p2_strata.py`, `p2_strata_recovery.py`, `p2_b3.py`, `p2_modelcorr.py`, `p2_pair_acc.py` and `p2_probe_strata.py` (usage in each docstring). They write the files in `results/`.

Statistics: 95% item-clustered paired bootstrap, B = 10,000, seed 20261016.

## Benchmark items and labels

`data/medgapqa.jsonl` has one record per item (fields in `data/README.md`): the original MedQA question (S0), the fact-removed question given to the models in S1 and S2, the two removed facts, the options and the answer, and the labels. `medqa_test_index` is the 0-based line number in `phrases_no_exclude_test.jsonl` of the Hugging Face copy of the MedQA US four-option test set (`GBaker/MedQA-USMLE-4-options`, revision `0fb93dd2`); `medqa_item_hash` is the first 16 hex digits of the sha256 of the original question, options and answer index (`src/medgapqa/data.py`, `item_hash`). The removed facts were selected and the questions rewritten by Claude Opus; all labels were produced by Claude models with no clinician review (necessity labels by Claude Sonnet, fact-type labels by Claude Opus).

## Not included

- Trajectories and run logs, because they contain StatPearls passages (CC BY-NC-ND).
- Model weights, caches and virtual environments.
