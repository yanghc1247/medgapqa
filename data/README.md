# Benchmark items

Derived from MedQA (Jin et al., 2021), released under the MIT License; the MedQA copyright and permission notice is in `LICENSE_MedQA`.
Our edits (removed facts, fact-removed questions) and labels are released under the repository `LICENSE`.

`medgapqa.jsonl`: 736 records, one per item. Fields (internal field name of `runs/v5/evalk/evalk.jsonl` in brackets):

| Field | Contents |
|---|---|
| `medqa_test_index` | 0-based line number in `phrases_no_exclude_test.jsonl` of `GBaker/MedQA-USMLE-4-options` (revision `0fb93dd2`) [`id` = `medqa_test_<index, 5 digits>`] |
| `medqa_item_hash` | first 16 hex digits of the sha256 of the original question, options and answer index (`src/medgapqa/data.py`, `item_hash`) [`hash`] |
| `question_s0` | original MedQA question (intact, S0) [`question`] |
| `question_fact_removed` | question with the two facts removed, the model input in S1 and S2 [`pert_question`] |
| `removed_facts` | the two removed facts, verbatim from the original question; the simulator can disclose them in S1 [`removed_facts`] |
| `options`, `answer_idx` | options A-D and the correct letter [`options`, `answer_idx`] |
| `necessity` | LLM label: `necessary`, `partially` (partially necessary) or `not_necessary` |
| `artifact_flag` | 1 if the labeler flagged a rewriting artifact |
| `fact_types`, `patient_can_report` | LLM labels for each removed fact, in the order of `removed_facts` |

All labels were produced by Claude models with no clinician review.
