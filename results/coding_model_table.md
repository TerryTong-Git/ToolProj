# Coding-Model Rebuttal Table

Accuracies use per-arm parse-normalized denominators: NL and Sim drop rows where that arm failed to parse; Code Exec keeps rows with `code_err_msg == ok,ok`.

| Model | NL | Sim | Code Exec |
|---|---:|---:|---:|
| x-ai/grok-code-fast-1 (25% data) | 47.71% | 47.71% | 55.99% |
| qwen/qwen3-coder (25% data) | 30.23% | 26.86% | 56.19% |
| codestral-2508 (original) | 19.89% | 23.14% | 59.65% |

## Denominators

| Model | Rows | NL correct/parse-ok | Sim correct/parse-ok | Code correct/executed |
|---|---:|---:|---:|---:|
| x-ai/grok-code-fast-1 (25% data) | 350 | 167/350 | 167/350 | 159/284 |
| qwen/qwen3-coder (25% data) | 350 | 104/344 | 94/350 | 168/299 |
| codestral-2508 (original) | 4740 | 943/4740 | 1097/4740 | 2266/3799 |

## Source Runs

- x-ai/grok-code-fast-1 (25% data): `grok_code_fast_1_quarter_b256_20260403_seed0`
- qwen/qwen3-coder (25% data): `qwen3_coder_quarter_b256_20260403_seed0`
- codestral-2508 (original): `run_20260111_052350; run_20260111_152653; run_20260112_021421`
