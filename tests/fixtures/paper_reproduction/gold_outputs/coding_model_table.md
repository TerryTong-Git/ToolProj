# Coding-Model Rebuttal Table

Accuracies use per-arm parse-normalized denominators: NL and Sim drop rows where that arm failed to parse; Code Exec keeps rows with `code_err_msg == ok,ok`.

| Model | NL | Sim | Code Exec |
|---|---:|---:|---:|
| x-ai/grok-code-fast-1 (25% data) | 49.45% | 65.93% | 82.42% |
| qwen/qwen3-coder (25% data) | 32.97% | 49.45% | 65.93% |
| codestral-2508 (original) | 16.67% | 33.33% | 50.00% |

## Denominators

| Model | Rows | NL correct/parse-ok | Sim correct/parse-ok | Code correct/executed |
|---|---:|---:|---:|---:|
| x-ai/grok-code-fast-1 (25% data) | 91 | 45/91 | 60/91 | 75/91 |
| qwen/qwen3-coder (25% data) | 91 | 30/91 | 45/91 | 60/91 |
| codestral-2508 (original) | 90 | 15/90 | 30/90 | 45/90 |

## Source Runs

- x-ai/grok-code-fast-1 (25% data): `grok`
- qwen/qwen3-coder (25% data): `qwen`
- codestral-2508 (original): `codestral`
