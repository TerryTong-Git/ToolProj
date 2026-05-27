# Frontier No-Patching Rebuttal Table

Accuracies use per-arm parse-normalized denominators: Route 1/NL and Route 2/Sim drop rows where that arm failed to parse; Route 3/Code keeps rows with `code_err_msg == ok,ok`.

| Model | Route 1 | Route 2 | Route 3 |
|---|---:|---:|---:|
| GPT-5.4 | 42.57% | 41.43% | 54.01% |
| Claude Opus 4.6 | 52.73% | 73.42% | 77.54% |

## Cell Provenance

| Model | Route | Correct / Denominator | Source run |
|---|---|---:|---|
| GPT-5.4 | Route 1 | 149/350 | `run_20260406_nopatch_gpt54_seed1_subset350_py310` |
| GPT-5.4 | Route 2 | 145/350 | `run_20260406_nopatch_gpt54_seed1_subset350_py310` |
| GPT-5.4 | Route 3 | 175/324 | `run_20260406_nopatch_gpt54_seed1_subset350_py310` |
| Claude Opus 4.6 | Route 1 | 174/330 | `run_20260406_nopatch_opus46_seed1_subset350_py310` |
| Claude Opus 4.6 | Route 2 | 116/158 | `run_20260406_nopatch_opus46_seed1_subset350_rerun1_py310` |
| Claude Opus 4.6 | Route 3 | 107/138 | `run_20260406_nopatch_opus46_seed1_subset350_py310` |

## Single-Run Diagnostics

| Run | Model | Rows | Route 1 | Route 2 | Route 3 |
|---|---|---:|---:|---:|---:|
| `run_20260406_nopatch_gpt54_seed1_subset350_py310` | GPT-5.4 | 350 | 42.57% | 41.43% | 54.01% |
| `run_20260406_nopatch_opus46_seed1_subset350_py310` | Claude Opus 4.6 | 350 | 52.73% | 75.16% | 77.54% |
| `run_20260406_nopatch_opus46_seed1_subset350_rerun1_py310` | Claude Opus 4.6 | 350 | 54.60% | 73.42% | 77.86% |
