# Frontier No-Patching Rebuttal Table

Accuracies use per-arm parse-normalized denominators: Route 1/NL and Route 2/Sim drop rows where that arm failed to parse; Route 3/Code keeps rows with `code_err_msg == ok,ok`.

| Model | Route 1 | Route 2 | Route 3 |
|---|---:|---:|---:|
| GPT-5.4 | 50.00% | 66.67% | 83.33% |
| Claude Opus 4.6 | 58.82% | 72.22% | 82.35% |

## Cell Provenance

| Model | Route | Correct / Denominator | Source run |
|---|---|---:|---|
| GPT-5.4 | Route 1 | 9/18 | `run_20260406_nopatch_gpt54_seed1_subset350_py310` |
| GPT-5.4 | Route 2 | 12/18 | `run_20260406_nopatch_gpt54_seed1_subset350_py310` |
| GPT-5.4 | Route 3 | 15/18 | `run_20260406_nopatch_gpt54_seed1_subset350_py310` |
| Claude Opus 4.6 | Route 1 | 10/17 | `run_20260406_nopatch_opus46_seed1_subset350_py310` |
| Claude Opus 4.6 | Route 2 | 13/18 | `run_20260406_nopatch_opus46_seed1_subset350_rerun1_py310` |
| Claude Opus 4.6 | Route 3 | 14/17 | `run_20260406_nopatch_opus46_seed1_subset350_py310` |

## Single-Run Diagnostics

| Run | Model | Rows | Route 1 | Route 2 | Route 3 |
|---|---|---:|---:|---:|---:|
| `run_20260406_nopatch_gpt54_seed1_subset350_py310` | GPT-5.4 | 18 | 50.00% | 66.67% | 83.33% |
| `run_20260406_nopatch_opus46_seed1_subset350_py310` | Claude Opus 4.6 | 17 | 58.82% | 52.94% | 82.35% |
| `run_20260406_nopatch_opus46_seed1_subset350_rerun1_py310` | Claude Opus 4.6 | 18 | 33.33% | 72.22% | 38.89% |
