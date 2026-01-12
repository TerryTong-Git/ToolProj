# Research Findings: Logistic Regression Synchronization

## Data Flow Discovery

### exps_performance → exps_logistic Pipeline

```
exps_performance/results/{model}_seed{N}/tb/run_*/res.jsonl
    ↓ (load_data via create_big_df)
exps_logistic/main.py
    ↓ (_convert_results_df: Record → canonical)
exps_logistic/results/{model}_seed{seed}_{rep}_{feats}-{embed}_{ts}.json
    ↓ (TARGET_RUN_DATES filtering)
generate_plots.py → 10 PNG figures
```

### Available Models (24 combinations)

| Model | Seeds | Total |
|-------|-------|-------|
| claude-haiku-4.5 | 0,1,2 | 3 |
| claude-opus-4 | 0 | 1 |
| codestral-2508 | 0,1 | 2 |
| gemini-2.0-flash-001 | 0,1 | 2 |
| gemini-2.5-flash | 0,1 | 2 |
| gpt-4o-mini | 0,1,2 | 3 |
| llama-3.1-405b-instruct | 0,1 | 2 |
| ministral-14b-2512 | 0,1,2 | 3 |
| mistral-large-2411 | 0,1 | 2 |
| mixtral-8x22b-instruct | 0,1 | 2 |
| qwen-2.5-coder-32b-instruct | 0,1 | 2 |
| **Total** | | **24** |

### res.jsonl Schema (Key Fields)

| Field | Description | Used By |
|-------|-------------|---------|
| `model` | Full model path | Filtering |
| `seed` | Random seed | Filtering |
| `digit` | Problem difficulty | Labels |
| `kind` | Problem type (44 types) | Labels |
| `nl_reasoning` | NL chain-of-thought | Features (nl rep) |
| `sim_code` | Generated code | Features (code rep) |
| `nl_correct` | NL correctness | Accuracy analysis |
| `code_correct` | Code correctness | Accuracy analysis |

### Gamma Label Format

```
{kind}|d{digits}|b{bin_id}
```

Example: `knap|d8|b24` = Knapsack problem, 8 digits, bin 24

### Synchronization Gap

**Problem:** Current `prod_logistic.sh` only runs 6 models × 3 seeds = 18 experiments, but 24 model-seed combinations exist.

**Solution:** New `sync_and_generate.sh` auto-discovers all models.

---

## Key Insights

1. **Date Filtering Critical:** `generate_plots.py` uses `TARGET_RUN_DATES` environment variable to filter which logistic results to include in plots. Must be set to today's date pattern.

2. **CPU Batch Size:** With `--device cpu`, batch size should be 1 for BERT embeddings to avoid memory issues.

3. **Extended Kinds:** 44 total problem kinds across fine-grained (9), CLRS (30), and NP-hard (5) categories.

4. **Expected Output:** 48 experiments (24 model-seeds × 2 reps) → 10 figures
