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

---

## Algorithm Name Filtering (2026-01-14)

### Problem
CoT text may contain algorithm names (e.g., "dijkstra", "quicksort") that allow the classifier to trivially predict labels, inflating MI estimates.

### Solution
Filter all 44 EXTENDED_KINDS names from CoT text (case-insensitive) before featurization.

### Implementation
```python
# In data_utils.py
def filter_algorithm_names(text: str, algorithm_names: Set[str]) -> str:
    for name in algorithm_names:
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        text = pattern.sub("", text)
    return text
```

### Filtered Names (44 total)
- **FG_KINDS (9):** add, sub, mul, lcs, knap, rod, ilp_assign, ilp_prod, ilp_partition
- **CLRS_KINDS (30):** activity_selector, articulation_points, bellman_ford, bfs, binary_search, bridges, bubble_sort, dag_shortest_paths, dfs, dijkstra, find_maximum_subarray_kadane, floyd_warshall, graham_scan, heapsort, insertion_sort, jarvis_march, kmp_matcher, lcs_length, matrix_chain_order, minimum, mst_kruskal, mst_prim, naive_string_matcher, optimal_bst, quickselect, quicksort, segments_intersect, strongly_connected_components, task_scheduling, topological_sort
- **NPHARD_KINDS (5):** edp, gcp, ksp, spp, tsp

### Comparison Results (llama-3.1-405b-instruct, fg preset, theta_new label)

**Code Arm:**
| Metric | WITH Filtering | WITHOUT Filtering | Δ |
|--------|---------------|------------------|---|
| Features | 27,184 | 27,422 | -238 |
| Cross-entropy | 2.7727 bits | 2.7764 bits | -0.0037 |
| Accuracy | 46.88% | 46.88% | 0 |
| MI ≥ | **3.3420 bits** | 3.3383 bits | **+0.0037** |

**NL Arm:**
| Metric | WITH Filtering | WITHOUT Filtering | Δ |
|--------|---------------|------------------|---|
| Features | 44,916 | 45,118 | -202 |
| Cross-entropy | 2.5143 bits | 2.4847 bits | +0.0296 |
| Accuracy | 62.50% | 62.50% | 0 |
| MI ≥ | 2.5989 bits | **2.6286 bits** | **-0.0297** |

### Key Findings (Algorithm Names)

1. **Minimal Impact:** Algorithm name filtering has <0.05 bits impact on MI estimates
2. **Mixed Direction:** Code arm shows slight increase with filtering; NL arm shows slight decrease
3. **Feature Count:** ~200-240 fewer TF-IDF features with filtering (algorithm name tokens removed)
4. **Accuracy Unchanged:** Classification accuracy identical with/without filtering
5. **Conclusion:** Algorithm names were not the primary predictive signal; CoT contains richer reasoning structure

---

## Comment Filtering (2026-01-14)

### Problem
Code comments in CoT (e.g., `# compute sum`, `"""docstring"""`) could leak algorithm information.

### Solution
Added `strip_comments()` function that removes:
- Triple-quoted docstrings (`"""..."""` and `'''...'''`)
- Single-line `# comments`
- Inline `# comments` after code

### Comparison Results (llama-3.1-405b-instruct, fg preset, code arm, theta_new label)

| Metric | WITH Filtering | WITHOUT Filtering | Δ |
|--------|---------------|------------------|---|
| Features | 20,468 | 27,184 | **-6,716 (-25%)** |
| Cross-entropy | 2.7746 bits | 2.7725 bits | +0.0021 |
| Accuracy | **48.44%** | 46.88% | **+1.56%** |
| MI ≥ | 3.3400 bits | 3.3422 bits | -0.0022 |

### Key Findings (Comments)

1. **Significant Feature Reduction:** Comment filtering removes ~25% of TF-IDF features (6,716 tokens)
2. **MI Unchanged:** Despite removing 25% of features, MI estimate is essentially identical (-0.002 bits)
3. **Accuracy Improved:** Slightly better accuracy with filtering (+1.56%), suggesting comments added noise
4. **Conclusion:** Comments were not predictive; removing them reduces noise without losing signal
