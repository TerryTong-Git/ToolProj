# Session Progress: Logistic Regression Synchronization

## Session Start: 2026-01-11 17:23 PST

---

## Activity Log

### 17:23 - Session Started
- Goal: Create synchronized pipeline for logistic regression experiments

### 17:25 - Exploration Phase
- Loaded exps-logistic and exps-performance skills
- Launched 2 Explore agents in parallel
- Discovered 24 model-seed combinations in exps_performance
- Mapped data flow between experiments

### 17:35 - Design Phase
- Created initial plan file
- Identified synchronization issues:
  - prod_logistic.sh only covers 6 models (missing 18 model-seeds)
  - generate_plots.py requires TARGET_RUN_DATES
- Got user preferences: local results, CPU, all models

### 17:45 - Implementation Phase
- Created `src/exps_logistic/sync_and_generate.sh`
- Features:
  - Auto-discovers all model-seed directories
  - Runs logistic for both code and nl representations
  - Sets TARGET_RUN_DATES dynamically
  - Includes --dry-run flag for testing
  - Progress tracking and error reporting

---

## Files Created

| Time | File | Status |
|------|------|--------|
| 17:45 | `src/exps_logistic/sync_and_generate.sh` | Created |
| 17:46 | `task_plan.md` | Created |
| 17:46 | `findings.md` | Created |
| 17:46 | `progress.md` | Created |

---

## Next Steps

1. [ ] Run dry-run to verify model discovery
2. [ ] Execute full pipeline
3. [ ] Verify results and plots

---

## Session: 2026-01-14 - Algorithm Name Filtering

### Activity Log

**Started:** Algorithm name filtering implementation

- Implemented `filter_algorithm_names()` function in `data_utils.py`
- Added `--no-filter-algo-names` CLI flag in `config.py`
- Filtering applies to both NL and code arms
- Removed comments from code files
- All 69 tests passed

### Comparison Experiments Completed (09:18-09:26)

Ran 4 experiments comparing WITH vs WITHOUT filtering:
- Model: llama-3.1-405b-instruct
- Label: theta_new (79 classes for code, 36 for NL)
- Kinds: fg preset (9 kinds)

**Results:**
| Arm | Filter | MI ≥ (bits) | Accuracy |
|-----|--------|-------------|----------|
| Code | ON | 3.3420 | 46.88% |
| Code | OFF | 3.3383 | 46.88% |
| NL | ON | 2.5989 | 62.50% |
| NL | OFF | 2.6286 | 62.50% |

**Conclusion:** Algorithm name filtering has minimal impact (<0.05 bits). The MI signal comes from reasoning structure, not algorithm keywords.

---

## Commands Reference

```bash
# Dry run (verify what would be executed)
bash src/exps_logistic/sync_and_generate.sh --dry-run

# Full execution (estimated 2.5 hours on CPU)
bash src/exps_logistic/sync_and_generate.sh

# Background execution with logging
nohup bash src/exps_logistic/sync_and_generate.sh > sync_output.log 2>&1 &
```
