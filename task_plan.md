# Task Plan: Synchronized Logistic Regression Pipeline

## Status: `complete`

## Goal
Create a synchronized pipeline that runs logistic regression experiments on all exps_performance results and generates figures.

---

## Phases

### Phase 1: Discovery & Analysis
**Status:** `complete`

- [x] Explore exps_logistic experiment structure
- [x] Explore exps_performance results structure
- [x] Identify synchronization requirements
- [x] Document data flow between experiments

**Findings:**
- 24 model-seed combinations available in exps_performance
- Current prod_logistic.sh only covers 6 models
- generate_plots.py requires TARGET_RUN_DATES to be set

### Phase 2: Design
**Status:** `complete`

- [x] Design sync_and_generate.sh script
- [x] Define parameters for CPU execution
- [x] Plan dynamic model discovery

**Decisions:**
- Auto-discover all models from results directory
- Use CPU with batch size 1 for memory safety
- Set TARGET_RUN_DATES dynamically to today

### Phase 3: Implementation
**Status:** `complete`

- [x] Create sync_and_generate.sh
- [x] Add --dry-run flag for testing
- [x] Add progress tracking and error reporting

**Files Created:**
- `src/exps_logistic/sync_and_generate.sh`

### Phase 4: Testing
**Status:** `pending`

- [ ] Run dry-run to verify discovery
- [ ] Run full pipeline (estimated 2.5 hours)
- [ ] Verify 48 result files created
- [ ] Verify 10 plots generated

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| None yet | - | - |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/exps_logistic/sync_and_generate.sh` | Main orchestration script |
| `src/exps_performance/results/` | Input data source |
| `src/exps_logistic/results/` | Logistic regression outputs |
| `src/exps_logistic/notebooks/generate_plots.py` | Plot generation |
