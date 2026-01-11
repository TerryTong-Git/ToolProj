# SWE Task: Fix Results Data Quality Issues

**Created**: 2026-01-10
**Priority**: High
**Status**: Open

## Problem Summary

Data validation of `src/exps_performance/results/final/` revealed significant quality issues that compromise the reliability of experimental results.

## Issues to Fix

### 1. Re-run Failed Experiments (Critical)

The following model/seed combinations have 0% accuracy and need to be re-run:

| Model | Seed | Issue |
|-------|------|-------|
| gemini-2.5-flash | 0 | 0% NL and Code accuracy |
| gemini-2.5-flash | 2 | 0% NL and Code accuracy |
| llama-3.1-405b-instruct | 2 | 0% across ALL metrics |
| qwen-2.5-coder-32b-instruct | 2 | 0% NL and Code accuracy |
| ministral-14b-2512 | 0 | 0% NL accuracy (partial failure) |

**Action Items**:
- [ ] Investigate why these runs failed (check logs if available)
- [ ] Re-run experiments with same configuration
- [ ] Validate new results before replacing old data

### 2. Standardize Dataset Sizes (High)

Dataset sizes vary across seeds for the same model:

| Model | Expected Size | Actual Sizes |
|-------|---------------|--------------|
| claude-haiku-4.5 | 1420 | 1420, **1786**, 1420 |
| gemini-2.5-flash | ? | 1216, 1637, 1280 |
| gpt-4o-mini | 1420 | 1420, **1637**, 1420 |
| llama-3.1-405b-instruct | 1518 | 1518, 1518, **1216** |
| qwen-2.5-coder-32b-instruct | 1518 | 1518, **1582**, **1216** |

**Root Cause**: NP-hard kinds (edp, gcp, ksp, spp, tsp) missing from some seeds.

**Action Items**:
- [ ] Determine canonical dataset size and composition
- [ ] Identify which kinds should be included in all runs
- [ ] Create a validation script that checks size consistency before accepting results

### 3. Investigate Parse Error Anomalies (Medium)

Some seeds have unusually high parse error rates:

- claude-haiku-4.5 (seeds 0,2): 100% code parse errors
- gemini-2.5-flash_seed1: 72% NL parse errors
- gemini-2.5-flash_seed2: 95% sim parse errors
- All gpt-4o-mini seeds: 100% code parse errors

**Action Items**:
- [ ] Determine if 100% code parse errors is expected (model can't generate valid code?)
- [ ] Check if parse error rates are consistent with model capabilities
- [ ] Document expected parse error baselines for each model

### 4. Create Data Validation Pipeline (Medium)

Prevent future data quality issues by adding validation.

**Action Items**:
- [ ] Create `validate_results.py` script that checks:
  - Dataset size consistency across seeds
  - No zero-accuracy runs
  - Parse error rates within expected bounds
  - All expected kinds present
- [ ] Add validation step to experiment pipeline
- [ ] Document acceptable ranges for each metric

## Validation Script Outline

```python
# src/exps_performance/scripts/validate_results.py

def validate_results(results_dir: Path) -> dict:
    """
    Validate results data quality.

    Checks:
    1. Dataset size consistency per model
    2. No zero-accuracy runs
    3. Parse error rates reasonable
    4. All expected kinds present

    Returns:
        dict with validation status and issues found
    """
    pass
```

## Files Affected

- `src/exps_performance/results/final/` - data to be fixed
- `src/exps_performance/scripts/validate_results.py` - new validation script
- `src/exps_performance/main.py` - add validation hook

## Acceptance Criteria

- [ ] All 18 model/seed combinations have consistent dataset sizes
- [ ] No zero-accuracy runs in final results
- [ ] Validation script passes for all data
- [ ] Documentation updated with data quality requirements
