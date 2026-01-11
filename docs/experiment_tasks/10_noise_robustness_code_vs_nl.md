# Experiment 10: Noise Robustness - Code vs NL Representations

## Research Question

Does code representation degrade more slowly than natural language (NL) under increasing noise injection? Is code more robust to perturbations?

---

## Hypotheses

**H1 (Primary):** Code execution accuracy degrades more slowly than NL accuracy as noise level increases.

**H2:** The code advantage (code_acc - nl_acc) increases with problem difficulty (digits).

**H3:** Code maintains higher accuracy under structural and numerical noise types specifically, due to syntactic redundancy.

---

## Experimental Design

### Independent Variables

| Variable | Values | Description |
|----------|--------|-------------|
| **Representation** | `code`, `nl` | Evaluation arm |
| **Noise Level (sigma)** | 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 | Perturbation intensity |
| **Noise Type** | numerical, gaussian, uniform, textual, structural, irrelevant | 6 orthogonal noise functions |
| **Digits** | 2, 4, 8, 16 | Problem hardness/size |

### Dependent Variable

- **Accuracy** (0-1): Correctness of answer extraction after noise injection

### Control Variables

- Model: claude-haiku-4.5 (primary), gpt-4o-mini, gemini-2.5-flash (validation)
- Temperature: 0
- Seed: 1, 2, 3 (for statistical power)
- n_samples: 10 per condition

---

## Task Categories

| Category | Kinds | Count |
|----------|-------|-------|
| **Fine-grained** | add, sub, mul, lcs, knap, rod | 6 |
| **CLRS (subset)** | binary_search, bellman_ford, dijkstra, dfs, segments_intersect | 5 |
| **NP-hard** | gcp, spp, tsp | 3 |

**Total: 14 kinds** (medium scale)

---

## Noise Functions (from noise.py)

| Type | Description | Target |
|------|-------------|--------|
| `numerical` | Replace sigma% of digits randomly | Digit values |
| `gaussian` | Digit deltas from Gaussian(0, sigma*5) | Measurement error |
| `uniform` | Digit deltas from Uniform(-9*sigma, 9*sigma) | Bounded perturbation |
| `textual` | Character insertions/deletions/swaps | Surface form |
| `structural` | Shuffle bracketed list elements | Word/element order |
| `irrelevant` | Insert distractor sentences | Semantic noise |

---

## Analysis Pipeline

### Plot 1: Accuracy vs Noise Level (X = sigma)

```
Y-axis: Average accuracy (aggregated over all noise types)
X-axis: Noise level sigma [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
Lines: code (blue), nl (orange)
Error bars: 95% CI via bootstrap
```

**Purpose:** Show degradation curves - code should have shallower slope.

### Plot 2: Accuracy vs Digits (X = problem hardness)

```
Y-axis: Average accuracy (aggregated over sigma > 0)
X-axis: Digits [2, 4, 8, 16]
Lines: code (blue), nl (orange)
Error bars: 95% CI
```

**Purpose:** Show code advantage persists/increases with difficulty.

### Statistical Tests

1. **Paired t-test** at each noise level: code_acc vs nl_acc
2. **AUC comparison**: Area under degradation curve (code vs nl)
3. **Regression slope**: fit linear model acc ~ sigma, compare slopes
4. **Cohen's d** effect size for overall code advantage

---

## Cleanup Plan

### Archive Existing Results

```bash
# Create archive directory
mkdir -p src/exps_performance/results_noise_archive/

# Move old results (too-coarse sigma levels)
mv src/exps_performance/results_noise/ src/exps_performance/results_noise_archive/results_noise_v1/
mv src/exps_performance/results_noise_fixed/ src/exps_performance/results_noise_archive/results_noise_v2/
```

### Code Modifications Needed

1. **Add `digit` field to output records** - Currently missing, needed for Plot 2
2. **Use finer sigma levels** - Current 0.25+ causes total collapse, need 0.05 increments
3. **Expand task coverage** - Add CLRS and NP-hard kinds

---

## Implementation Plan

### Phase 1: Pilot Calibration (2-4 hours)

- Run with finer sigma: [0.0, 0.05, 0.10, 0.15, 0.20]
- Single model (claude-haiku-4.5)
- Subset of kinds: add, sub, binary_search
- Validate degradation curve is captured

### Phase 2: Full Experiment (12-24 hours)

- All 6 noise types
- All 7 sigma levels
- All 14 kinds
- 3 seeds for statistical power
- Primary model: claude-haiku-4.5

### Phase 3: Validation (4-6 hours)

- Run subset on gpt-4o-mini and gemini-2.5-flash
- Confirm effect generalizes across models

### Phase 4: Analysis (2-4 hours)

- Generate Plot 1 and Plot 2
- Compute statistics
- Write up findings

---

## Expected Outcomes

| Metric | Expected Result | Threshold |
|--------|-----------------|-----------|
| Code AUC > NL AUC | Yes | p < 0.05 |
| Slope ratio (code/nl) | < 0.7 | - |
| Cohen's d | > 0.2 (small effect) | - |
| Effect at sigma=0.15 | code_acc - nl_acc > 5% | - |

---

## Output Files

```
src/exps_performance/results_noise_v3/
  {model}_seed{seed}_noise_{timestamp}.json

src/exps_performance/figures_noise/
  accuracy_vs_noise_level.png    # Plot 1
  accuracy_vs_digits_under_noise.png  # Plot 2
  degradation_curves_by_type.png  # Supplementary
  code_advantage_heatmap.png      # Supplementary
```

---

## Success Criteria

- [ ] Code accuracy degrades slower than NL (shallower slope)
- [ ] Code maintains > 5% advantage at moderate noise (sigma=0.15)
- [ ] Effect holds across at least 2 noise types
- [ ] Results reproduce across 3 seeds
- [ ] Effect generalizes to at least 1 validation model

---

## Estimated Timeline

| Phase | Duration |
|-------|----------|
| Pilot | 2-4 hours |
| Full experiment | 12-24 hours |
| Validation | 4-6 hours |
| Analysis | 2-4 hours |
| **Total** | **~2 days** |

---

## Notes from De-risking

- Current sigma levels (0.25+) are too coarse - cause complete accuracy collapse
- Need finer granularity in 0.05-0.20 range to capture degradation curves
- Code execution already shows higher baseline accuracy in clean condition
- Infrastructure exists in `noise.py` - just need runner modifications
