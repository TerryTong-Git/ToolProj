# Experiment 2: MI Gap vs Problem Difficulty Scaling

## Research Question

Does the MI advantage of code over NL increase systematically with problem difficulty τ?

---

## Hypothesis

Positive relationship between problem difficulty and MI gap. Harder problems encode more information; code's precision preserves this better than NL. At low τ, even NL may suffice; at high τ, code's compressibility provides stronger advantage.

**Prediction**: ΔMI(τ) = MI_code(τ) - MI_nl(τ) increases monotonically with τ.

---

## Methodology

### 1. Aggregation by Difficulty Level

- Group results by (kind, digits τ) pairs
- Compute average MI per group across models/seeds
- Exclude groups with <3 samples

### 2. MI Gap Computation

- ΔMI(τ) = mi_code - mi_nl per (kind, τ)
- Z-score normalize within each kind to remove kind-specific offsets

### 3. Regression Analysis

| Model | Formula | Purpose |
|-------|---------|---------|
| Linear | gap ~ τ | Test monotonic relationship |
| Polynomial | gap ~ τ + τ² | Test acceleration |
| Robust (Huber) | gap ~ τ, ε=1.35 | Down-weight outliers |

- Report R², slope, p-value, 95% CI
- Bootstrap CIs (10,000 replicates)

### 4. Significance Testing

- Shapiro-Wilk on residuals (normality check)
- Permutation test (10,000 shuffles) if normality violated
- Bonferroni correction for multiple kinds

---

## Expected Outcomes

- Positive slope (β₁ > 0)
- p-value < 0.05
- R² > 0.15
- Gap ranges from near-zero at τ=1-2 to +0.5-1.5 bits at τ=8-10

---

## Validation Criteria

- [ ] Scatter plot (gap vs τ) with OLS fit, 95% CI band
- [ ] Box plots by τ quartiles showing increasing median
- [ ] Q-Q plot and scale-location plot for residuals
- [ ] Per-kind faceted regression

---

## Implementation

```python
import pandas as pd
from scipy.stats import shapiro, spearmanr, permutation_test
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM

# Aggregate by difficulty
grouped = results_df.groupby(['kind', 'digits', 'rep'])['mi_lower_bound'].mean()
pivot = grouped.unstack('rep')
pivot['gap'] = pivot['code'] - pivot['nl']

# Regression
X = sm.add_constant(pivot.index.get_level_values('digits'))
y = pivot['gap']
model = sm.OLS(y, X).fit()
print(model.summary())
```

---

## Estimated Time: 1 hour (re-aggregate existing data)
