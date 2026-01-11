# Experiment 7: Token Normalization Control (Length Confound)

## Research Question

Does code's MI advantage persist after controlling for length as a confounding variable?

---

## Hypothesis

**Primary (H1)**: Code maintains a statistically significant MI advantage **per token** over NL, even after normalizing by token count. This indicates advantage derives from structure, not raw length.

**Secondary (H2)**: The MI-vs-length relationship differs between representations (code slopes steeper), suggesting code is more efficient encoding.

---

## Methodology

### Three Complementary Methods

| Method | Approach | Strength | Limitation |
|--------|----------|----------|------------|
| A: MI/Token | Normalize by length | Simple; no resampling | Assumes linear scaling |
| B: Truncation | Match code to NL lengths | Concrete; interpretable | May discard info |
| C: PSM | Propensity score matching | Principled causal inference | Reduces sample size |

### Method A: MI per Token

1. Compute token counts using `transformers.AutoTokenizer`
2. Normalize: `mi_per_token = MI / token_count`
3. Compare distributions with paired t-test
4. Report mean MI/token ± SE, Cohen's d, 95% CI

### Method B: Truncation

1. For each stratum, compute median NL token count
2. Truncate code to median length (keep first N tokens)
3. Re-featurize truncated rationales
4. Re-evaluate classifier (no retraining)
5. Paired Wilcoxon signed-rank test on ΔMI_matched

### Method C: Propensity Score Matching

1. Logistic regression: Pr(code | token_count, kind, digits)
2. 1:1 nearest-neighbor matching, caliper < 0.1 × SD
3. Balance check: SMD < 0.1 for key covariates
4. Paired t-test on matched cohort

---

## Expected Outcomes

| Method | Code | NL |
|--------|------|-----|
| MI per Token (A) | 0.08-0.12 bits/token | 0.05-0.08 bits/token |
| Length-Matched (B) | 85-95% of original MI | ~100% |
| PSM (C) | Positive ΔMI, p < 0.05 | baseline |

**Cross-Method Consistency**: All three methods agree on direction (code > NL)

---

## Validation Criteria

### Statistical

- [ ] 95% CI for ΔMI/token
- [ ] Bonferroni correction if multiple kinds tested
- [ ] Minimum n ≥ 30 per representation per model

### Visual

- [ ] Length vs MI scatter plot with regression lines
- [ ] Density plots of token_count and MI/token by rep
- [ ] Residual plot for Method B
- [ ] Propensity score distribution for Method C

---

## Interpretation Guide

| Outcome | Conclusion |
|---------|------------|
| All three methods confirm code > NL | Length is NOT a confounder |
| Only Method A shows code > NL | Length IS the driver (spurious) |
| Mixed results by kind | Some kinds compress well; others don't |

---

## Implementation

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def compute_token_counts(texts):
    return [len(tokenizer.encode(t)) for t in texts]

# Method A
token_counts = compute_token_counts(rationales)
mi_per_token = mi_estimates / np.array(token_counts)

# Method B
median_length = np.median(nl_token_counts)
truncated_code = [tokenizer.decode(tokenizer.encode(t)[:median_length]) for t in code_rationales]

# Method C
from sklearn.neighbors import NearestNeighbors
# ... propensity score matching logic
```

---

## Output Files

- `length_confound_summary.json`
- `length_confound_plots.pdf`
- `matched_cohorts.csv`

---

## Estimated Time: 2 hours
