# Experiment 9: Out-of-Distribution Generalization

## Research Question

Does code's MI advantage transfer to unseen problem families?

---

## Hypothesis

Code generalizes better because it captures abstract algorithm structure. Code shows smaller MI drop on OOD than NL.

**Predictions:**
- Code transfer ratio (OOD MI / ID MI) ≥ 0.75
- NL transfer ratio < 0.65
- Difference statistically significant (p < 0.05)

---

## Methodology

### Problem Family Taxonomy

| Family | Kinds (n) | Characteristics |
|--------|-----------|-----------------|
| **FG** | 9 | Arithmetic + DP |
| **CLRS** | 30 | Classical algorithms |
| **NPHARD** | 5 | Combinatorial optimization |

### Leave-One-Family-Out (LOFO) Protocol

| Experiment | Train | Test (OOD) |
|------------|-------|------------|
| LOFO-1 | FG + CLRS | NPHARD |
| LOFO-2 | FG + NPHARD | CLRS |
| LOFO-3 | CLRS + NPHARD | FG |

### Leave-One-Kind-Out (LOKO) Sampling

- Randomly select 5 kinds, train on remaining 39
- Average transfer ratios across 5 repeats

### Training Procedure

1. **Data stratification**: 80/20 train/val on ID via stratified sampling
2. **Featurizer training**: Fit ONLY on ID training texts
3. **Hyperparameter selection**: 5-fold CV on ID only, grid search C ∈ {0.25, 0.5, 1.0, 2.0, 4.0}
4. **Classifier training**: Fit on ID with selected hyperparameters
5. **OOD evaluation**: Apply fitted classifier (no retraining)

### Transfer Ratio Metric

$$\text{Transfer Ratio} = \frac{\text{MI}_{\text{OOD}}}{\text{MI}_{\text{ID}}}$$

---

## Expected Outcomes

| Metric | Code | NL |
|--------|------|-----|
| Transfer ratio | ≥ 0.75 | < 0.65 |
| MI drop on OOD | <25% | >35% |
| Mean TR | ≥ 0.70 | baseline |

---

## Validation Criteria

### Primary

- [ ] Paired t-test p < 0.05 for code vs NL TR difference
- [ ] Code TR > NL TR across all LOFO scenarios

### Secondary

- [ ] LOKO confirms LOFO findings
- [ ] TR > 0.5 even for furthest family pairs

### Failure Mode

- TR < 0.4 for both modalities indicates label space too fragmented

---

## Implementation

```python
def get_family_labels(kinds):
    """Map each kind to its family."""
    families = {}
    for kind in kinds:
        if kind in FG_KINDS:
            families[kind] = "fg"
        elif kind in CLRS_KINDS:
            families[kind] = "clrs"
        else:
            families[kind] = "nphard"
    return families

def lofo_split(df, held_out_family, family_map, test_size=0.2, seed=0):
    """Leave-one-family-out split."""
    # Filter OOD
    ood_mask = df["kind"].map(family_map) == held_out_family
    ood_df = df[ood_mask].reset_index(drop=True)

    # Split ID into train/val
    id_df = df[~ood_mask].reset_index(drop=True)
    id_train, id_val = stratified_split_robust(
        id_df, y_col="label", test_size=test_size, seed=seed
    )

    return id_train, id_val, ood_df
```

---

## Output Visualizations

- **Transfer ratio scatter**: LOFO-1, LOFO-2, LOFO-3 by (code, NL)
- **Box plots**: TR distributions across LOKO repeats
- **Heatmap**: TR for each (train_family, test_family) pair
- **Scatter + fit**: TR vs ID accuracy by modality

---

## Timeline

- Week 1: Implement family taxonomy and LOFO/LOKO utilities
- Week 2: Run LOFO experiments (6 models × 3 pairs × 2 reps = 36 runs)
- Week 3: LOKO validation and statistical testing
- Week 4: Visualization and interpretation

---

## Estimated Time: 1-2 days
