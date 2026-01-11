# Experiment 3: Per-Kind MI Analysis Breakdown

## Research Question

**Primary:** Which problem types benefit most from code-based CoT representations compared to natural language?

**Specific:** Do structured problems (integer linear programming, graph algorithms) show larger mutual information (MI) gaps between code and NL representations compared to simpler problems (arithmetic, DP), and how do families differ in leveraging code semantics?

---

## Hypothesis

We hypothesize a **family-level hierarchy in code representation advantage**:

| Family | Predicted ΔMI | Rationale |
|--------|---------------|-----------|
| **ILP Problems** (ilp_assign, ilp_prod, ilp_partition) | ~0.40 bits | Structured constraint specifications favor code syntax |
| **NP-hard** (TSP, GCP, KSP) | ~0.25 bits | Heuristics naturally expressible in pseudocode |
| **Graph Algorithms** (CLRS graph traversal, shortest path) | ~0.18 bits | Concise edge semantics in code |
| **Arithmetic** (add, sub, mul) | ~0.08 bits | Simple operand selection; minimal code advantage |

**Secondary hypothesis:** Within-family variance decreases as problem complexity increases.

---

## Methodology

### 3.1 Grouping and Stratification

Define problem families using frozen sets from `src/exps_logistic/config.py`:
- **FG (Fine-Grained, 9 kinds)**: add, sub, mul, lcs, knap, rod, ilp_assign, ilp_prod, ilp_partition
- **CLRS (30 algorithms)**: graph, sorting, string matching, DP, geometry algorithms
- **NPHARD (5 problems)**: edp, gcp, ksp, spp, tsp

### 3.2 MI Computation Per Kind

For each (family, kind, representation):
1. Load preprocessed results from `src/exps_logistic/results/`
2. Extract MI lower bound: I(Y; Z_r) ≥ H(Y) - CE(Y | Z_r)
3. Aggregate per-kind: MI_code(kind), MI_nl(kind), ΔMI(kind)

### 3.3 Statistical Testing

- **Kruskal-Wallis H-test** for family differences (non-parametric)
- **Post-hoc Mann-Whitney U tests** with Bonferroni correction (α=0.05)
- Expected H-statistic: 8-15 with p < 0.05

### 3.4 Effect Size Calculation

- **Cohen's d** per family: d = (μ_code - μ_nl) / σ_pooled
- Interpretation: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, ≥0.8 large
- **95% bootstrap CIs** around each family's ΔMI estimate

---

## Expected Outcomes

### Predicted Rankings (by ΔMI magnitude)

1. ILP problems: ΔMI ≈ 0.40 bits ± 0.08 (largest)
2. NP-hard: ΔMI ≈ 0.25 bits ± 0.12
3. CLRS graph/DP: ΔMI ≈ 0.18 bits ± 0.10
4. Arithmetic: ΔMI ≈ 0.08 bits ± 0.06 (smallest)

### Effect Sizes

- ILP vs Arithmetic: d ≈ 1.2 (large)
- NP-hard vs Arithmetic: d ≈ 0.65 (medium)
- CLRS vs Arithmetic: d ≈ 0.45 (small-to-medium)

---

## Validation Criteria

### Visualizations

1. **Family-Level Heatmap**: 44 kinds grouped, showing MI_code, MI_nl, ΔMI, Cohen's d
2. **Grouped Bar Chart**: Family means with non-overlapping CIs
3. **Complexity Scatter**: Complexity vs ΔMI colored by family

### Quantitative Thresholds

- [ ] Kruskal-Wallis p < 0.05
- [ ] ILP ΔMI ≥ 0.30 bits
- [ ] Arithmetic ΔMI ≤ 0.15 bits
- [ ] Between-family Cohen's d ≥ 0.5 for ≥2 pairs
- [ ] Within-family CV ≤ 0.5 for ILP, ≤ 0.75 for others

---

## Implementation

```python
from scipy.stats import kruskal, mannwhitneyu

# Group by family and kind
family_deltas = {
    fam: kind_mi_pivot[kind_mi_pivot['family'] == fam]['delta_mi'].values
    for fam in ['fg', 'clrs', 'nphard']
}

# Kruskal-Wallis test
h_stat, p_val = kruskal(*family_deltas.values())
print(f"H-statistic: {h_stat:.2f}, p-value: {p_val:.4f}")

# Post-hoc with Bonferroni
for (f1, f2) in [('fg', 'clrs'), ('fg', 'nphard'), ('clrs', 'nphard')]:
    stat, p = mannwhitneyu(family_deltas[f1], family_deltas[f2])
    print(f"{f1} vs {f2}: U={stat:.2f}, p={p:.4f}")
```

---

## Estimated Time: 1 hour (already have data)
