# Experiment 6: Noise Injection Robustness Study

## Research Question

Is code's MI advantage robust to noise injection?

---

## Hypothesis

Code degrades more slowly because structure provides redundancy. Shallower degradation curve than NL. σ_cross (crossing point where code MI = NL MI) should be > 0.15.

---

## Methodology

### 1. Noise Types (5 orthogonal functions)

| Noise Type | Description | Target |
|------------|-------------|--------|
| Textual | Random insertions/deletions/swaps | ~σ% of chars |
| Structural | Shuffle bracketed list elements | Word order |
| Numerical | Replace σ% of digits uniformly | Digit values |
| Irrelevant | Insert σ% distractor sentences | Semantic noise |
| Gaussian | Digit deltas from Gaussian(0, σ*5) | Measurement error |

### 2. Noise Intensity Levels

σ = {0.0, 0.05, 0.10, 0.20, 0.30}

- σ = 0.0: Clean baseline
- σ = 0.05-0.10: Mild/moderate noise
- σ = 0.20-0.30: Severe noise

### 3. Pipeline per (noise_type, σ)

```
1. Load rationales from performance results
2. Apply noise: corrupted = perturb(rationale, type, σ, seed=42)
3. Extract embeddings on corrupted text
4. Run MI estimation
5. Store: {noise_type, sigma, rep, ce_loss, accuracy}
```

### 4. Degradation Analysis

- Plot MI vs σ for code and NL
- Fit degradation curves (linear, polynomial, exponential)
- Compute **Area Under Degradation Curve (AUDC)**

### 5. Crossing Point Analysis

- σ_cross: noise level where code MI = NL MI
- If σ_cross > 0.30, code advantage is robust across test range

---

## Expected Outcomes

| Metric | Expected Value |
|--------|----------------|
| Code slope | ≤ 0.5 × NL slope |
| σ_cross | ≥ 0.15 |
| AUDC differential | code - NL > 0.02 bits |

---

## Validation Criteria

- [ ] Reproducibility: ±2% across re-runs (seed=42)
- [ ] 95% CI via bootstrap (n=1000)
- [ ] Code advantage persists across ≥2 noise types
- [ ] Minimum 5 noise levels for reliable curve fitting

---

## Implementation

```python
from src.exps_performance.noise import perturb_text

noise_types = ['textual', 'structural', 'numerical', 'irrelevant', 'gaussian']
sigma_levels = [0.0, 0.05, 0.10, 0.20, 0.30]

results = []
for noise_type in noise_types:
    for sigma in sigma_levels:
        for rep in ['code', 'nl']:
            # Load rationales
            rationales = load_rationales(rep)

            # Apply noise
            corrupted = [perturb_text(r, noise_type, sigma, seed=42) for r in rationales]

            # Extract embeddings and compute MI
            embeddings = featurizer.transform(corrupted)
            mi_estimate = compute_mi_lower_bound(embeddings, labels)

            results.append({
                'noise_type': noise_type,
                'sigma': sigma,
                'rep': rep,
                'mi': mi_estimate
            })

# Plot degradation curves
plot_degradation_curves(results)
```

---

## Output Files

- `results_noise/{model}_{seed}_{rep}_{noise_type}_σ{sigma}.json`
- `degradation_curves.png`
- `crossing_point_analysis.json`

---

## Estimated Time: 1 day (infrastructure exists in noise.py)
