# Experiment 5: Token-Level Information Attribution

## Research Question

Which specific tokens and token types in code rationales carry the most information about the problem parameter label γ? We seek mechanistic insight into **why** code representations exhibit higher MI lower bounds.

---

## Hypothesis

1. **Code Structure**: Variable names, numeric literals, and function signatures will exhibit concentrated, high-magnitude attribution
2. **NL Diffusion**: Natural language will show dispersed attribution across narrative tokens (pronouns, conjunctions)
3. **Token Type Correlation**: Fine-grained parameters correlate with syntactic elements in code
4. **Gini Concentration**: Code G ≈ 0.50-0.65; NL G ≈ 0.30-0.45

---

## Methodology

### 1. TF-IDF Attribution

- Extract logistic regression coefficients as feature importance
- Map weights back to original tokens
- Aggregate by token type

### 2. Deep Transformer Attribution (BERT)

- **Integrated Gradients**: 50 interpolation steps from zero baseline
- Accumulate gradients across steps
- Aggregate subword tokens to word-level
- Validate with attention rollout

### 3. Token Classification (via regex)

**Code tokens:**
- Variables: `[a-z_][a-z0-9_]*`
- Constants: `[A-Z_][A-Z0-9_]*` or CamelCase
- Numeric literals: `\d+`, `\d+.\d+`
- Keywords: if, else, for, while, def, return
- Operators: +, -, *, /, ==, !=, <, >

**NL tokens:**
- Pronouns: he, she, it, they, we
- Articles: a, an, the
- Connectives: and, or, but, because
- Verbs: compute, assign, compare

### 4. Aggregation Metrics

- Per-token-type attribution profiles (mean/std)
- Statistical significance (t-tests, Cohen's d)
- **Gini coefficient** for concentration
- Cumulative attribution curves (top-k)

---

## Expected Outcomes

### Code Rationales

| Token Type | Attribution % |
|------------|---------------|
| Numeric literals | 35-45% |
| Variable names | 25-35% |
| Keywords | 15-25% |
| Punctuation/operators | <10% |

### NL Rationales

| Token Type | Attribution % |
|------------|---------------|
| Pronouns & connectives | 30-40% |
| Action verbs | 20-30% |
| Numeric mentions | 15-25% |
| Articles/prepositions | 10-20% |

### Cross-Representation Delta

- Code vs NL numeric literals: +15-25 percentage points
- Gini coefficient: code G significantly > NL G (d > 0.8)

---

## Validation Criteria

1. [ ] Attribution heatmaps on 40 sample rationales (visual inspection)
2. [ ] Top-K curves: code reaches 60-70% mass in top-20; NL <50%
3. [ ] Problem-specific profiles verify semantic plausibility
4. [ ] Statistical significance with Bonferroni correction
5. [ ] Cross-method correlation r > 0.6 (IG vs coefficients)

---

## Implementation

```python
from captum.attr import IntegratedGradients

def compute_integrated_gradients(model, classifier, tokenized_input, target_class, steps=50):
    embeddings = model.embedding(tokenized_input)
    baseline = torch.zeros_like(embeddings)

    accumulated_grads = torch.zeros_like(embeddings)
    for step in range(steps):
        alpha = step / steps
        interpolated = baseline + alpha * (embeddings - baseline)
        interpolated.requires_grad = True

        hidden = model.encoder(interpolated)
        pooled = pool(hidden)
        logits = classifier(pooled)
        loss = logits[:, target_class].sum()

        grad = torch.autograd.grad(loss, interpolated)[0]
        accumulated_grads += grad

    ig = (embeddings - baseline) * (accumulated_grads / steps)
    return aggregate_subword_attributions(ig, tokenized_input)
```

---

## Dependencies

```
captum>=0.5.0
torch>=2.0
sklearn>=1.0
matplotlib>=3.5
seaborn>=0.12
```

---

## Estimated Time: 1-2 days
