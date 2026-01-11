# Experiment 1: Embedding Model Comparison Ablation

## Research Question

How does the choice of embedding model affect MI lower bound estimates for code vs NL representations? Do sparse models (TF-IDF) and dense models (BERT, Sentence-Transformers, OpenAI embeddings) yield consistent conclusions about code vs NL?

---

## Hypothesis

- **H1**: Code representations will consistently show higher MI lower bounds than NL across all embedding models
- **H2**: Absolute MI magnitude will vary substantially by model
- **H3**: Smaller efficient models (Sentence-Transformers) will show comparable ordinal rankings to larger models

---

## Methodology

### A. Embedding Model Selection (4 models)

| Model | Type | Dimensions | Notes |
|-------|------|------------|-------|
| TfidfFeaturizer | Sparse | ~200K | Word n-grams (1-2), char n-grams (3-5) |
| HFCLSFeaturizer (BERT-base) | Dense | 768 | Mean-pooled transformer encoder |
| SentenceTransformersFeaturizer (all-MiniLM-L6-v2) | Dense | 384 | Optimized for semantic similarity |
| OpenAIEmbeddingFeaturizer (text-embedding-3-small) | Dense | 512 | Proprietary state-of-the-art |

### B. Fair Comparison Protocol

- Fixed multinomial logistic regression (L2, SAGA solver, max_iter=400)
- Fixed hyperparameter grid: C ∈ {0.25, 0.5, 1.0, 2.0, 4.0}
- Fixed 5-fold stratified CV for tuning
- Fixed 80/20 train-test split with seed=0
- Same gamma labels across all models

### C. Statistical Analysis

- 95% CI via bootstrap (1000 resamples)
- Spearman rank correlation across model pairs
- Paired t-tests on per-seed MI differences (Bonferroni α=0.01)
- Mixed-effects ANOVA

---

## Expected Outcomes

- Code MI > NL MI for 3/4+ models with consistent effect size (>0.05 bits)
- Spearman ρ > 0.8 between model pairs
- TF-IDF shows weaker discriminability but maintains code > NL ordering

---

## Validation Criteria

- [ ] Verify embedding tensor shapes match expected dimensions
- [ ] Confirm CV splits are deterministic
- [ ] Verify 0 ≤ I ≤ H(Y) and cross-entropy ≥ 0
- [ ] Spot-check hyperparameter choices

---

## Implementation

```bash
for feats in tfidf hf-cls st openai; do
  uv run python src/exps_logistic/main.py \
    --feats "$feats" \
    --embed-model {model} \
    --kinds-preset extended \
    --label gamma \
    --bits
done
```

---

## Estimated Time: 2 hours per model (~8 hours total)
