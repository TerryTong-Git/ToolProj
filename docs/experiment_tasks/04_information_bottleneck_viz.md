# Experiment 4: Information Bottleneck Visualization

## Research Question

Are code embedding clusters more separable by γ label than NL embeddings?

---

## Hypothesis

Code embeddings show tighter, more distinct clusters per γ. Higher silhouette scores, lower Davies-Bouldin indices. Visual evidence of non-overlapping clusters.

---

## Methodology

### 1. Sample Selection

- Paired samples: same (kind, digits) under both code and NL
- Target: 1,000-5,000 samples per representation
- γ labels as clustering ground truth

### 2. Embedding Extraction

- Sentence-Transformers (all-MiniLM-L6-v2) with mean pooling
- Normalize to unit L2 norm for fair comparison

### 3. Dimensionality Reduction

| Method | Hyperparameters | Purpose |
|--------|-----------------|---------|
| t-SNE | perplexity=30, n_iter=1000, seed=42 | Local structure |
| UMAP | n_neighbors=15, min_dist=0.1, seed=42 | Global topology |

Apply identical settings to both representations.

### 4. Cluster Quality Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Silhouette Score | s(i) = (b(i) - a(i)) / max(a(i), b(i)) | Range [-1, 1], higher = better |
| Davies-Bouldin Index | Lower = better separation | Values <1.0 ideal |

### 5. Visualization

- Side-by-side 2D scatter plots colored by γ label
- Overlay cluster centroids
- Compute convex hulls per γ label

### 6. Statistical Validation

- 5-fold cross-validation on cluster metrics
- Paired t-test on metrics across folds

---

## Expected Outcomes

| Metric | Code | NL |
|--------|------|-----|
| Silhouette Score | ≥ 0.35 | ≤ 0.25 |
| Davies-Bouldin Index | ≤ 0.8 | ≥ 1.2 |
| Inter-cluster overlap | 20-40% lower | baseline |

---

## Validation Criteria

- [ ] **Quantitative**: Code outperforms on both metrics, p < 0.05
- [ ] **Qualitative**: Human reviewers rate code as "clearly separable" ≥80%
- [ ] **Reproducibility**: CV < 15% across random seeds

---

## Implementation

```python
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap

# Extract embeddings
embeddings_code = featurizer.transform(code_rationales)
embeddings_nl = featurizer.transform(nl_rationales)

# Dimensionality reduction
tsne_code = TSNE(perplexity=30, n_iter=1000, random_state=42).fit_transform(embeddings_code)
tsne_nl = TSNE(perplexity=30, n_iter=1000, random_state=42).fit_transform(embeddings_nl)

# Cluster quality
sil_code = silhouette_score(embeddings_code, labels)
sil_nl = silhouette_score(embeddings_nl, labels)
db_code = davies_bouldin_score(embeddings_code, labels)
db_nl = davies_bouldin_score(embeddings_nl, labels)

print(f"Silhouette: Code={sil_code:.3f}, NL={sil_nl:.3f}")
print(f"Davies-Bouldin: Code={db_code:.3f}, NL={db_nl:.3f}")
```

---

## Output Artifacts

- `visualization_tsne_comparison.png` - Side-by-side t-SNE plots
- `visualization_umap_comparison.png` - Side-by-side UMAP plots
- `cluster_metrics.json` - Quantitative scores

---

## Estimated Time: 4 hours
