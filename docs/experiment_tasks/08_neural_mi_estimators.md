# Experiment 8: Alternative MI Estimators (MINE/InfoNCE)

## Research Question

Does the code > NL MI finding replicate with neural MI estimators?

---

## Hypothesis

Yes - the effect is real, not an artifact of linear classifier. Both MINE and InfoNCE should recover same ordering: MI_code > MI_nl.

---

## Methodology

### Three Estimators

#### 1. Baseline - Logistic Regression

I(Y; Z_r) ≥ H(Y) - CE(Y|Z_r)

#### 2. MINE (Mutual Information Neural Estimation)

- **Architecture**: Discriminator T_ψ: MLP with 2-3 hidden layers (256-512 units, ReLU)
- **Objective**: Maximize E[T_ψ(x,y)] - log(E[exp(T_ψ(x,ỹ))])
- **Training**: 100 epochs, Adam lr=1e-3, EMA smoothing
- **Citation**: Belghazi et al. (2018)

#### 3. InfoNCE (Contrastive)

- **Architecture**: Score function f_θ: MLP on concatenated embeddings
- **Objective**: N=256 negative samples per batch, cross-entropy loss
- **Training**: 100 epochs
- **Citation**: Oord et al. (2019)

### Cross-Validation Protocol

- 5-fold stratified CV on test data
- Train each estimator on 4 folds, evaluate on held-out
- Report mean ± std MI across folds

---

## Expected Outcomes

| Metric | Expected |
|--------|----------|
| Ordering | MI_code > MI_nl with all three estimators |
| Spearman ρ | > 0.7 between estimators |
| Magnitude | MINE > InfoNCE > LogReg |
| Significance | ΔMI positive, p < 0.05 |

---

## Validation Criteria

- [ ] Ordering flips in <15% of folds
- [ ] Kendall τ > 0.6 between estimators
- [ ] |ΔMI| ≥ 0.1 bits consistent sign

---

## Implementation

```python
import torch
import torch.nn as nn

class MINEDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        joint = torch.cat([x, y], dim=-1)
        return self.net(joint)

def mine_loss(discriminator, x, y):
    """MINE objective with exponential moving average."""
    joint = discriminator(x, y)

    # Shuffle y for marginal samples
    y_shuffled = y[torch.randperm(len(y))]
    marginal = discriminator(x, y_shuffled)

    return joint.mean() - torch.log(torch.exp(marginal).mean())

# Training loop
optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
for epoch in range(100):
    for x_batch, y_batch in dataloader:
        loss = -mine_loss(discriminator, x_batch, y_batch)  # Maximize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        optimizer.step()
```

---

## Dependencies

```
torch>=2.0
numpy
scipy
```

---

## Training Notes

- Gradient clipping (||∇|| ≤ 1.0) for stability
- Orthogonal weight initialization
- Batch normalization on hidden layers
- Checkpoint best-performing discriminators

---

## Estimated Time: 1-2 days
