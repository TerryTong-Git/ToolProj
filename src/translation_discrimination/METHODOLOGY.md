# Source-Label Discrimination Methodology

## Motivation

The pairwise SAME/DIFFERENT approach has limitations:
- Requires comparing two texts (doubles the input)
- Conflates "same source type" with "same author"
- Indirectly measures what we care about

**This approach directly estimates distinguishability between distributions.**

## Theoretical Foundation

We want to test whether:
```
p_NL(z|x) ≈ p_translated(z|x)
```

The cleanest test: for each task x, present the judge with (x, z) and ask:
> "Did this trace come from the native-NL channel or the translated-NL channel?"

This directly estimates the **total variation distance** between the joint distributions.

## Dataset Construction

**Balanced binary classification:**

| Label | Source | Description |
|-------|--------|-------------|
| 0 | Native NL | z ~ p_NL(·\|x) - original NL explanation |
| 1 | Translated | z ~ p_translated(·\|x) - code-to-NL translation |

Each sample contributes two trials:
1. (question, original_NL) → label 0
2. (question, translated_NL) → label 1

## Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                      SOURCE CLASSIFIER                          │
│                                                                 │
│   Input: (question, explanation_trace)                          │
│   Output: NATIVE (0) or TRANSLATED (1)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────────┐
│      MAIN TRIALS        │     │      POSITIVE CONTROLS          │
│                         │     │                                 │
│  Native NL (label 0)    │     │  Native NL (label 0)            │
│         or              │     │         vs                      │
│  Translated (label 1)   │     │  Raw Code (label 1)             │
│                         │     │                                 │
│  → Accuracy near 50%    │     │  → Accuracy >> 50%              │
│    means indistinguish- │     │    means judge has power        │
│    able                 │     │                                 │
└─────────────────────────┘     └─────────────────────────────────┘
```

## Positive Controls

**Purpose:** Verify the judge can actually discriminate when differences exist.

**Design:** Present NL explanation vs raw code (obviously different)
- If judge accuracy >> 50% on controls → judge has discriminative power
- If judge accuracy ≈ 50% on controls → judge is broken/task too hard

**Interpretation requires both:**
1. Controls succeed (judge works)
2. Main test accuracy ≈ 50% (indistinguishable)

## Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Accuracy** | correct / total | ≈ 50% for indistinguishability |
| **95% CI** | Wilson score interval | Should contain 50% |
| **Control accuracy** | control_correct / control_total | > 70% (judge has power) |

## Interpretation Matrix

| Control Accuracy | Main Accuracy | Interpretation |
|------------------|---------------|----------------|
| > 70% | CI contains 50% | **STRONG EVIDENCE** for indistinguishability |
| > 70% | CI excludes 50% | Distributions ARE distinguishable |
| ≤ 70% | Any | **INCONCLUSIVE** - judge lacks power |

## Why This Is Better

| Aspect | Pairwise (old) | Source-Label (new) |
|--------|----------------|---------------------|
| Input complexity | Two texts | One text |
| What it measures | Stylistic similarity | Distribution distinguishability |
| Calibration | Needs control pairs | Built-in via random chance |
| Theoretical alignment | Indirect | Direct (matches TV assumption) |
| Interpretation | Adjusted score | Accuracy vs 50% |

## Usage

```bash
uv run python -m src.translation_discrimination.cli \
    --n_samples 200 \
    --kinds "add,sub,mul,knap,lcs" \
    --concurrency 32
```

## Expected Output

```
SOURCE DISCRIMINATION RESULTS
============================================================

MAIN TEST (native vs translated):
  Accuracy: 52.3% [48.1%, 56.5%]   # CI contains 50% ✓
  N trials: 400

POSITIVE CONTROLS (NL vs raw code):
  Accuracy: 89.2% (n=100)          # >> 50% ✓
  Judge has power: YES

INTERPRETATION:
  ✓ STRONG EVIDENCE: Judge has power but cannot distinguish
    → Supports Assumption 2 (indistinguishability)
```

## Files

| File | Purpose |
|------|---------|
| `run_source_discrimination.py` | Main experiment |
| `prompts/source_classifier.md` | Classification prompt |
| `results/*.json` | Experiment outputs |
