# Functional Shot Ablation Summary

This compact rebuttal summary reports the verified translation-additivity shot ablation.

Metric definitions:
- `x`: question only
- `x + native NL`: question plus native NL reasoning
- `x + translated NL`: question plus translated natural-language reasoning produced from code
- `Delta native`: `(x + native NL) - x`
- `Delta translated`: `(x + translated NL) - x`
- `Gap`: `(x + translated NL) - (x + native NL)`

## Claude Haiku 4.5

| Shots | x | x + native NL | x + translated NL | Delta native | Delta translated | Gap |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 50.00% | 70.00% | 40.00% | +20.00pp | -10.00pp | -30.00pp |
| 10 (paper legacy) | 60.00% | 80.00% | 70.00% | +20.00pp | +10.00pp | -10.00pp |

The 10-shot row is the older paper result and is not from the same `subset_fraction=0.25` sweep as the 0-5 shot rows.
