# Functional Shot Ablation Summary

This compact rebuttal summary reports the verified Claude Haiku 4.5 translation-additivity shot ablation recovered from the `ec1b99c` local artifact commit.

Metric definitions:
- `x`: question only
- `x + native NL`: question plus native NL reasoning
- `x + translated NL`: question plus translated natural-language reasoning produced from code
- `Delta native`: `(x + native NL) - x`
- `Delta translated`: `(x + translated NL) - x`
- `Gap`: `(x + translated NL) - (x + native NL)`

## Reproduction Command

Run the full 0-5 shot sweep:

```bash
rtk uv run --no-sync python src/exps_functional/run_translation_additivity.py \
  --model anthropic/claude-haiku-4.5 \
  --source_model claude-haiku-4.5 \
  --subset_fraction 0.25 \
  --n_shots <0|1|2|3|4|5> \
  --n_samples 1000000 \
  --max_per_kind 1000000 \
  --concurrency 32 \
  --seed 42
```

The helper script `src/exps_functional/scripts/run_translation_additivity_shot_ablation.sh` loops this command over shots `0` through `5` and the configured model set.

## Claude Haiku 4.5

| Shots | x | x + native NL | x + translated NL | Delta native | Delta translated | Gap |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 32.70% | 55.70% | 42.41% | +23.00pp | +9.70pp | -13.29pp |
| 1 | 32.70% | 55.70% | 45.78% | +23.00pp | +13.08pp | -9.92pp |
| 2 | 32.70% | 55.70% | 49.58% | +23.00pp | +16.88pp | -6.12pp |
| 3 | 32.70% | 55.70% | 48.52% | +23.00pp | +15.82pp | -7.17pp |
| 4 | 32.70% | 55.70% | 49.37% | +23.00pp | +16.67pp | -6.33pp |
| 5 | 32.70% | 55.70% | 50.00% | +23.00pp | +17.30pp | -5.70pp |
| 10 (paper legacy) | 39.00% | 56.50% | 52.00% | +17.50pp | +13.00pp | -4.50pp |

The 10-shot row is the older paper result and is not from the same `subset_fraction=0.25` sweep as the 0-5 shot rows.
