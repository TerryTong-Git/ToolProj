"""
Mixed-model experiment package for analyzing LLM benchmark performance.

The package provides:
- Benchmark generation via OpenRouter/LM-Eval-Harness (optional UK AI Safety Evals).
- Data loading/merging utilities for benchmark CSVs.
- Linear mixed-model fitting utilities.
- 3D visualization helpers for predicted performance surfaces.
"""

__all__ = [
    "benchmarker",
    "data_loader",
    "mixed_model",
    "visualization",
]
