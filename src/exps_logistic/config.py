#!/usr/bin/env python3
"""Configuration and argument parsing for logistic regression experiments.

This module defines the configuration for MI estimation experiments using
multinomial logistic regression on CoT embeddings.
"""

import argparse
from dataclasses import dataclass
from typing import FrozenSet, List, Optional

# Fine-grained problem kinds (custom examples only)
FG_KINDS: FrozenSet[str] = frozenset(
    {
        "add",
        "sub",
        "mul",  # Arithmetic
        "lcs",
        "knap",
        "rod",  # Dynamic programming
        "ilp_assign",
        "ilp_prod",
        "ilp_partition",  # Integer linear programming
    }
)

# CLRS algorithmic reasoning benchmark (30 algorithms)
CLRS_KINDS: FrozenSet[str] = frozenset(
    {
        "activity_selector",
        "articulation_points",
        "bellman_ford",
        "bfs",
        "binary_search",
        "bridges",
        "bubble_sort",
        "dag_shortest_paths",
        "dfs",
        "dijkstra",
        "find_maximum_subarray_kadane",
        "floyd_warshall",
        "graham_scan",
        "heapsort",
        "insertion_sort",
        "jarvis_march",
        "kmp_matcher",
        "lcs_length",
        "matrix_chain_order",
        "minimum",
        "mst_kruskal",
        "mst_prim",
        "naive_string_matcher",
        "optimal_bst",
        "quickselect",
        "quicksort",
        "segments_intersect",
        "strongly_connected_components",
        "task_scheduling",
        "topological_sort",
    }
)

# NP-hard problem evaluation
NPHARD_KINDS: FrozenSet[str] = frozenset(
    {
        "edp",
        "gcp",
        "ksp",
        "spp",
        "tsp",
    }
)

# Extended kinds: fine-grained + CLRS + NP-hard (excluding gsm8k)
EXTENDED_KINDS: FrozenSet[str] = FG_KINDS | CLRS_KINDS | NPHARD_KINDS

# Preset mappings for --kinds-preset argument
KINDS_PRESETS: dict[str, FrozenSet[str]] = {
    "fg": FG_KINDS,
    "clrs": CLRS_KINDS,
    "nphard": NPHARD_KINDS,
    "extended": EXTENDED_KINDS,
}


@dataclass
class ExperimentConfig:
    """Configuration for the logistic regression concept classification experiment."""

    # Data sources
    results_dir: Optional[str] = None  # Directory containing JSONL result files
    models: Optional[List[str]] = None  # Optional model name filter
    seeds: Optional[List[int]] = None  # Optional seed filter
    kinds: FrozenSet[str] = FG_KINDS  # Problem kinds to include (default: fine-grained only)
    rep: str = "all"  # nl, code, or all

    # Label configuration
    label: str = "gamma"  # theta_new, gamma, or kind
    value_bins: int = 8
    test_size: float = 0.2
    seed: int = 0
    cv: int = 0
    enable_cv: bool = True

    # Feature extraction
    feats: str = "tfidf"  # tfidf, hf-cls, st, openai
    embed_model: Optional[str] = None
    pool: str = "mean"  # mean or cls
    device: Optional[str] = None
    batch: int = 128
    hf_batch: int = 16
    hf_dtype: str = "auto"
    hf_window_stride: int = 0
    strip_fences: bool = False

    # Classifier
    C: float = 2.0
    max_iter: int = 400
    logreg_c_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    logreg_max_iter_grid: tuple[int, ...] = (100, 200, 400)
    logreg_cv_folds: int = 5

    # Reporting
    bits: bool = False
    save_preds: Optional[str] = None  # Path to save predictions JSON, None for auto-generated path


def parse_args() -> ExperimentConfig:
    """Parse command-line arguments and return an ExperimentConfig."""
    p = argparse.ArgumentParser(description="CoT -> joint label classification with LM embeddings + multinomial logistic regression.")

    # Data sources
    p.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing JSONL result files (Record schema) to load via create_big_df.",
    )
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of model names to include (matches the `model` column).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of seed values to include (matches the `seed` column).",
    )
    p.add_argument(
        "--kinds",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of problem kinds to include. Defaults to fine-grained only (add, sub, mul, lcs, knap, rod, ilp_*).",
    )
    p.add_argument(
        "--kinds-preset",
        type=str,
        choices=list(KINDS_PRESETS.keys()),
        default=None,
        help="Use a preset kinds configuration: 'fg' (fine-grained only) or 'extended' (fg + clrs30 + nphardeval). Overridden by --kinds.",
    )
    p.add_argument("--rep", choices=["nl", "code", "all"], default="all")

    # Label configuration
    p.add_argument(
        "--label",
        choices=["theta_new", "gamma", "kind"],
        default="gamma",
        help="Classification target: θ_new=(kind,digits) or γ=(kind,digits,value-bin) or kind only.",
    )
    p.add_argument("--value-bins", type=int, default=8, help="Number of equal-width bins per operand for gamma.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cv", type=int, default=0, help="K-fold cross-validation (0 to disable)")
    p.add_argument(
        "--enable-cv",
        dest="enable_cv",
        action="store_true",
        default=True,
        help="Enable hyperparameter CV grid search (default: enabled).",
    )
    p.add_argument(
        "--no-cv",
        dest="enable_cv",
        action="store_false",
        help="Disable hyperparameter CV grid search and use provided C/max_iter.",
    )

    # Feature extraction
    p.add_argument("--feats", choices=["tfidf", "hf-cls", "st", "openai"], default="tfidf")
    p.add_argument("--embed-model", type=str, default=None, help="hf-cls: HF repo; st: ST repo; openai: embedding model.")
    p.add_argument("--pool", choices=["mean", "cls"], default="mean", help="Pooling for hf-cls.")
    p.add_argument("--device", type=str, default=None, help="Force device for hf-cls/st (e.g. cuda, cpu).")
    p.add_argument("--batch", type=int, default=128, help="Batch size for OpenAI embeddings.")
    p.add_argument("--hf-batch", type=int, default=16, help="Batch size for hf-cls encoder to reduce memory.")
    p.add_argument(
        "--hf-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for hf-cls encoder to reduce memory.",
    )
    p.add_argument(
        "--hf-window-stride",
        type=int,
        default=0,
        help="Stride for sliding window over long sequences for hf-cls. 0 = truncate to max length (lowest memory).",
    )
    p.add_argument("--strip-fences", action="store_true", help="Strip ``` code fences before embedding.")

    # Classifier
    p.add_argument("--C", type=float, default=2.0, help="Regularization strength")
    p.add_argument("--max_iter", type=int, default=400)
    p.add_argument(
        "--logreg-c-grid",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
        help="Grid of C values for logistic regression CV.",
    )
    p.add_argument(
        "--logreg-max-iter-grid",
        type=int,
        nargs="+",
        default=[100, 200, 400],
        help="Grid of max_iter values for logistic regression CV.",
    )
    p.add_argument("--logreg-cv-folds", type=int, default=5, help="CV folds for logistic regression hyperparameter search.")

    # Reporting
    p.add_argument("--bits", action="store_true", help="Report CE in bits and print MI lower bound.")
    p.add_argument("--save-preds", type=str, default=None, help="Optional path to save test predictions JSON.")

    args = p.parse_args()

    # Determine kinds: explicit --kinds overrides --kinds-preset, which overrides default
    if args.kinds:
        kinds = frozenset(args.kinds)
    elif args.kinds_preset:
        kinds = KINDS_PRESETS[args.kinds_preset]
    else:
        kinds = FG_KINDS

    return ExperimentConfig(
        results_dir=args.results_dir,
        models=args.models,
        seeds=args.seeds,
        kinds=kinds,
        rep=args.rep,
        label=args.label,
        value_bins=getattr(args, "value_bins", 8),
        test_size=getattr(args, "test_size", 0.2),
        seed=args.seed,
        cv=args.cv,
        enable_cv=args.enable_cv,
        feats=args.feats,
        embed_model=getattr(args, "embed_model", None),
        pool=args.pool,
        device=args.device,
        batch=args.batch,
        hf_batch=args.hf_batch,
        hf_dtype=args.hf_dtype,
        hf_window_stride=args.hf_window_stride,
        strip_fences=getattr(args, "strip_fences", False),
        C=args.C,
        max_iter=args.max_iter,
        logreg_c_grid=tuple(args.logreg_c_grid),
        logreg_max_iter_grid=tuple(args.logreg_max_iter_grid),
        logreg_cv_folds=args.logreg_cv_folds,
        bits=args.bits,
        save_preds=getattr(args, "save_preds", None),
    )
