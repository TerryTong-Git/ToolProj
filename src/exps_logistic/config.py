#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from typing import FrozenSet, List, Optional

FG_KINDS: FrozenSet[str] = frozenset(
    {
        "add",
        "sub",
        "mul",
        "lcs",
        "knap",
        "rod",
        "ilp_assign",
        "ilp_prod",
        "ilp_partition",
    }
)

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

NPHARD_KINDS: FrozenSet[str] = frozenset(
    {
        "edp",
        "gcp",
        "ilp_assign",
        "ilp_partition",
        "ilp_prod",
        "knap",
        "ksp",
        "spp",
        "tsp",
    }
)

ARITHMETIC_KINDS: FrozenSet[str] = frozenset(
    {
        "add",
        "sub",
        "mul",
    }
)

ILP_KINDS: FrozenSet[str] = frozenset(
    {
        "ilp_assign",
        "ilp_partition",
        "ilp_prod",
    }
)

EXTENDED_KINDS: FrozenSet[str] = FG_KINDS | CLRS_KINDS | NPHARD_KINDS

KINDS_PRESETS: dict[str, FrozenSet[str]] = {
    "fg": FG_KINDS,
    "clrs": CLRS_KINDS,
    "nphard": NPHARD_KINDS,
    "extended": EXTENDED_KINDS,
    "arithmetic": ARITHMETIC_KINDS,
    "ilp": ILP_KINDS,
}


@dataclass
class ExperimentConfig:
    results_dir: Optional[str] = None
    models: Optional[List[str]] = None
    seeds: Optional[List[int]] = None
    kinds: FrozenSet[str] = FG_KINDS
    rep: str = "all"

    label: str = "gamma"
    value_bins: int = 8
    test_size: float = 0.2
    seed: int = 0
    cv: int = 0
    enable_cv: bool = True

    feats: str = "tfidf"
    embed_model: Optional[str] = None
    pool: str = "mean"
    device: Optional[str] = None
    batch: int = 128
    hf_batch: int = 16
    hf_dtype: str = "auto"
    hf_window_stride: int = 0
    strip_fences: bool = False
    filter_algo_names: bool = True
    filter_comments: bool = True

    C: float = 2.0
    max_iter: int = 400
    logreg_c_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    logreg_max_iter_grid: tuple[int, ...] = (100, 200, 400)
    logreg_cv_folds: int = 5

    bits: bool = False
    save_preds: Optional[str] = None


def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser()

    p.add_argument("--results-dir", type=str, required=True)
    p.add_argument("--models", type=str, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--kinds", type=str, nargs="+", default=None)
    p.add_argument("--kinds-preset", type=str, choices=list(KINDS_PRESETS.keys()), default=None)
    p.add_argument("--rep", choices=["nl", "code", "sim_reasoning", "all"], default="all")

    p.add_argument("--label", choices=["theta_new", "gamma", "kind"], default="gamma")
    p.add_argument("--value-bins", type=int, default=8)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cv", type=int, default=0)
    p.add_argument("--enable-cv", dest="enable_cv", action="store_true", default=True)
    p.add_argument("--no-cv", dest="enable_cv", action="store_false")

    p.add_argument("--feats", choices=["tfidf", "hf-cls", "st", "openai", "openrouter"], default="tfidf")
    p.add_argument("--embed-model", type=str, default=None)
    p.add_argument("--pool", choices=["mean", "cls"], default="mean")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--hf-batch", type=int, default=16)
    p.add_argument("--hf-dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--hf-window-stride", type=int, default=0)
    p.add_argument("--strip-fences", action="store_true")
    p.add_argument("--filter-algo-names", dest="filter_algo_names", action="store_true", default=True)
    p.add_argument("--no-filter-algo-names", dest="filter_algo_names", action="store_false")
    p.add_argument("--filter-comments", dest="filter_comments", action="store_true", default=True)
    p.add_argument("--no-filter-comments", dest="filter_comments", action="store_false")

    p.add_argument("--C", type=float, default=2.0)
    p.add_argument("--max_iter", type=int, default=400)
    p.add_argument("--logreg-c-grid", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--logreg-max-iter-grid", type=int, nargs="+", default=[100, 200, 400])
    p.add_argument("--logreg-cv-folds", type=int, default=5)

    p.add_argument("--bits", action="store_true")
    p.add_argument("--save-preds", type=str, default=None)

    args = p.parse_args()

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
        filter_algo_names=args.filter_algo_names,
        filter_comments=args.filter_comments,
        C=args.C,
        max_iter=args.max_iter,
        logreg_c_grid=tuple(args.logreg_c_grid),
        logreg_max_iter_grid=tuple(args.logreg_max_iter_grid),
        logreg_cv_folds=args.logreg_cv_folds,
        bits=args.bits,
        save_preds=getattr(args, "save_preds", None),
    )
