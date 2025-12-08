#!/usr/bin/env python3
"""Configuration and argument parsing for logistic regression experiments."""

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration for the logistic regression concept classification experiment."""

    # Data sources
    csv: Optional[str] = None
    rep: str = "all"  # nl, code, or all

    # Label configuration
    label: str = "gamma"  # theta_new, gamma, or kind
    value_bins: int = 8
    test_size: float = 0.2
    seed: int = 0
    cv: int = 0

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
    save_preds: Optional[str] = None


def parse_args() -> ExperimentConfig:
    """Parse command-line arguments and return an ExperimentConfig."""
    p = argparse.ArgumentParser(description="CoT -> joint label classification with LM embeddings + multinomial logistic regression.")

    # Data sources
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV input. Accepts canonical format (rationale, kind, digits, [prompt], [rep]) "
        "or exps_performance results CSVs (digit, kind, question, nl_reasoning, code_answer).",
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
    p.add_argument("--save-preds", type=str, default=None, help="Optional path to save test predictions CSV.")

    args = p.parse_args()

    return ExperimentConfig(
        csv=args.csv,
        rep=args.rep,
        label=args.label,
        value_bins=getattr(args, "value_bins", 8),
        test_size=getattr(args, "test_size", 0.2),
        seed=args.seed,
        cv=args.cv,
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
