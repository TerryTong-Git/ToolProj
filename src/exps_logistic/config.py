#!/usr/bin/env python3
"""Configuration and argument parsing for logistic regression experiments."""

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration for the logistic regression concept classification experiment."""

    # Data sources
    tbdir: Optional[str] = None
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
    strip_fences: bool = False

    # Classifier
    C: float = 2.0
    max_iter: int = 400

    # Reporting
    bits: bool = False
    save_preds: Optional[str] = None


def parse_args() -> ExperimentConfig:
    """Parse command-line arguments and return an ExperimentConfig."""
    p = argparse.ArgumentParser(description="CoT -> joint label classification with LM embeddings + multinomial logistic regression.")

    # Data sources
    p.add_argument("--tbdir", type=str, default=None, help="Read rationales directly from TensorBoard logs")
    p.add_argument("--csv", type=str, default=None, help="CSV with columns: rationale, kind, digits, [prompt], [rep], [split]")
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
    p.add_argument("--strip-fences", action="store_true", help="Strip ``` code fences before embedding.")

    # Classifier
    p.add_argument("--C", type=float, default=2.0, help="Regularization strength")
    p.add_argument("--max_iter", type=int, default=400)

    # Reporting
    p.add_argument("--bits", action="store_true", help="Report CE in bits and print MI lower bound.")
    p.add_argument("--save-preds", type=str, default=None, help="Optional path to save test predictions CSV.")

    args = p.parse_args()

    return ExperimentConfig(
        tbdir=args.tbdir,
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
        strip_fences=getattr(args, "strip_fences", False),
        C=args.C,
        max_iter=args.max_iter,
        bits=args.bits,
        save_preds=getattr(args, "save_preds", None),
    )
