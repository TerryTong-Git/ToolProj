#!/usr/bin/env python3
"""
CoT -> joint label classification with LM embeddings + multinomial logistic regression.

Targets (choose with --label):
  - kind         : concept only
  - theta_new    : (kind, digits)
  - gamma        : (kind, digits, value-bin from problem text)

Embedding backends (choose with --feats):
  - tfidf        : word+char TF-IDF baseline
  - hf-cls       : HuggingFace Transformer encoder
  - st           : Sentence-Transformers encode
  - openai       : OpenAI embeddings API
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .classifier import ConceptClassifier
from .config import ExperimentConfig, parse_args
from .data_utils import (
    filter_by_kinds,
    filter_by_rep,
    load_data,
    prepare_labels,
    stratified_split_robust,
)
from .featurizer import build_featurizer
from .metrics import compute_metrics, print_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _infer_metadata(df: pd.DataFrame, config: ExperimentConfig) -> Tuple[str, str]:
    """Infer model name and seed for serialization."""
    model = "unknown_model"
    if "model" in df.columns and len(df["model"].dropna()):
        model = str(df["model"].iloc[0]).replace("/", "-")

    if "seed" in df.columns and len(df["seed"].dropna()):
        try:
            seed = str(int(df["seed"].iloc[0]))
        except Exception:
            seed = str(df["seed"].iloc[0])
    else:
        seed = str(config.seed)
    return model, seed


def _default_save_path(model: str, seed: str, config: ExperimentConfig) -> Path:
    """Automatic save path that encodes model/seed/date/rep/featurizer."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    feats = config.feats
    rep = config.rep
    embed = (config.embed_model or "na").replace("/", "-")
    fname = f"{model}_seed{seed}_{rep}_{feats}-{embed}_{ts}.json"
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / fname


def run(config: ExperimentConfig) -> None:
    """Main experiment runner."""

    # Load data
    logger.info("Loading data...")
    assert config.results_dir is not None, "Results directory must be provided"
    df = load_data(config.results_dir, config.models, config.seeds)
    logger.info(f"Loaded {len(df)} samples")
    model_name, seed_str = _infer_metadata(df, config)
    if config.models is not None and len(config.models) > 0:
        model_name = config.models[0].split("/")[-1]

    # Filter by problem kinds (default: fine-grained only)
    df = filter_by_kinds(df, set(config.kinds) if config.kinds else None)

    # Filter by representation type
    df = filter_by_rep(df, config.rep)

    # Prepare labels
    logger.info("Preparing labels...")
    df = prepare_labels(df, config.label, config.value_bins)

    # Log gamma diversity sanity check
    gb = df.assign(kd=df["kind"].astype(str) + "|d" + df["digits"].astype(str)).groupby("kd")["gamma"].nunique()  # type: ignore[operator]
    logger.info(f"[gamma sanity] distinct bins per (kind,d):\n{gb.value_counts().sort_index()}")

    n_empty = (df["prompt"].astype(str).str.len() == 0).sum()
    logger.info(f"[gamma sanity] empty prompt rows: {n_empty} / {len(df)}")

    # Filter valid samples
    df = df[df["label"].astype(str).str.len() > 0].reset_index(drop=True)
    df = df[df["rationale"].astype(str).str.len() > 0].reset_index(drop=True)

    # Split data
    logger.info("Splitting data...")
    train_df, test_df = stratified_split_robust(df, y_col="label", test_size=config.test_size, seed=config.seed)
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Extract features
    logger.info("Extracting features...")
    texts_tr = train_df["rationale"].astype(str).tolist()
    texts_te = test_df["rationale"].astype(str).tolist()

    # import pdb; pdb.set_trace()
    featurizer = build_featurizer(
        config.feats,
        config.embed_model,
        config.pool,
        config.strip_fences,
        config.device,
        config.batch,
        config.hf_batch,
        config.hf_dtype,
        config.hf_window_stride,
    )

    if hasattr(featurizer, "fit"):
        featurizer.fit(texts_tr)

    X_train = featurizer.transform(texts_tr)
    X_test = featurizer.transform(texts_te)
    logger.info(f"Feature shape: {X_train.shape}")

    # Train classifier
    logger.info("Training classifier...")
    if config.enable_cv:
        # Hyperparameter search (up to 5-fold stratified CV) to reduce overfitting.
        best_C, best_max_iter, cv_score = ConceptClassifier.tune_hyperparams(
            X_train,
            train_df["label"].astype(str).tolist(),
            config.logreg_c_grid,
            config.logreg_max_iter_grid,
            cv_folds=config.logreg_cv_folds,
        )
        logger.info(f"CV best accuracy={cv_score:.4f} with C={best_C}, max_iter={best_max_iter}")
    else:
        best_C, best_max_iter = config.C, config.max_iter
        logger.info(f"CV disabled; using C={best_C}, max_iter={best_max_iter}")

    classifier = ConceptClassifier(C=best_C, max_iter=best_max_iter)
    classifier.fit(X_train, train_df["label"].astype(str).tolist())

    # Evaluate
    logger.info("Evaluating...")
    result = classifier.evaluate(X_test, test_df["label"].astype(str).tolist())

    # Compute and print metrics
    metrics = compute_metrics(
        y_true=result.true_labels,
        y_pred=result.predictions,
        probabilities=result.probabilities,
        classes_idx=np.arange(classifier.n_classes),
        n_train=len(train_df),
        n_test=len(test_df),
        test_labels=test_df["label"].astype(str).tolist(),
    )
    print_results(metrics, config)

    # Save predictions
    if config.save_preds is None:
        config.save_preds = str(_default_save_path(model_name, seed_str, config))

    if config.save_preds:
        logger.info(f"Saving predictions to {config.save_preds}...")
        out = test_df[["rationale", "kind", "digits", "prompt"]].copy()
        out["true_label"] = test_df["label"].values
        out["pred_label"] = result.label_encoder.inverse_transform(result.predictions)
        out["neglogp_true_nat"] = -np.log(result.probabilities[np.arange(len(result.probabilities)), result.true_labels] + 1e-15)
        # metadata columns for downstream analysis
        out["model"] = model_name
        out["seed"] = seed_str
        out["rep"] = config.rep
        out["feats"] = config.feats
        out["embed_model"] = config.embed_model or ""
        out["run_ts"] = datetime.now().isoformat()
        records = out.to_dict(orient="records")
        results_summary = {
            "target_label": config.label,
            "n_classes": metrics.n_classes,
            "feats": config.feats,
            "embed_model": config.embed_model or "n/a",
            "pool": config.pool if config.feats == "hf-cls" else "-",
            "rep_filter": config.rep,
            "n_train": metrics.n_train,
            "n_test": metrics.n_test,
            "cross_entropy_bits": metrics.cross_entropy_bits,
            "cross_entropy_nats": metrics.cross_entropy,
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "empirical_entropy_bits": metrics.empirical_entropy,
            "mutual_info_lower_bound_bits": metrics.mutual_info_lower_bound,
        }
        with open(config.save_preds, "w", encoding="utf-8") as f:
            json.dump(records + [results_summary], f, ensure_ascii=False)
        logger.info(f"Saved predictions: {config.save_preds}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    config = parse_args()
    run(config)
    elapsed = time.perf_counter() - start_time
    logger.info(f"Elapsed time: {elapsed:.4f} seconds")
