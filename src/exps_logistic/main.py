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

import logging
import time

import numpy as np

from .classifier import ConceptClassifier
from .config import ExperimentConfig, parse_args
from .data_utils import (
    filter_by_rep,
    load_data,
    prepare_labels,
    stratified_split_robust,
)
from .featurizer import build_featurizer
from .metrics import compute_metrics, print_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run(config: ExperimentConfig) -> None:
    """Main experiment runner."""

    # Load data
    logger.info("Loading data...")
    df = load_data(config.csv)
    logger.info(f"Loaded {len(df)} samples")

    # Filter by representation type
    df = filter_by_rep(df, config.rep)

    # Prepare labels
    logger.info("Preparing labels...")
    df = prepare_labels(df, config.label, config.value_bins)

    # Log gamma diversity sanity check
    gb = df.assign(kd=df["kind"].astype(str) + "|d" + df["digits"].astype(str)).groupby("kd")["gamma"].nunique()
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

    featurizer = build_featurizer(
        config.feats,
        config.embed_model,
        config.pool,
        config.strip_fences,
        config.device,
        config.batch,
    )

    if hasattr(featurizer, "fit"):
        featurizer.fit(texts_tr)

    X_train = featurizer.transform(texts_tr)
    X_test = featurizer.transform(texts_te)
    logger.info(f"Feature shape: {X_train.shape}")

    # Train classifier
    logger.info("Training classifier...")
    classifier = ConceptClassifier(C=config.C, max_iter=config.max_iter)
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
    if config.save_preds:
        logger.info(f"Saving predictions to {config.save_preds}...")
        out = test_df[["rationale", "kind", "digits", "prompt"]].copy()
        out["true_label"] = test_df["label"].values
        out["pred_label"] = result.label_encoder.inverse_transform(result.predictions)
        out["neglogp_true_nat"] = -np.log(result.probabilities[np.arange(len(result.probabilities)), result.true_labels] + 1e-15)
        out.to_csv(config.save_preds, index=False)
        logger.info(f"Saved predictions: {config.save_preds}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    config = parse_args()
    run(config)
    elapsed = time.perf_counter() - start_time
    logger.info(f"Elapsed time: {elapsed:.4f} seconds")
