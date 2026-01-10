#!/usr/bin/env python3
"""
Pretraining Data Mutual Information Pipeline

Compute mutual information between semantic concept labels and representations
(code vs NL) using existing DCLM (NL) and StarCoder (code) pretraining data.

Approach:
    I(label; Z) >= H(label) - CE(label|Z)   (variational lower bound)

This script reuses components from:
- exps_logistic: ConceptClassifier, SentenceTransformersFeaturizer, EvaluationMetrics
- probes.py: Probe generation, evaluators, SemanticLabeler
- data_loader.py: Dataset loading and subsampling
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse from exps_logistic
from exps_logistic.classifier import ConceptClassifier
from exps_logistic.featurizer import SentenceTransformersFeaturizer
from exps_logistic.metrics import EvaluationMetrics, compute_metrics

# Local modules
from .data_loader import (
    Document,
    get_default_data_records,
    load_multiple_datasets,
    proportional_subsample,
)
from .probes import SemanticLabeler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ------------------------------------------------------------------------------
# MI computation result
# ------------------------------------------------------------------------------


@dataclass
class ChannelMIResult:
    """Mutual information results for a single channel."""

    channel: str
    n_samples: int
    metrics: EvaluationMetrics

    @property
    def mi_lower_bound(self) -> float:
        """I(label; Z) >= H(label) - CE(label|Z)"""
        return float(self.metrics.mutual_info_lower_bound)


# ------------------------------------------------------------------------------
# Pipeline functions
# ------------------------------------------------------------------------------


def label_documents(
    documents: List[Document],
    labeler: SemanticLabeler,
    out_path: Optional[str] = None,
) -> List[dict]:
    """
    Label documents using the semantic labeler.

    Args:
        documents: List of documents to label
        labeler: SemanticLabeler instance
        out_path: Optional path to save labeled data

    Returns:
        List of labeled document dictionaries
    """
    labeled = []
    t0 = time.time()

    for i, doc in enumerate(tqdm(documents, desc="Labeling documents")):
        result = labeler.label(doc.text)

        if result.label is not None:
            labeled.append(
                {
                    "id": doc.id,
                    "rep": doc.rep,
                    "text": doc.text,
                    "label": result.label,
                    "conf": result.confidence,
                    "source": doc.source,
                }
            )

        if (i + 1) % 100 == 0:
            dt = time.time() - t0
            logger.info(f"Processed {i + 1}/{len(documents)}, kept {len(labeled)}, rate={i / max(dt, 1e-6):.1f}/s")

    logger.info(f"Labeled {len(labeled)}/{len(documents)} documents")

    # Save if requested
    if out_path and labeled:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "rep", "label", "conf", "source", "text"])
            writer.writeheader()
            writer.writerows(labeled)
        logger.info(f"Saved labels to: {out_path}")

    return labeled


def compute_channel_mi(
    texts: List[str],
    labels: List[str],
    channel: str,
    featurizer: SentenceTransformersFeaturizer,
    classifier_C: float = 2.0,
    classifier_max_iter: int = 400,
    test_size: float = 0.2,
    seed: int = 0,
) -> ChannelMIResult:
    """
    Compute MI lower bound for a single channel.

    Args:
        texts: List of text documents
        labels: List of corresponding labels
        channel: Channel name ('code' or 'nl')
        featurizer: Feature extractor
        classifier_C: Regularization strength
        classifier_max_iter: Max iterations for logistic regression
        test_size: Fraction of data for testing
        seed: Random seed

    Returns:
        ChannelMIResult with metrics
    """
    logger.info(f"Computing MI for channel: {channel} ({len(texts)} samples)")

    # Extract embeddings
    logger.info("  Extracting embeddings...")
    X = featurizer.transform(texts)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=seed, stratify=labels)

    # Train classifier
    logger.info(f"  Training classifier ({len(set(labels))} classes)...")
    clf = ConceptClassifier(C=classifier_C, max_iter=classifier_max_iter)
    clf.fit(X_train, y_train)

    # Evaluate
    result = clf.evaluate(X_test, y_test)

    # Compute metrics
    classes_idx = np.arange(clf.n_classes)
    test_labels_str = [clf.label_encoder.inverse_transform([y])[0] for y in result.true_labels]

    metrics = compute_metrics(
        y_true=result.true_labels,
        y_pred=result.predictions,
        probabilities=result.probabilities,
        classes_idx=classes_idx,
        n_train=len(y_train),
        n_test=len(y_test),
        test_labels=test_labels_str,
    )

    logger.info(
        f"  {channel}: H={metrics.empirical_entropy:.4f} bits, "
        f"CE={metrics.cross_entropy_bits:.4f} bits, "
        f"I>={metrics.mutual_info_lower_bound:.4f} bits"
    )

    return ChannelMIResult(channel=channel, n_samples=len(texts), metrics=metrics)


def run_pipeline(args: object) -> None:  # type: ignore[no-untyped-def]
    """Run the full MI computation pipeline."""

    # Step 1: Load data
    logger.info("=" * 60)
    logger.info("Step 1: Loading data...")
    logger.info("=" * 60)

    all_data = load_multiple_datasets(
        records=get_default_data_records(),
        max_samples_per_dataset=args.max_samples_per_dataset,  # type: ignore[attr-defined]
    )

    if not any(all_data):
        logger.error("No data loaded! Check data paths.")
        return

    # Step 2: Subsample
    logger.info("=" * 60)
    logger.info("Step 2: Subsampling...")
    logger.info("=" * 60)

    sample = proportional_subsample(all_data, rate=args.subsample_rate, seed=args.seed)  # type: ignore[attr-defined]
    logger.info(f"Subsampled {len(sample)} documents (~{args.subsample_rate * 100:.1f}%)")  # type: ignore[attr-defined]

    rep_counts = Counter(doc.rep for doc in sample)
    for rep, cnt in rep_counts.items():
        logger.info(f"  {rep}: {cnt} samples")

    # Step 3: Label via LLM probing
    logger.info("=" * 60)
    logger.info("Step 3: Semantic labeling...")
    logger.info("=" * 60)

    labeler = SemanticLabeler(
        model_name=args.hf_model,  # type: ignore[attr-defined]
        probes_per_kind=args.probes_per_kind,  # type: ignore[attr-defined]
        max_new_tokens=args.max_new_tokens,  # type: ignore[attr-defined]
        batch_size=args.apply_batch_size,  # type: ignore[attr-defined]
        conf_threshold=args.conf_threshold,  # type: ignore[attr-defined]
        seed=args.seed,  # type: ignore[attr-defined]
        dtype=args.dtype,  # type: ignore[attr-defined]
        device=args.device,  # type: ignore[attr-defined]
    )

    labeled = label_documents(sample, labeler, out_path=args.out_labels)  # type: ignore[attr-defined]

    if len(labeled) < args.min_samples:  # type: ignore[attr-defined]
        logger.error(f"Too few labeled samples ({len(labeled)}). Need at least {args.min_samples}.")  # type: ignore[attr-defined]
        return

    # Step 4: Compute MI per channel
    logger.info("=" * 60)
    logger.info("Step 4: Computing MI per channel...")
    logger.info("=" * 60)

    featurizer = SentenceTransformersFeaturizer(args.embed_model, device=args.embed_device)  # type: ignore[attr-defined]

    by_channel: dict[str, List[dict]] = defaultdict(list)
    for row in labeled:
        by_channel[row["rep"]].append(row)

    results = []
    for channel, rows in by_channel.items():
        if len(rows) < args.min_samples:  # type: ignore[attr-defined]
            logger.warning(f"Channel {channel} has only {len(rows)} samples, skipping")
            continue

        label_counts = Counter(r["label"] for r in rows)
        if len(label_counts) < 2:
            logger.warning(f"Channel {channel} has only {len(label_counts)} unique labels, skipping")
            continue

        texts = [r["text"] for r in rows]
        labels = [r["label"] for r in rows]

        result = compute_channel_mi(
            texts,
            labels,
            channel,
            featurizer,
            classifier_C=args.C,  # type: ignore[attr-defined]
            classifier_max_iter=args.max_iter,  # type: ignore[attr-defined]
            test_size=args.test_size,  # type: ignore[attr-defined]
            seed=args.seed,  # type: ignore[attr-defined]
        )
        results.append(result)

    # Step 5: Report results
    print_results(results)

    # Save results
    if args.out_results:  # type: ignore[attr-defined]
        save_results(results, args.out_results)  # type: ignore[attr-defined]


def print_results(results: List[ChannelMIResult]) -> None:
    """Print MI results to console."""
    print("\n" + "=" * 60)
    print("=== MI Results ===")
    print("=" * 60)

    for r in results:
        m = r.metrics
        print(f"\nChannel: {r.channel}")
        print(f"  N samples:     {r.n_samples}")
        print(f"  N classes:     {m.n_classes}")
        print(f"  H(label):      {m.empirical_entropy:.4f} bits")
        print(f"  CE(label|Z):   {m.cross_entropy_bits:.4f} bits")
        print(f"  I(label;Z) >=  {m.mutual_info_lower_bound:.4f} bits")
        print(f"  Accuracy:      {m.accuracy:.4f}")
        print(f"  Macro F1:      {m.macro_f1:.4f}")

    # Comparison
    if len(results) >= 2:
        print("\n" + "-" * 40)
        print("Comparison:")
        code_result = next((r for r in results if r.channel == "code"), None)
        nl_result = next((r for r in results if r.channel == "nl"), None)

        if code_result and nl_result:
            diff = code_result.mi_lower_bound - nl_result.mi_lower_bound
            print(f"  I(label; Z_code) - I(label; Z_nl) = {diff:.4f} bits")
            if diff > 0:
                print("  => Code representations capture MORE information about semantic labels")
            else:
                print("  => NL representations capture MORE information about semantic labels")

    print("=" * 60)


def save_results(results: List[ChannelMIResult], out_path: str) -> None:
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    results_dict = [
        {
            "channel": r.channel,
            "n_samples": r.n_samples,
            "n_classes": r.metrics.n_classes,
            "entropy_bits": r.metrics.empirical_entropy,
            "cross_entropy_bits": r.metrics.cross_entropy_bits,
            "mi_lower_bound_bits": r.metrics.mutual_info_lower_bound,
            "accuracy": r.metrics.accuracy,
            "macro_f1": r.metrics.macro_f1,
        }
        for r in results
    ]
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Saved results to: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute MI between concept labels and code/NL representations")

    # Data
    p.add_argument("--max-samples-per-dataset", type=int, default=10000)
    p.add_argument("--subsample-rate", type=float, default=0.01)
    p.add_argument("--min-samples", type=int, default=50)

    # LLM probing
    p.add_argument("--hf-model", type=str, default="google/gemma-2-9b-it")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu"])
    p.add_argument("--probes-per-kind", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--apply-batch-size", type=int, default=32)
    p.add_argument("--conf-threshold", type=float, default=0.6)

    # Embeddings
    p.add_argument("--embed-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embed-device", type=str, default="cuda")

    # Classifier
    p.add_argument("--C", type=float, default=2.0)
    p.add_argument("--max-iter", type=int, default=400)
    p.add_argument("--test-size", type=float, default=0.2)

    # Output
    p.add_argument("--out-labels", type=str, default="outputs/labeled_pretrain.csv")
    p.add_argument("--out-results", type=str, default="outputs/mi_results.json")

    # General
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
