#!/usr/bin/env python3
"""
Data loading utilities for pretraining corpora.

Supports loading from local HuggingFace dataset formats (parquet, jsonl, etc.)
with automatic channel detection (code vs NL).
"""

from __future__ import annotations

import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)

# Default paths
BASE_DATA_PATH = Path(__file__).parent / "data"
DCLM_PATH = BASE_DATA_PATH / "dclm_small" / "main"
STARCODER_PATH = BASE_DATA_PATH / "starcoder_small" / "main"


# ------------------------------------------------------------------------------
# Data records
# ------------------------------------------------------------------------------


@dataclass
class DataRecord:
    """Configuration for a dataset to load."""

    name: str
    path: Path
    file_type: str  # 'parquet', 'json', 'text'
    column_name: str  # column containing the text
    rep: str  # 'code' or 'nl'


@dataclass
class Document:
    """A loaded document with metadata."""

    id: str
    text: str
    rep: str  # 'code' or 'nl'
    source: str  # dataset name


# ------------------------------------------------------------------------------
# Code detection
# ------------------------------------------------------------------------------

CODE_MARKERS = re.compile(r"(\bdef\b|\bclass\b|;|{|}|\breturn\b|\bfor\b|\bwhile\b)")


def looks_like_code(text: str) -> bool:
    """Heuristic detection of code content."""
    s = text.strip()
    return bool(CODE_MARKERS.search(s))


# ------------------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------------------


def get_default_data_records() -> List[DataRecord]:
    """Return default data records for DCLM (NL) and StarCoder (code)."""
    return [
        DataRecord(
            name="starcoder",
            path=STARCODER_PATH,
            file_type="parquet",
            column_name="content",
            rep="code",
        ),
        DataRecord(
            name="dclm",
            path=DCLM_PATH,
            file_type="json",
            column_name="text",
            rep="nl",
        ),
    ]


def load_dataset_from_record(
    record: DataRecord,
    max_samples: Optional[int] = None,
) -> List[Document]:
    """
    Load documents from a dataset record.

    Args:
        record: DataRecord specifying the dataset configuration
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of Document objects
    """
    logger.info(f"Loading {record.name} from {record.path}")

    if not record.path.exists():
        logger.warning(f"Path does not exist: {record.path}")
        return []

    try:
        if record.file_type == "parquet":
            ds = load_dataset("parquet", data_dir=str(record.path))
        elif record.file_type == "json":
            ds = load_dataset("json", data_dir=str(record.path))
        else:
            ds = load_dataset("text", data_dir=str(record.path))
    except Exception as e:
        logger.warning(f"Failed to load {record.name}: {e}")
        return []

    # Get the data split
    data = ds.get("train", ds.get(list(ds.keys())[0]))

    documents = []
    for i, row in enumerate(data):
        if max_samples and i >= max_samples:
            break

        text = row.get(record.column_name) or row.get("text") or row.get("content") or ""
        if not text.strip():
            continue

        documents.append(
            Document(
                id=f"{record.name}_{i}",
                text=text,
                rep=record.rep,
                source=record.name,
            )
        )

    logger.info(f"Loaded {len(documents)} samples from {record.name}")
    return documents


def load_multiple_datasets(
    records: Optional[List[DataRecord]] = None,
    max_samples_per_dataset: Optional[int] = None,
) -> List[List[Document]]:
    """
    Load multiple datasets.

    Args:
        records: List of DataRecords (uses defaults if None)
        max_samples_per_dataset: Maximum samples per dataset

    Returns:
        List of document lists (one per dataset)
    """
    if records is None:
        records = get_default_data_records()

    return [load_dataset_from_record(record, max_samples=max_samples_per_dataset) for record in records]


# ------------------------------------------------------------------------------
# Subsampling
# ------------------------------------------------------------------------------


def proportional_subsample(
    datasets: List[List[Document]],
    rate: float,
    seed: int = 0,
) -> List[Document]:
    """
    Proportionally subsample from each dataset, stratified by rep.

    Args:
        datasets: List of document lists (one per dataset)
        rate: Sampling rate (0.01 = 1%)
        seed: Random seed for reproducibility

    Returns:
        Combined list of sampled documents
    """
    random.seed(seed)
    out = []

    for ds in datasets:
        if not ds:
            continue

        n = len(ds)
        k = max(1, int(round(n * rate)))

        # Stratify by rep
        by_rep: dict[str, List[Document]] = defaultdict(list)
        for doc in ds:
            by_rep[doc.rep].append(doc)

        take = []
        for rep, lst in by_rep.items():
            kk = max(1, int(round(len(lst) * rate)))
            take.extend(random.sample(lst, min(kk, len(lst))))

        # Adjust to target k if needed
        if len(take) < k:
            pool = [doc for doc in ds if doc not in take]
            take.extend(random.sample(pool, min(k - len(take), len(pool))))

        out.extend(take[:k])

    return out
