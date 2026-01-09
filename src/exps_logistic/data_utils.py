#!/usr/bin/env python3
"""Data loading, splitting, and label creation utilities."""

import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, cast

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exps_performance.logger import create_big_df

from .parsers import (
    parse_arithmetic_operands,
    parse_ilp_assign_n,
    parse_ilp_partition_n,
    parse_ilp_prod_PR,
    parse_knap_stats,
    parse_lcs_lengths,
    parse_rod_N,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Binning utilities
# ------------------------------------------------------------------------------


def equal_width_bin(x: int, lo: int, hi: int, K: int) -> int:
    """Equal-width binning for CLOSED interval [lo, hi], returns in {0..K-1}."""
    if K <= 1:
        return 0
    lo, hi = int(lo), int(hi)
    if lo > hi:
        lo, hi = hi, lo
    x = max(lo, min(hi, int(x)))
    span = hi - lo + 1
    idx = ((x - lo) * K) // span
    return max(0, min(K - 1, int(idx)))


# ------------------------------------------------------------------------------
# Label creation
# ------------------------------------------------------------------------------


def make_gamma_label(kind: str, digits: int, problem_text: str, K_bins: int = 8, use_joint_id: bool = True) -> str:
    """
    Create γ = (kind, digits, value-bin) label.

    The value-bin is derived from fields parsed exactly from Problem.text() for each kind.
    """
    k = str(kind)
    d = int(digits)
    t = problem_text or ""

    # Arithmetic: operands in [10^(d-1), 10^d - 1]
    if k in {"add", "sub", "mul", "mix"}:
        parsed = parse_arithmetic_operands(k, t, d)
        lo = 10 ** (d - 1)
        hi = 10**d - 1
        if parsed is None:
            return f"{k}|d{d}|bNA"
        A, B = parsed
        ba = equal_width_bin(A, lo, hi, K_bins)
        bb = equal_width_bin(B, lo, hi, K_bins)
        bin_id = ba * K_bins + bb if use_joint_id else (ba, bb)
        return f"{k}|d{d}|b{bin_id}"

    # LCS: (|S|, |T|)
    if k == "lcs":
        Ls, Lt = parse_lcs_lengths(t, d)
        bLs = equal_width_bin(Ls, 1, max(2, 2 * d), K_bins)
        bLt = equal_width_bin(Lt, 1, max(2, 2 * d), K_bins)
        return f"{k}|d{d}|b{bLs * K_bins + bLt}"

    # Knapsack: (#items, capacity-ratio)
    if k == "knap":
        n_items, cap_ratio = parse_knap_stats(t, d)
        bN = equal_width_bin(n_items, 1, max(3, 2 * d), K_bins)
        bR = equal_width_bin(int(round(cap_ratio * 1000)), 0, 1000, K_bins)
        return f"{k}|d{d}|b{bN * K_bins + bR}"

    # Rod: N
    if k == "rod":
        N = parse_rod_N(t, d)
        bN = equal_width_bin(N, 1, max(2, 2 * d), K_bins)
        return f"{k}|d{d}|b{bN}"

    # ILP assignment: n
    if k == "ilp_assign":
        n = parse_ilp_assign_n(t, d)
        bN = equal_width_bin(n, 2, 7, K_bins)
        return f"{k}|d{d}|b{bN}"

    # ILP production: (P, R)
    if k == "ilp_prod":
        P, R = parse_ilp_prod_PR(t, d)
        bP = equal_width_bin(P, 2, 6, K_bins)
        bR = equal_width_bin(R, 2, 4, K_bins)
        return f"{k}|d{d}|b{bP * K_bins + bR}"

    # ILP partition: #items
    if k == "ilp_partition":
        n_items = parse_ilp_partition_n(t, d)
        bN = equal_width_bin(n_items, 4, 24, K_bins)
        return f"{k}|d{d}|b{bN}"

    return f"{k}|d{d}|bNA"


def create_theta_new_label(kind: str, digits: int) -> str:
    """Create θ_new = (kind, digits) label."""
    return f"{kind}__d{digits}"


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------


def _convert_results_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert results produced by exps_performance (Record schema) into the
    canonical format expected by the logistic pipeline.
    """
    required = {"digit", "kind", "question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results data missing required columns {missing}. Got: {df.columns.tolist()}")

    rows = []
    for _, row in df.iterrows():
        nl = str(row.get("nl_reasoning", "") or "").strip()
        code = str(row.get("sim_code", "") or "").strip()

        # Require both modalities to be present for fairness/balanced classes
        if not nl or not code:
            continue

        digits = int(row["digit"])
        base = {
            "kind": row["kind"],
            "digits": digits,
            # Use the original question text for γ parsing / context
            "prompt": row.get("question", ""),
        }

        rows.append({**base, "rationale": nl, "rep": "nl"})
        rows.append({**base, "rationale": code, "rep": "code"})

    if not rows:
        raise ValueError("No rationale text found in results CSV (nl_reasoning/code_answer were empty).")

    return pd.DataFrame(rows)


def load_data(results_dir: str, models: Optional[List[str]] = None, seeds: Optional[List[int]] = None) -> pd.DataFrame:
    """Load data from a results directory containing JSONL files (Record schema)."""
    root = Path(results_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    jsonl_files = sorted(root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL result files found under {results_dir}")

    df = create_big_df(jsonl_files)
    if df.empty:
        raise ValueError(f"No rows loaded from JSONL files under {results_dir}")
    # import pdb; pdb.set_trace()
    if models:
        df = df[df["model"].isin(models)]
        if df.empty:
            raise ValueError(f"No rows left after filtering by models={models}")

    if seeds is not None:
        if "seed" not in df.columns:
            raise ValueError("Seed filtering requested but 'seed' column is missing in the data.")
        df = df[df["seed"].isin(seeds)]
        if df.empty:
            raise ValueError(f"No rows left after filtering by seeds={seeds}")

    # Already in canonical format (tests/synthetic data)
    if "rationale" in df.columns and ("digits" in df.columns or "digit" in df.columns):
        if "digit" in df.columns and "digits" not in df.columns:
            df = df.rename(columns={"digit": "digits"})
        return df

    # exps_performance results (Record schema)
    if {"digit", "kind", "question"} <= set(df.columns):
        converted = _convert_results_df(df)
        logger.info(
            "Loaded Record-schema results; converted %d rows to %d rationale entries (nl/code).",
            len(df),
            len(converted),
        )
        return converted

    raise ValueError(
        "Unsupported results format. Expected columns {rationale, kind, digits} or Record schema "
        f"(digit, kind, question, nl_reasoning/code_answer). Got: {df.columns.tolist()}"
    )


def filter_by_rep(df: pd.DataFrame, rep: str) -> pd.DataFrame:
    """Filter dataframe by representation type (nl, code, or all)."""
    if rep == "all" or "rep" not in df.columns:
        return df
    filtered = df[df["rep"].astype(str).str.lower() == rep]
    if len(filtered) == 0:
        raise ValueError(f"No rows after filtering rep={rep}")
    return filtered


def filter_by_kinds(df: pd.DataFrame, kinds: Optional[set]) -> pd.DataFrame:
    """Filter dataframe by problem kinds."""
    if kinds is None or "kind" not in df.columns:
        return df
    filtered = df[df["kind"].isin(kinds)]
    if len(filtered) == 0:
        raise ValueError(f"No rows after filtering by kinds={kinds}")
    logger.info("Filtered to %d rows with kinds=%s", len(filtered), sorted(kinds))
    return filtered


def prepare_labels(df: pd.DataFrame, label_type: str, value_bins: int) -> pd.DataFrame:
    """Add label columns to dataframe."""
    df = df.copy()
    df["digits"] = df["digits"].astype(int)

    if "prompt" not in df.columns:
        df["prompt"] = ""

    src_text = df["prompt"].astype(str)

    # Create both theta_new and gamma labels
    df["theta_new"] = df.apply(lambda row: create_theta_new_label(row["kind"], int(row["digits"])), axis=1)
    df["gamma"] = [make_gamma_label(k, int(d), t, K_bins=value_bins) for k, d, t in zip(df["kind"].astype(str), df["digits"].astype(int), src_text)]

    # Select the target label
    label_col = {"theta_new": "theta_new", "gamma": "gamma", "kind": "kind"}[label_type]
    df["label"] = df[label_col].astype(str)

    return df


# ------------------------------------------------------------------------------
# Data splitting
# ------------------------------------------------------------------------------


def stratified_split_robust(
    df: pd.DataFrame,
    y_col: str = "label",
    test_size: float = 0.2,
    seed: int = 0,
    min_count: int = 2,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split with robust handling of rare classes.

    Drops classes with fewer than min_count examples and falls back
    to non-stratified split if necessary.
    """
    y = df[y_col].astype(str).values
    cnt = Counter(y)
    keep_mask = df[y_col].map(cnt).ge(min_count)
    dropped = int((~keep_mask).sum())

    if dropped and verbose:
        logger.info(f"Dropping {dropped} samples from classes with <{min_count} total examples.")

    df = df[keep_mask].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("All samples dropped due to rare classes; try lowering bin granularity (--value-bins).")

    ts = float(test_size)
    for _ in range(6):
        try:
            tr, te = train_test_split(df, test_size=ts, random_state=seed, stratify=df[y_col])
            return tr, te
        except ValueError as e:
            ts *= 0.5
            if verbose:
                logger.warning(f"Stratified split failed ({e}); retrying with test_size={ts:.4f}")
            if ts < 0.02:
                break

    if verbose:
        logger.warning("Falling back to non-stratified split.")
    return cast(Tuple[pd.DataFrame, pd.DataFrame], train_test_split(df, test_size=test_size, random_state=seed, shuffle=True))
