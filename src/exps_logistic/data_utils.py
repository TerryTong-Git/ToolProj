#!/usr/bin/env python3

import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Set, Tuple, cast

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exps_performance.logger import create_big_df

from .config import EXTENDED_KINDS
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


def filter_algorithm_names(text: str, algorithm_names: Set[str]) -> str:
    for name in algorithm_names:
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        text = pattern.sub("", text)
    return text


def strip_comments(text: str) -> str:
    text = re.sub(r'"""[\s\S]*?"""', '', text)
    text = re.sub(r"'''[\s\S]*?'''", '', text)
    text = re.sub(r'#[^\n]*', '', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def equal_width_bin(x: int, lo: int, hi: int, K: int) -> int:
    if K <= 1:
        return 0
    lo, hi = int(lo), int(hi)
    if lo > hi:
        lo, hi = hi, lo
    x = max(lo, min(hi, int(x)))
    span = hi - lo + 1
    idx = ((x - lo) * K) // span
    return max(0, min(K - 1, int(idx)))


def make_gamma_label(kind: str, digits: int, problem_text: str, K_bins: int = 8, use_joint_id: bool = True) -> str:
    k = str(kind)
    d = int(digits)
    t = problem_text or ""

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

    if k == "lcs":
        Ls, Lt = parse_lcs_lengths(t, d)
        bLs = equal_width_bin(Ls, 1, max(2, 2 * d), K_bins)
        bLt = equal_width_bin(Lt, 1, max(2, 2 * d), K_bins)
        return f"{k}|d{d}|b{bLs * K_bins + bLt}"

    if k == "knap":
        n_items, cap_ratio = parse_knap_stats(t, d)
        bN = equal_width_bin(n_items, 1, max(3, 2 * d), K_bins)
        bR = equal_width_bin(int(round(cap_ratio * 1000)), 0, 1000, K_bins)
        return f"{k}|d{d}|b{bN * K_bins + bR}"

    if k == "rod":
        N = parse_rod_N(t, d)
        bN = equal_width_bin(N, 1, max(2, 2 * d), K_bins)
        return f"{k}|d{d}|b{bN}"

    if k == "ilp_assign":
        n = parse_ilp_assign_n(t, d)
        bN = equal_width_bin(n, 2, 7, K_bins)
        return f"{k}|d{d}|b{bN}"

    if k == "ilp_prod":
        P, R = parse_ilp_prod_PR(t, d)
        bP = equal_width_bin(P, 2, 6, K_bins)
        bR = equal_width_bin(R, 2, 4, K_bins)
        return f"{k}|d{d}|b{bP * K_bins + bR}"

    if k == "ilp_partition":
        n_items = parse_ilp_partition_n(t, d)
        bN = equal_width_bin(n_items, 4, 24, K_bins)
        return f"{k}|d{d}|b{bN}"

    return f"{k}|d{d}|bNA"


def create_theta_new_label(kind: str, digits: int) -> str:
    return f"{kind}__d{digits}"


def _convert_results_df(df: pd.DataFrame, filter_algo: bool = True, filter_comments: bool = True) -> pd.DataFrame:
    required = {"digit", "kind", "question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results data missing required columns {missing}. Got: {df.columns.tolist()}")

    rows = []
    for _, row in df.iterrows():
        nl = str(row.get("nl_reasoning", "") or "").strip()
        code = str(row.get("sim_code", "") or "").strip()
        sim_reasoning = str(row.get("sim_reasoning", "") or "").strip()

        if filter_comments:
            code = strip_comments(code)
            sim_reasoning = strip_comments(sim_reasoning)

        if filter_algo:
            nl = filter_algorithm_names(nl, EXTENDED_KINDS)
            code = filter_algorithm_names(code, EXTENDED_KINDS)
            sim_reasoning = filter_algorithm_names(sim_reasoning, EXTENDED_KINDS)

        # Skip if no rationale text available at all
        if not nl and not code and not sim_reasoning:
            continue

        digits = int(row["digit"])
        base = {
            "kind": row["kind"],
            "digits": digits,
            "prompt": row.get("question", ""),
        }

        # Add each representation if non-empty
        if nl:
            rows.append({**base, "rationale": nl, "rep": "nl"})
        if code:
            rows.append({**base, "rationale": code, "rep": "code"})
        if sim_reasoning:
            rows.append({**base, "rationale": sim_reasoning, "rep": "sim_reasoning"})

    if not rows:
        raise ValueError("No rationale text found in results CSV (nl_reasoning/sim_code/sim_reasoning were empty).")

    return pd.DataFrame(rows)


def load_data(
    results_dir: str,
    models: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    filter_algo_names: bool = True,
    filter_comments: bool = True,
) -> pd.DataFrame:
    root = Path(results_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    jsonl_files = sorted(root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL result files found under {results_dir}")

    df = create_big_df(jsonl_files)
    if df.empty:
        raise ValueError(f"No rows loaded from JSONL files under {results_dir}")

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

    if "rationale" in df.columns and ("digits" in df.columns or "digit" in df.columns):
        if "digit" in df.columns and "digits" not in df.columns:
            df = df.rename(columns={"digit": "digits"})
        return df

    if {"digit", "kind", "question"} <= set(df.columns):
        converted = _convert_results_df(df, filter_algo=filter_algo_names, filter_comments=filter_comments)
        logger.info(
            "Loaded Record-schema results; converted %d rows to %d rationale entries (nl/code/sim_reasoning).",
            len(df),
            len(converted),
        )
        return converted

    raise ValueError(
        "Unsupported results format. Expected columns {rationale, kind, digits} or Record schema "
        f"(digit, kind, question, nl_reasoning/sim_code/sim_reasoning). Got: {df.columns.tolist()}"
    )


def filter_by_rep(df: pd.DataFrame, rep: str) -> pd.DataFrame:
    if rep == "all" or "rep" not in df.columns:
        return df
    filtered = df[df["rep"].astype(str).str.lower() == rep]
    if len(filtered) == 0:
        raise ValueError(f"No rows after filtering rep={rep}")
    return filtered


def filter_by_kinds(df: pd.DataFrame, kinds: Optional[set]) -> pd.DataFrame:
    if kinds is None or "kind" not in df.columns:
        return df
    filtered = df[df["kind"].isin(kinds)]
    if len(filtered) == 0:
        raise ValueError(f"No rows after filtering by kinds={kinds}")
    logger.info("Filtered to %d rows with kinds=%s", len(filtered), sorted(kinds))
    return filtered


def prepare_labels(df: pd.DataFrame, label_type: str, value_bins: int) -> pd.DataFrame:
    df = df.copy()
    df["digits"] = df["digits"].astype(int)

    if "prompt" not in df.columns:
        df["prompt"] = ""

    src_text = df["prompt"].astype(str)

    df["theta_new"] = df.apply(lambda row: create_theta_new_label(row["kind"], int(row["digits"])), axis=1)
    df["gamma"] = [make_gamma_label(k, int(d), t, K_bins=value_bins) for k, d, t in zip(df["kind"].astype(str), df["digits"].astype(int), src_text)]

    label_col = {"theta_new": "theta_new", "gamma": "gamma", "kind": "kind"}[label_type]
    df["label"] = df[label_col].astype(str)

    return df


def stratified_split_robust(
    df: pd.DataFrame,
    y_col: str = "label",
    test_size: float = 0.2,
    seed: int = 0,
    min_count: int = 2,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
