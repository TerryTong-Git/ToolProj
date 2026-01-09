#!/usr/bin/env python3
"""Data quality tests for mutual information pipelines using CoT rationales."""

import json
import tempfile
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from src.exps_performance.logger import create_big_df


def _is_ok(err: str) -> bool:
    """Return True if error marker indicates a successful parse/generation."""
    return str(err).strip().lower() in {"", "ok", "ok,ok"}


def write_mock_records(tmpdir: Path, records: Iterable[dict]) -> Path:
    """
    Write mock records (Record schema) to a JSONL file and return the path.

    The schema matches src/exps_performance/logger.Record so tests exercise
    the same loading path used in analysis.py via create_big_df.
    """
    tmpdir.mkdir(parents=True, exist_ok=True)
    path = tmpdir / "mock.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec))
            f.write("\n")
    return path


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Record-style dataframe to canonical rationale rows, filtering:
    - rationale text must be non-empty
    - all err_msg values in the record must be '', 'ok', or 'ok,ok'
    """
    rows: List[dict] = []
    for _, row in df.iterrows():
        nl_err_ok = _is_ok(str(row.get("nl_err_msg", "")))
        code_err_ok = _is_ok(str(row.get("code_err_msg", "")))
        # Drop the entire record if either modality reports an error to avoid
        # mixing partial failures into the MI dataset.
        if not nl_err_ok or not code_err_ok:
            continue
        base = {"kind": row["kind"], "digits": int(row["digit"]), "prompt": row.get("question", "")}

        nl_reasoning = str(row.get("nl_reasoning", "") or "").strip()
        if nl_reasoning and nl_err_ok:
            rows.append({**base, "rationale": nl_reasoning, "rep": "nl"})

        code_answer = str(row.get("code_answer", "") or "").strip()
        if code_answer and code_err_ok:
            rows.append({**base, "rationale": code_answer, "rep": "code"})

    return pd.DataFrame(rows)


def _balanced_counts(count_a: int, count_b: int, tol: float = 0.2) -> bool:
    """Check approximate balance within a fractional tolerance."""
    if count_a == 0 and count_b == 0:
        return True
    if count_a == 0 or count_b == 0:
        return False
    diff = abs(count_a - count_b)
    return diff / max(count_a, count_b) < tol


class TestMutualInfoDataQuality:
    """Quality checks over CoT rationales used for MI estimation."""

    def test_rationales_non_empty(self) -> None:
        """Ensure rationales used are non-empty after conversion."""
        good = {
            "digit": 2,
            "kind": "add",
            "question": "Q",
            "nl_reasoning": "step by step",
            "nl_err_msg": "ok",
            "code_answer": "print(1+1)",
            "code_err_msg": "",
        }
        empty_nl = {**good, "nl_reasoning": ""}
        empty_code = {**good, "code_answer": "   "}

        with tempfile.TemporaryDirectory() as td:
            path = write_mock_records(Path(td), [good, empty_nl, empty_code])
            df_raw = create_big_df([path])
            df = to_canonical(df_raw)

        assert not df.empty
        assert (df["rationale"].str.len() > 0).all()
        assert set(df["rep"]) == {"nl", "code"}

    def test_rationales_not_error_messages(self) -> None:
        """Ensure records with error markers are excluded."""
        base = {
            "digit": 3,
            "kind": "mul",
            "question": "Q",
            "nl_reasoning": "valid nl",
            "nl_err_msg": "",
            "code_answer": "valid code",
            "code_err_msg": "",
        }
        bad_nl = {**base, "nl_err_msg": "parse_failed"}
        bad_code = {**base, "code_err_msg": "timeout"}

        with tempfile.TemporaryDirectory() as td:
            path = write_mock_records(Path(td), [base, bad_nl, bad_code])
            df_raw = create_big_df([path])
            df = to_canonical(df_raw)

        reps = df["rep"].tolist()
        assert "nl" in reps and "code" in reps
        # Bad error rows should have been dropped
        assert len(df) == 2
        assert (df["rationale"].str.len() > 0).all()

    def test_balanced_representation_counts(self) -> None:
        """Check nl and code samples remain approximately balanced."""
        records = []
        for i in range(5):
            records.append(
                {
                    "digit": i % 3 + 1,
                    "kind": "knap",
                    "question": f"Q{i}",
                    "nl_reasoning": f"nl rationale {i}",
                    "nl_err_msg": "ok",
                    "code_answer": f"code answer {i}",
                    "code_err_msg": "",
                }
            )
        # Add one extra nl with an error to ensure drop keeps balance
        records.append(
            {
                "digit": 9,
                "kind": "knap",
                "question": "Q_bad",
                "nl_reasoning": "bad nl",
                "nl_err_msg": "error",
                "code_answer": "bad code",
                "code_err_msg": "error",
            }
        )

        with tempfile.TemporaryDirectory() as td:
            path = write_mock_records(Path(td), records)
            df_raw = create_big_df([path])
            df = to_canonical(df_raw)

        nl_count = (df["rep"] == "nl").sum()
        code_count = (df["rep"] == "code").sum()
        assert _balanced_counts(nl_count, code_count, tol=0.2)
