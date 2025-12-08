from typing import Any

import pandas as pd

from src.exps_performance.logger import CheckpointManager, Record


def test_res_csv_counts_match_requested_n_with_resume(tmp_path: Any) -> None:
    """
    Simulate a crash/restart: write half the rows, flush, reload checkpoint,
    then write the remainder. Ensure counts are exactly n per kind (no duplicates).
    """
    kinds = ["add", "sub", "mul"]
    n = 25
    digits = [2, 4]

    csv_path = tmp_path / "res.csv"

    def make_record(kind: str, i: int) -> Record:
        return Record(
            request_id=f"{kind}-{i}",
            model="test-model",
            seed=0,
            exp_id="run_test",
            digit=digits[i % len(digits)],
            kind=kind,
            question="q",
            answer="a",
            sim_code="print('hi')",
        )

    # First run writes half then "crashes"
    ckpt = CheckpointManager(str(csv_path))
    for kind in kinds:
        for i in range(n // 2):
            ckpt.upsert(make_record(kind, i), flush=False)
    ckpt.flush()

    # Simulate restart: reload from disk and continue
    ckpt = CheckpointManager(str(csv_path))
    for kind in kinds:
        # Re-send already written rows (should not duplicate) and remaining rows
        for i in range(n):
            ckpt.upsert(make_record(kind, i), flush=False)
    ckpt.flush()

    df = pd.read_csv(csv_path)
    for kind in kinds:
        count = (df["kind"] == kind).sum()
        assert count == n, f"kind {kind} count mismatch after resume (got {count}, expected {n})"
