from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.exps_performance.logger import (
    CheckpointManager,
    Record,
    generate_unique_tag,
    make_request_id,
)
from src.exps_performance.main import assign_sequential_indices
from src.exps_performance.problems import Question


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


@dataclass
class _StubQuestion(Question):
    kind: str
    digits: int
    code: str = ""
    question: str = ""
    answer: str = ""
    record: Record = field(default_factory=Record)

    @property
    def util_pointer(self) -> Any:
        return None


def test_pending_counts_ignore_other_models(tmp_path: Any) -> None:
    """
    Pending counts should ignore checkpoint rows that do not match the current model/seed.
    """
    path = tmp_path / "res.jsonl"
    ckpt = CheckpointManager(str(path))
    for i in range(25):
        ckpt.upsert(
            Record(
                request_id=f"old-{i}",
                unique_tag=f"old-{i}",
                model="old-model",
                seed=0,
                kind="add",
                digit=2,
                index_in_kind=i + 1,
            ),
            flush=False,
        )
    ckpt.flush()
    ckpt = CheckpointManager(str(path))

    pending = ckpt.get_pending_count({"add": 1}, digits=[2], model="new-model", seed=0)
    assert pending["add"] == 1


def test_assign_indices_stable_and_respects_n(tmp_path: Any) -> None:
    """
    Sequential indices should be stable (digit, original_pos) and drop out-of-scope items.
    """
    ckpt_path = tmp_path / "res.jsonl"
    ckpt = CheckpointManager(str(ckpt_path))
    ckpt.upsert(
        Record(
            request_id=make_request_id("add", 2, 1, 0, "old-model"),
            unique_tag=generate_unique_tag("add", 2, 1, 0, "old-model"),
            model="old-model",
            seed=0,
            kind="add",
            digit=2,
            index_in_kind=1,
        ),
        flush=True,
    )

    questions = [
        _StubQuestion(kind="add", digits=4, record=Record()),
        _StubQuestion(kind="add", digits=2, record=Record()),
        _StubQuestion(kind="add", digits=4, record=Record()),
    ]
    for i, q in enumerate(questions):
        setattr(q, "original_pos", i)

    assigned, dropped_by_kind, restored_by_kind, _ = assign_sequential_indices(
        questions, n=2, seed=0, model="new-model", exp_id="exp", checkpoint=ckpt
    )

    assert len(assigned) == 2
    assert dropped_by_kind.get("add", 0) == 1
    assert restored_by_kind == {}
    assert assigned[0].digits == 2
    assert assigned[0].record.index_in_kind == 1
    assert assigned[0].record.unique_tag == generate_unique_tag("add", 2, 1, 0, "new-model")
    assert assigned[1].digits == 4
    assert assigned[1].record.index_in_kind == 2
    assert assigned[1].record.unique_tag == generate_unique_tag("add", 4, 2, 0, "new-model")


def test_jsonl_serialization_uses_distinct_records(tmp_path: Any) -> None:
    """
    JSONL serialization should not duplicate a single record across questions.
    """
    ckpt_path = tmp_path / "res.jsonl"
    ckpt = CheckpointManager(str(ckpt_path))

    questions = [
        _StubQuestion(kind="add", digits=2),
        _StubQuestion(kind="sub", digits=2),
    ]
    for i, q in enumerate(questions):
        setattr(q, "original_pos", i)

    assigned, _, _, _ = assign_sequential_indices(questions, n=5, seed=1, model="m", exp_id="exp", checkpoint=ckpt)
    ckpt.save_batch([q.record for q in assigned], flush=True)

    reloaded = CheckpointManager(str(ckpt_path))
    assert len(reloaded.all_records()) == 2
    tags = {rec.unique_tag for rec in reloaded.all_records()}
    assert len(tags) == 2
