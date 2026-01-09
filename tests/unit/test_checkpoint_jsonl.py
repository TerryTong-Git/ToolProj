from dataclasses import dataclass, field
from typing import Any, List

from src.exps_performance.logger import (
    CheckpointManager,
    Record,
    generate_unique_tag,
    make_request_id,
    write_to_csv,
)
from src.exps_performance.main import assign_sequential_indices
from src.exps_performance.problems import Question


@dataclass
class DummyQuestion(Question):
    kind: str = "dummy"
    digits: int = 0
    code: str = ""
    question: str = ""
    answer: str = ""
    record: Record = field(default_factory=Record)

    @property
    def util_pointer(self) -> Any:
        return None


def _make_record(kind: str, digit: int, idx: int, model: str = "m", seed: int = 1) -> Record:
    return Record(
        kind=kind,
        digit=digit,
        index_in_kind=idx,
        model=model,
        seed=seed,
        unique_tag=generate_unique_tag(kind, digit, idx, seed, model),
        request_id=make_request_id(kind, digit, idx, seed, model),
    )


def test_generate_unique_tag_is_deterministic() -> None:
    tag1 = generate_unique_tag("add", 2, 1, 3, "m")
    tag2 = generate_unique_tag("add", 2, 1, 3, "m")
    tag3 = generate_unique_tag("add", 2, 2, 3, "m")
    assert tag1 == tag2
    assert tag1 != tag3


def test_assign_sequential_indices_orders_and_caps(tmp_path) -> None:  # type: ignore[no-untyped-def]
    checkpoint = CheckpointManager(str(tmp_path / "res.jsonl"))
    questions: List[DummyQuestion] = [
        DummyQuestion(kind="add", digits=4, question="q3", answer="a3"),
        DummyQuestion(kind="add", digits=2, question="q1", answer="a1"),
        DummyQuestion(kind="add", digits=4, question="q2", answer="a2"),
    ]
    for i, q in enumerate(questions):
        setattr(q, "original_pos", i)
    assigned, dropped, restored, debug = assign_sequential_indices(questions, n=2, seed=7, model="gemma", exp_id="exp", checkpoint=checkpoint)

    assert len(assigned) == 2
    assert [q.record.index_in_kind for q in assigned] == [1, 2]
    assert [q.record.digit for q in assigned] == [2, 4]
    assert dropped == {"add": 1}
    assert restored == {}
    assert debug  # sample entries captured
    assert all(q.record.unique_tag for q in assigned)


def test_checkpoint_save_and_load_batch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "res.jsonl"
    manager = CheckpointManager(str(path))
    records = [_make_record("add", 2, 1), _make_record("sub", 4, 1)]
    manager.save_batch(records, flush=True)

    reloaded = CheckpointManager(str(path))
    assert len(reloaded.all_records()) == 2
    for rec in records:
        loaded = reloaded.get(rec.unique_tag)
        assert loaded is not None
        assert loaded.unique_tag == rec.unique_tag


def test_checkpoint_upsert_latest_wins(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "res.jsonl"
    manager = CheckpointManager(str(path))
    base = _make_record("add", 2, 1)
    manager.save_batch([base], flush=True)

    updated = base.model_copy(update={"nl_answer": "new", "nl_correct": True})
    manager.upsert(updated, flush=True)

    reloaded = CheckpointManager(str(path))
    stored = reloaded.get(base.unique_tag)
    assert stored is not None
    assert stored.nl_answer == "new"
    assert stored.nl_correct is True


def test_checkpoint_loads_legacy_csv_and_writes_jsonl(tmp_path) -> None:  # type: ignore[no-untyped-def]
    jsonl_path = tmp_path / "res.jsonl"
    csv_path = tmp_path / "res.csv"
    records = [_make_record("add", 2, 1), _make_record("add", 4, 2)]
    write_to_csv(str(csv_path), records)

    manager = CheckpointManager(str(jsonl_path))
    assert len(manager.all_records()) == len(records)

    # Trigger a new append to ensure JSONL is created and latest record is readable.
    appended = records[0].model_copy(update={"sim_answer": "latest"})
    manager.upsert(appended, flush=True)
    assert jsonl_path.exists()

    reloaded = CheckpointManager(str(jsonl_path))
    stored = reloaded.get(appended.unique_tag)
    assert stored is not None
    assert stored.sim_answer == "latest"


def test_save_batch_merges_updates_and_persists(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """
    Restored records updated in-place should keep prior fields and persist new ones.
    """
    path = tmp_path / "res.jsonl"
    manager = CheckpointManager(str(path))

    base = _make_record("add", 2, 1)
    base = base.model_copy(
        update={
            "sim_question": "sq",
            "sim_answer": "sim-old",
            "code_question": "cq-old",
            "code_answer": "code-old",
        }
    )
    manager.save_batch([base], flush=True)

    restored = CheckpointManager(str(path))
    updated = Record(
        kind=base.kind,
        digit=base.digit,
        index_in_kind=base.index_in_kind,
        model=base.model,
        seed=base.seed,
        unique_tag=base.unique_tag,
        request_id=base.request_id,
        code_question="cq-new",
        code_answer="code-new",
        # sim fields intentionally left empty to ensure they are preserved.
    )
    restored.save_batch([updated], flush=True)

    reloaded = CheckpointManager(str(path))
    stored = reloaded.get(base.unique_tag)
    assert stored is not None
    assert stored.sim_answer == "sim-old", "sim fields should be preserved"
    assert stored.code_answer == "code-new", "code updates should be persisted"
