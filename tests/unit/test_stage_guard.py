from dataclasses import dataclass, field
from typing import Any, List

from src.exps_performance.logger import CheckpointManager, Record
from src.exps_performance.main import run_stage_batch
from src.exps_performance.problems import Question


@dataclass
class DummyArgs:
    checkpoint_every: int = 2
    model: str = "m"
    seed: int = 1


@dataclass
class DummyQuestion(Question):
    kind: str = "add"
    digits: int = 2
    record: Record = field(default_factory=Record)

    @property
    def util_pointer(self) -> Any:  # pragma: no cover - not used in these tests
        return None


class FakeArmPartial:
    def __init__(self, data_subset: List[DummyQuestion], default_args: Any, client: Any) -> None:
        self.problems = data_subset

    def run(self) -> tuple[float, List[DummyQuestion]]:
        # Leave stage fields empty so completion fails.
        return 0.0, self.problems


class FakeArmComplete:
    def __init__(self, data_subset: List[DummyQuestion], default_args: Any, client: Any) -> None:
        self.problems = data_subset

    def run(self) -> tuple[float, List[DummyQuestion]]:
        for q in self.problems:
            q.record.code_question = "q"
            q.record.code_answer = "a"
        return 1.0, self.problems


def test_incomplete_batch_not_checkpointed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    args = DummyArgs(checkpoint_every=2)
    ckpt = CheckpointManager(str(tmp_path / "res.jsonl"))
    q = DummyQuestion(record=Record(unique_tag="t1", request_id="r1", kind="add", digit=2))

    updated = run_stage_batch([q], FakeArmPartial, "Arm3", args, client=None, checkpoint=ckpt)

    assert updated == []
    assert ckpt.all_records() == []


def test_complete_batch_checkpointed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    args = DummyArgs(checkpoint_every=2)
    ckpt = CheckpointManager(str(tmp_path / "res.jsonl"))
    q = DummyQuestion(record=Record(unique_tag="t1", request_id="r1", kind="add", digit=2))

    updated = run_stage_batch([q], FakeArmComplete, "Arm3", args, client=None, checkpoint=ckpt)

    assert len(updated) == 1
    assert len(ckpt.all_records()) == 1
    stored = ckpt.get("t1")
    assert stored is not None
    assert stored.code_question == "q"
    assert stored.code_answer == "a"
