from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from src.reasoning_benchmark.problems import Question
from src.reasoning_benchmark.reasoning_strategies import Arm1, Arm2
from src.reasoning_benchmark.records import Record


@dataclass
class DummyArgs:
    model: str = "openai/gpt-5.4"
    seed: int = 0
    retry_failed_records: bool = True


@dataclass
class DummyQuestion(Question):
    question: str = "q"
    answer: str = "1"
    kind: str = "add"
    digits: int = 2
    code: str = ""
    record: Record = field(default_factory=Record)

    @property
    def util_pointer(self) -> Any:  # pragma: no cover - not used in these tests
        return None


def test_arm2_failed_retry_preserves_existing_successful_answer() -> None:
    q = DummyQuestion(
        record=Record(
            sim_question="old-prompt",
            sim_answer='{"Answer":"1"}',
            sim_reasoning="old reasoning",
            sim_parse_err=False,
            sim_err_msg="ok",
            sim_code="",
        )
    )
    arm = Arm2([q], DummyArgs(), client=None)

    updated = arm.each_record(
        q,
        "",
        (SimpleNamespace(code="", simulation=""), "parse_failed"),
        "new-prompt",
        False,
    )

    assert updated.record.sim_answer == '{"Answer":"1"}'
    assert updated.record.sim_reasoning == "old reasoning"
    assert updated.record.sim_code == ""
    assert updated.record.sim_parse_err is False


def test_arm1_failed_retry_preserves_existing_successful_answer() -> None:
    q = DummyQuestion(
        record=Record(
            nl_question="old-prompt",
            nl_answer='{"Answer":"1"}',
            nl_reasoning="old reasoning",
            nl_parse_err=False,
            nl_err_msg="ok",
        )
    )
    arm = Arm1([q], DummyArgs(), client=None)

    updated = arm.each_record(
        q,
        "",
        (SimpleNamespace(simulation=""), "parse_failed"),
        "new-prompt",
        False,
    )

    assert updated.record.nl_answer == '{"Answer":"1"}'
    assert updated.record.nl_reasoning == "old reasoning"
    assert updated.record.nl_parse_err is False
