import json
import re
from dataclasses import dataclass, field
from typing import Any, List

import pytest
from pydantic import BaseModel, Field

import src.exps_performance.arms as arms
from src.exps_performance.arms import Arm2, Arm3
from src.exps_performance.logger import Record
from src.exps_performance.problems import CheckAndFormat, Question


class ToyAnswer(BaseModel):
    value: int = Field(default=0, description="Deterministic stub value.")


class ToyCheckAndFormat(CheckAndFormat):
    def __init__(self, prob_type: str):
        super().__init__(prob_type, "int", "Toy problem", ToyAnswer)

    def decision_check(self, q: "ToyQuestion", output: ToyAnswer) -> tuple[bool, str]:
        return output.value == q.target, ""

    def parse_output(self, output: Any) -> tuple[BaseModel, str]:
        val = 0
        if isinstance(output, str):
            try:
                loaded = json.loads(output)
                if isinstance(loaded, dict) and "answer" in loaded:
                    val = int(loaded["answer"])
                else:
                    val = int(output)
            except Exception:
                match = re.search(r"-?\d+", output)
                if match:
                    val = int(match.group(0))
        elif isinstance(output, (int, float)):
            val = int(output)
        parsed = self.PROB_TYPES[self.prob_type](value=val, simulation="ok")
        return parsed, "ok"

    def type_check_code(self, code: str) -> bool:
        return str(code).strip().lstrip("-").isdigit()

    def get_field_kwargs(self, code: str) -> dict[str, int]:
        return {"value": int(code)}

    def format_one(self, q: "ToyQuestion") -> str:
        return f"target={q.target}"

    def load_data(self) -> List[Any]:
        return []


@dataclass
class ToyQuestion(Question):
    target: int = 0
    question: str = ""
    answer: str = "0"
    code: str = "0"
    kind: str = "toy"
    digits: int = 1
    record: Record = field(default_factory=Record)

    def util_pointer(self, prob_type: str) -> ToyCheckAndFormat:  # type: ignore[override]
        return ToyCheckAndFormat(prob_type)


@pytest.fixture
def toy_questions() -> list[ToyQuestion]:
    return [
        ToyQuestion(target=1, answer="1", question="one", code="1"),
        ToyQuestion(target=0, answer="0", question="zero", code="0"),
    ]


@pytest.fixture
def stub_run_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_batch(messages_list: list[list[dict[str, str]]], args: Any, client: Any) -> list[str]:
        outs: list[str] = []
        for msgs in messages_list:
            content = msgs[-1]["content"]
            match = re.search(r"target=(-?\d+)", content)
            outs.append(str(int(match.group(1)) if match else 0))
        return outs

    monkeypatch.setattr(arms, "run_batch", _fake_run_batch)


@pytest.fixture
def stub_program_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyExecutor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def run(self, code: str) -> tuple[str, str]:
            return code, ""

    monkeypatch.setattr(arms, "ProgramChatInterface", _DummyExecutor)


def test_arm2_fast(default_args: Any, toy_questions: list[ToyQuestion], stub_run_batch: None) -> None:
    arm2 = Arm2(list(toy_questions), default_args, client=None)
    accuracy, edited = arm2.run()
    assert edited == arm2.edited_problems
    assert accuracy == 1.0
    assert arm2.parse_fail == 0
    for q in edited:
        assert q.record.sim_answer != ""
        assert q.record.sim_question != ""


def test_arm3_fast(default_args: Any, toy_questions: list[ToyQuestion], stub_program_executor: None) -> None:
    arm3 = Arm3(list(toy_questions), default_args, client=None)
    accuracy, edited = arm3.run()
    assert edited == arm3.edited_problems
    assert accuracy == 1.0
    assert arm3.parse_fail == 0
    for q in edited:
        assert q.record.code_answer != ""
        assert q.record.code_question != ""
