import asyncio
import copy
import inspect
import re
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel, Field

import src.exps_performance.arms as arms
from src.exps_performance.arms import ArmRLMCode, ArmRLMNL
from src.exps_performance.logger import CheckpointManager, Record
from src.exps_performance.main import run_stage_batch
from src.exps_performance.problems import CheckAndFormat, Question
from tests.unit.test_probs import ToyCheckAndFormat


@dataclass
class ToyRLMQuestion(Question):
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
def toy_rlm_questions() -> list[ToyRLMQuestion]:
    return [
        ToyRLMQuestion(target=1, answer="1", question="one", code="1"),
        ToyRLMQuestion(target=0, answer="0", question="zero", code="0"),
    ]


@pytest.fixture
def stub_rlm_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Result:
        def __init__(self, response: str, err: str = "ok") -> None:
            self.response = response
            self.err = err

    class _DummyExecutor:
        def __init__(self, args: Any) -> None:
            self.args = args

        def run(self, prompt: str) -> _Result:
            match = re.search(r"target=(-?\d+)", prompt)
            answer = match.group(1) if match else "0"
            return _Result(answer)

        async def arun(self, prompt: str) -> _Result:
            return self.run(prompt)

    monkeypatch.setattr(arms, "RecursiveLMExecutor", _DummyExecutor)


def test_rlm_nl_arm_fast(default_args: Any, toy_rlm_questions: list[ToyRLMQuestion], stub_rlm_executor: None) -> None:
    arm = ArmRLMNL(list(toy_rlm_questions), default_args, client=None)
    accuracy, edited = arm.run()
    assert edited == arm.edited_problems
    assert accuracy == 1.0
    for q in edited:
        assert q.record.rlmnl_answer != ""
        assert q.record.rlmnl_question != ""


def test_rlm_code_arm_fast(default_args: Any, toy_rlm_questions: list[ToyRLMQuestion], stub_rlm_executor: None) -> None:
    arm = ArmRLMCode(list(toy_rlm_questions), default_args, client=None)
    accuracy, edited = arm.run()
    assert edited == arm.edited_problems
    assert accuracy == 1.0
    for q in edited:
        assert q.record.rlmcode_answer != ""
        assert q.record.rlmcode_question != ""


class RetryToyAnswer(BaseModel):
    value: int = Field(default=0, description="Deterministic stub value.")


class RetryToyCheckAndFormat(CheckAndFormat):
    def __init__(self, prob_type: str):
        super().__init__(prob_type, "int", "Retry toy problem", RetryToyAnswer)

    def decision_check(self, q: "RetryToyQuestion", output: BaseModel) -> tuple[bool, str]:
        return getattr(output, "value", 0) == q.target, ""

    def parse_output(self, output: Any) -> tuple[BaseModel, str]:
        model_cls = self.PROB_TYPES[self.prob_type]
        if str(output).strip() == "invalid":
            return model_cls(), "parse_failed"
        match = re.search(r"-?\d+", str(output))
        if not match:
            return model_cls(), "parse_failed"
        kwargs: dict[str, Any] = {"value": int(match.group(0)), "simulation": "ok"}
        if self.prob_type == "code":
            kwargs["code"] = "```python\ndef solution() -> int:\n    return 0\n```"
        return model_cls(**kwargs), "ok"

    def type_check_code(self, code: str) -> bool:
        return True

    def get_field_kwargs(self, code: str) -> dict[str, int]:
        return {"value": int(code)}

    def format_one(self, q: "RetryToyQuestion") -> str:
        return f"retry_target={q.target}"

    def load_data(self) -> list[Any]:
        return []


@dataclass
class RetryToyQuestion(Question):
    target: int = 0
    question: str = ""
    answer: str = "0"
    code: str = "0"
    kind: str = "retry_toy"
    digits: int = 1
    record: Record = field(default_factory=Record)

    def util_pointer(self, prob_type: str) -> RetryToyCheckAndFormat:  # type: ignore[override]
        return RetryToyCheckAndFormat(prob_type)


def test_rlm_retry_updates_stored_answer(default_args: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Result:
        def __init__(self, response: str, err: str = "ok") -> None:
            self.response = response
            self.err = err

    class _DummyExecutor:
        def __init__(self, args: Any) -> None:
            self.args = args
            self.calls: dict[str, int] = {}

        def run(self, prompt: str) -> _Result:
            count = self.calls.get(prompt, 0)
            self.calls[prompt] = count + 1
            if count == 0:
                return _Result("invalid")
            match = re.search(r"retry_target=(-?\d+)", prompt)
            answer = match.group(1) if match else "0"
            return _Result(answer)

        async def arun(self, prompt: str) -> _Result:
            return self.run(prompt)

    monkeypatch.setattr(arms, "RecursiveLMExecutor", _DummyExecutor)

    arm = ArmRLMNL([RetryToyQuestion(target=7, answer="7", question="seven")], default_args, client=None)
    accuracy, edited = arm.run()

    assert accuracy == 1.0
    assert edited[0].record.rlmnl_correct is True
    assert edited[0].record.rlmnl_parse_err is False
    assert edited[0].record.rlmnl_answer == "7"


def test_rlm_examples_run_concurrently(default_args: Any, toy_rlm_questions: list[ToyRLMQuestion], monkeypatch: pytest.MonkeyPatch) -> None:
    active = 0
    max_active = 0
    lock = asyncio.Lock()

    class _Result:
        def __init__(self, response: str, err: str = "ok") -> None:
            self.response = response
            self.err = err

    class _DummyExecutor:
        def __init__(self, args: Any) -> None:
            self.args = args

        async def arun(self, prompt: str) -> _Result:
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.05)
                match = re.search(r"target=(-?\d+)", prompt)
                answer = match.group(1) if match else "0"
                return _Result(answer)
            finally:
                async with lock:
                    active -= 1

    monkeypatch.setattr(arms, "RecursiveLMExecutor", _DummyExecutor)

    args = copy.deepcopy(default_args)
    args.exec_workers = 2

    arm = ArmRLMNL(list(toy_rlm_questions), args, client=None)
    accuracy, _edited = arm.run()

    assert accuracy == 1.0
    assert max_active >= 2


def test_rlm_stage_checkpoints_each_completed_item(
    default_args: Any,
    toy_rlm_questions: list[ToyRLMQuestion],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    upsert_calls: list[tuple[str, bool]] = []
    original_upsert = CheckpointManager.upsert

    def _spy_upsert(self: CheckpointManager, record: Record, flush: bool = True) -> None:
        upsert_calls.append((record.question, flush))
        original_upsert(self, record, flush=flush)

    class _Result:
        def __init__(self, response: str, err: str = "ok") -> None:
            self.response = response
            self.err = err
            self.execution_time = 0.01
            self.metadata = None

    class _DummyExecutor:
        def __init__(self, args: Any) -> None:
            self.args = args

        async def arun_many(
            self,
            prompts: list[str],
            max_concurrent: int = 4,
            progress_desc: str | None = None,
            parent_task_id: int | None = None,
            on_completed: Any = None,
        ) -> list[_Result]:
            results: list[_Result] = []
            for idx, prompt in enumerate(prompts):
                match = re.search(r"target=(-?\d+)", prompt)
                answer = match.group(1) if match else "0"
                result = _Result(answer)
                results.append(result)
                if on_completed is not None:
                    maybe_awaitable = on_completed(idx, result)
                    if inspect.isawaitable(maybe_awaitable):
                        await maybe_awaitable
            return results

    monkeypatch.setattr(arms, "RecursiveLMExecutor", _DummyExecutor)
    monkeypatch.setattr(CheckpointManager, "upsert", _spy_upsert)

    args = copy.deepcopy(default_args)
    args.checkpoint_every = 1
    args.exec_workers = 2
    ckpt = CheckpointManager(str(tmp_path / "res.jsonl"), only_rlm=True)

    updated = run_stage_batch(list(toy_rlm_questions), ArmRLMNL, "ArmRLMNL", args, client=None, checkpoint=ckpt)

    assert len(updated) == 2
    assert len(upsert_calls) == 2
    assert all(flush is True for _question, flush in upsert_calls)
