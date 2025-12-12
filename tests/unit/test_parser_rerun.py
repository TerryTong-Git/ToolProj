from typing import Any

import pytest

from src.exps_performance.arms import RERUN, Arm2
from src.exps_performance.problems.finegrained import FgQuestion


def _fg_question(answer: str = "2") -> FgQuestion:
    return FgQuestion(kind="finegrained", digits=1, question="Compute: 1 + 1", answer=answer)


def test_parser_reruns_until_valid_json(monkeypatch: pytest.MonkeyPatch, default_args: Any) -> None:
    calls: list[int] = []
    question = _fg_question("2")

    def _fake_run_batch(messages_list: list[list[dict[str, str]]], args: Any, client: Any) -> list[str]:
        calls.append(len(messages_list))
        if len(messages_list) == 1:
            return ['{"Answer": "2", bad}']  # malformed JSON to force parse failure
        return [
            '{"Answer": "2", "simulation": "ok",}',  # trailing comma -> still a parse error
            '{"Answer": "2", "simulation": "ok"}',  # succeeds on second attempt
            '{"Answer": "2", "simulation": "ok"}',
        ]

    monkeypatch.setattr("src.exps_performance.arms.run_batch", _fake_run_batch)

    arm = Arm2([question], default_args, client=None)
    accuracy, edited = arm.run()

    assert calls == [1, RERUN]
    assert arm.reparse_ind == [0]
    assert arm.parse_fail == 1  # first attempt failed
    assert accuracy == 1.0
    assert edited[0].record.sim_parse_err is False
    assert arm.parsed_answer[0].Answer == "2"


def test_parser_stops_after_rerun_budget(monkeypatch: pytest.MonkeyPatch, default_args: Any) -> None:
    calls: list[int] = []
    question = _fg_question("4")

    def _always_bad(messages_list: list[list[dict[str, str]]], args: Any, client: Any) -> list[str]:
        calls.append(len(messages_list))
        if len(messages_list) == 1:
            return ["Answer: oops"]  # clearly not JSON
        return ["not json", "still bad", "{bad json"]  # all invalid

    monkeypatch.setattr("src.exps_performance.arms.run_batch", _always_bad)

    arm = Arm2([question], default_args, client=None)
    accuracy, edited = arm.run()

    default_model = question.util_pointer("code").PROB_TYPES["code"]()

    assert calls == [1, RERUN]
    assert arm.reparse_ind == [0]
    assert arm.parse_fail == 1
    assert accuracy == 0.0
    parsed = arm.parsed_answer[0]
    if hasattr(parsed, "model_dump"):
        assert parsed.model_dump() == default_model.model_dump()
    else:
        assert parsed.dict() == default_model.dict()
    assert edited[0].record.sim_parse_err is True


def test_arm2_rerun_recovers_from_messy_json(monkeypatch: pytest.MonkeyPatch, default_args: Any) -> None:
    calls: list[int] = []
    question = _fg_question("2")

    def _messy(messages_list: list[list[dict[str, str]]], args: Any, client: Any) -> list[str]:
        calls.append(len(messages_list))
        if len(messages_list) == 1:
            return ["completely not json"]
        return [
            '```json\n{"Answer": "2", "simulation": "ok",}\n```',
            "{'Answer': '2', 'simulation': 'ok'}",
            'prefix {"Answer": "2", "simulation": "ok"} suffix',
        ]

    monkeypatch.setattr("src.exps_performance.arms.run_batch", _messy)

    arm = Arm2([question], default_args, client=None)
    accuracy, edited = arm.run()

    assert calls == [1, RERUN]
    assert arm.reparse_ind == [0]
    assert arm.parse_fail == 1
    assert accuracy == 1.0
    assert edited[0].record.sim_parse_err is False
    assert arm.parsed_answer[0].Answer == "2"
