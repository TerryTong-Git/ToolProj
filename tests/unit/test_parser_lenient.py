from typing import Tuple

from src.exps_performance.problems.finegrained import FgCheckAndFormat


def _fg_util() -> FgCheckAndFormat:
    return FgCheckAndFormat("code")


def _parse(text: str) -> Tuple[str, str]:
    parsed, err = _fg_util().parse_output(text)
    return parsed.Answer, err


def test_parse_handles_trailing_comma_and_fence() -> None:
    answer, err = _parse('```json\n{"Answer": "2", "simulation": "ok",}\n```')
    assert answer == "2"
    assert err == "ok"


def test_parse_handles_single_quotes_and_noise() -> None:
    noisy = "prefix {'Answer': '3', 'simulation': 'ok'} suffix"
    answer, err = _parse(noisy)
    assert answer == "3"
    assert err == "ok"


def test_parse_returns_default_on_bad_payload() -> None:
    util = _fg_util()
    parsed, err = util.parse_output("not json at all")
    assert parsed == util.PROB_TYPES["code"]()
    assert err != "ok"
