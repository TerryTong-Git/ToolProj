import pytest

from src.exps_performance.problems.nphard.spp import SppCheckAndFormat
from src.exps_performance.problems.nphard.tsp import TspCheckAndFormat


@pytest.mark.parametrize(
    "util_cls,result,expected_path,expected_total",
    [
        (TspCheckAndFormat, ([0, 3], 12), [0, 3], "12"),
        (TspCheckAndFormat, "([1, 2, 1], 7)", [1, 2, 1], "7"),
        (SppCheckAndFormat, ([4, 5, 6], 9), [4, 5, 6], "9"),
        (SppCheckAndFormat, "([0, 2], 3)", [0, 2], "3"),
    ],
)
def test_path_kwargs_parse_cleanly(util_cls, result, expected_path, expected_total) -> None:
    util = util_cls("code")
    kwargs = util.get_field_kwargs(result)  # should return native types
    model_cls = util.PROB_TYPES["code"]
    parsed = model_cls(**kwargs)

    assert parsed.Path == expected_path
    # TotalDistance is declared as str; coerce to str for a stable assertion
    assert str(parsed.TotalDistance) == expected_total
