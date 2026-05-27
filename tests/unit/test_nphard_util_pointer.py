import pytest

from src.reasoning_benchmark.problems.nphard.bsp import BspCheckAndFormat, BspQuestion
from src.reasoning_benchmark.problems.nphard.edp import EdpCheckAndFormat, EdpQuestion
from src.reasoning_benchmark.problems.nphard.gcp import GcpCheckAndFormat, GcpQuestion
from src.reasoning_benchmark.problems.nphard.gcp_d import GcpdCheckAndFormat, GcpdQuestion
from src.reasoning_benchmark.problems.nphard.ksp import KspCheckAndFormat, KspQuestion
from src.reasoning_benchmark.problems.nphard.msp import MspCheckAndFormat, MspQuestion
from src.reasoning_benchmark.problems.nphard.spp import SppCheckAndFormat, SppQuestion
from src.reasoning_benchmark.problems.nphard.tsp import TspCheckAndFormat, TspQuestion
from src.reasoning_benchmark.problems.nphard.tsp_d import TspdCheckAndFormat, TspdQuestion
from src.reasoning_benchmark.problems.nphardeval import NpCheckAndFormat, NpQuestion


@pytest.mark.parametrize(
    ("question_cls", "util_cls"),
    [
        (SppQuestion, SppCheckAndFormat),
        (MspQuestion, MspCheckAndFormat),
        (KspQuestion, KspCheckAndFormat),
        (GcpQuestion, GcpCheckAndFormat),
        (GcpdQuestion, GcpdCheckAndFormat),
        (EdpQuestion, EdpCheckAndFormat),
        (BspQuestion, BspCheckAndFormat),
        (TspQuestion, TspCheckAndFormat),
        (TspdQuestion, TspdCheckAndFormat),
    ],
)
def test_util_pointer_property_returns_class_and_instantiates(question_cls: type[NpQuestion], util_cls: type[NpCheckAndFormat]) -> None:
    q = question_cls()

    # util_pointer should be a property returning the CheckAndFormat class
    assert q.util_pointer is util_cls

    # And the returned class must be callable with the prob_type (run_type) argument
    instance = q.util_pointer("code")
    assert isinstance(instance, util_cls)
