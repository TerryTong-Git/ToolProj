import pytest

from src.exps_performance.dataset import CLRS, GSM8K, NPHARD, make_dataset
from src.exps_performance.problems.clrs import ClrsQuestion
from src.exps_performance.problems.finegrained import FgQuestion
from src.exps_performance.problems.gsm8k import Gsm8kQuestion
from src.exps_performance.problems.nphard.bsp import BspQuestion
from src.exps_performance.problems.nphard.edp import EdpQuestion
from src.exps_performance.problems.nphard.gcp import GcpQuestion
from src.exps_performance.problems.nphard.gcp_d import GcpdQuestion
from src.exps_performance.problems.nphard.ksp import KspQuestion
from src.exps_performance.problems.nphard.msp import MspQuestion
from src.exps_performance.problems.nphard.spp import SppQuestion
from src.exps_performance.problems.nphard.tsp import TspQuestion
from src.exps_performance.problems.nphard.tsp_d import TspdQuestion

problem_types = {
    "spp": SppQuestion,
    "tsp": TspQuestion,
    "tsp_d": TspdQuestion,
    "msp": MspQuestion,
    "ksp": KspQuestion,
    "gcp": GcpQuestion,
    "gcp_d": GcpdQuestion,
    "bsp": BspQuestion,
    "edp": EdpQuestion,
    "clrs": ClrsQuestion,
    "gsm8k": Gsm8kQuestion,
    "add": FgQuestion,
    "sub": FgQuestion,
    "mul": FgQuestion,
    "lcs": FgQuestion,
    "rod": FgQuestion,
    "knap": FgQuestion,
    "ilp_assign": FgQuestion,
    "ilp_prod": FgQuestion,
    "ilp_partition": FgQuestion,
}


def test_np() -> None:
    data = NPHARD().load()
    assert data is not None, "no data"


def test_make(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        GSM8K,
        "load",
        lambda self: [Gsm8kQuestion(question="q", answer="1") for _ in range(3)],
    )
    monkeypatch.setattr(
        CLRS,
        "load",
        lambda self: [ClrsQuestion(kind="clrs_alg", digits=0, answer="", text_data="") for _ in range(3)],
    )
    for p, probclass in problem_types.items():
        data = make_dataset([p])
        for d in data:
            assert isinstance(d, probclass), "didn't choose right class"
    data = make_dataset(["clrs", "spp"])
    for d in data:
        assert isinstance(d, ClrsQuestion) or isinstance(d, SppQuestion), "didn't choose right class"


def test_fixed_samples_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        GSM8K,
        "load",
        lambda self: [Gsm8kQuestion(question=str(i), answer=str(i)) for i in range(600)],
    )
    monkeypatch.setattr(
        CLRS,
        "load",
        lambda self: [ClrsQuestion(kind=f"clrs_alg_{i%5}", digits=0, answer="", text_data="") for i in range(600)],
    )
    data = make_dataset(["gsm8k", "clrs30"], n=5)
    gsm_count = len([q for q in data if isinstance(q, Gsm8kQuestion)])
    clrs_count = len([q for q in data if isinstance(q, ClrsQuestion)])
    assert gsm_count == 500
    assert clrs_count == 500

    data_override = make_dataset(["gsm8k", "clrs30"], n=5, gsm_samples=10, clrs_samples=12)
    gsm_override = len([q for q in data_override if isinstance(q, Gsm8kQuestion)])
    clrs_override = len([q for q in data_override if isinstance(q, ClrsQuestion)])
    assert gsm_override == 10
    assert clrs_override == 12
