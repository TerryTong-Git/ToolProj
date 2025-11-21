from src.exps_performance.dataset import NPHARD, make_dataset
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


def test_np():
    data = NPHARD().load()
    assert data is not None, "no data"


def test_make():
    for p, probclass in problem_types.items():
        data = make_dataset([p])
        for d in data:
            assert isinstance(d, probclass), "didn't choose right class"
    data = make_dataset(["clrs", "spp"])
    for d in data:
        assert isinstance(d, ClrsQuestion) or isinstance(d, SppQuestion), "didn't choose right class"
