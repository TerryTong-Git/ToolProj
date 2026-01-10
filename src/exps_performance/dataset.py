from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Sequence

import torch

from src.exps_performance.problems import Question
from src.exps_performance.problems.clrs import ClrsCheckAndFormat
from src.exps_performance.problems.finegrained import (
    AddCheckAndFormat,
    IlpAssignCheckAndFormat,
    IlpPartitionCheckAndFormat,
    IlpProdCheckAndFormat,
    Knap01CheckAndFormat,
    LcsCheckAndFormat,
    MulCheckAndFormat,
    RodCheckAndFormat,
    SubCheckAndFormat,
)
from src.exps_performance.problems.gsm8k import Gsm8kCheckAndFormat
from src.exps_performance.problems.nphard.bsp import BspCheckAndFormat
from src.exps_performance.problems.nphard.edp import EdpCheckAndFormat
from src.exps_performance.problems.nphard.gcp import GcpCheckAndFormat
from src.exps_performance.problems.nphard.gcp_d import GcpdCheckAndFormat
from src.exps_performance.problems.nphard.ksp import KspCheckAndFormat
from src.exps_performance.problems.nphard.msp import MspCheckAndFormat
from src.exps_performance.problems.nphard.spp import SppCheckAndFormat
from src.exps_performance.problems.nphard.tsp import TspCheckAndFormat
from src.exps_performance.problems.nphard.tsp_d import TspdCheckAndFormat

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.deterministic = True

try:
    torch.set_float32_matmul_precision("high")
except (AttributeError, RuntimeError):
    # AttributeError if method doesn't exist, RuntimeError if CUDA not available
    pass

# data interface

# runner interface -> track statistics


@dataclass
class Dataset(ABC):
    @abstractmethod
    def load(self) -> Sequence[Question]:
        raise NotImplementedError


problem_types = {
    "spp": SppCheckAndFormat,
    "tsp": TspCheckAndFormat,
    "tsp_d": TspdCheckAndFormat,
    "msp": MspCheckAndFormat,
    "ksp": KspCheckAndFormat,
    "gcp": GcpCheckAndFormat,
    "gcp_d": GcpdCheckAndFormat,
    "bsp": BspCheckAndFormat,
    "edp": EdpCheckAndFormat,
}


clrs_problem_types = {
    "clrs": ClrsCheckAndFormat,
    "clrs30": ClrsCheckAndFormat,
}

gsm_problem_types = {
    "gsm8k": Gsm8kCheckAndFormat,
}

fg_problem_types = {
    "add": AddCheckAndFormat,
    "sub": SubCheckAndFormat,
    "mul": MulCheckAndFormat,
    "lcs": LcsCheckAndFormat,
    "rod": RodCheckAndFormat,
    "knap": Knap01CheckAndFormat,
    "ilp_assign": IlpAssignCheckAndFormat,
    "ilp_prod": IlpProdCheckAndFormat,
    "ilp_partition": IlpPartitionCheckAndFormat,
}


@dataclass
class NPHARD(Dataset):
    n: int = 10

    def load(self) -> Sequence[Question]:
        all_data: List[Question] = []
        for ProblemType in problem_types.values():
            classInstance = ProblemType("code")  # type: ignore
            all_data += classInstance.load_data()[: self.n]  # type: ignore[abstract]
        return all_data

    def load_subset(self, subset: List[str]) -> Sequence[Question]:
        for s in subset:
            assert s in list(problem_types.keys()), "invalid subset"
        all_data: List[Question] = []
        for key, ProblemType in problem_types.items():
            if key not in subset:
                continue
            classInstance = ProblemType("code")  # type: ignore
            all_data += classInstance.load_data()[: self.n]  # type: ignore[abstract]
        return all_data


@dataclass
class CLRS(Dataset):
    def load(self) -> Sequence[Question]:
        all_data: List[Question] = []
        for ProblemType in dict.fromkeys(clrs_problem_types.values()):
            classInstance = ProblemType("code")  # type: ignore
            all_data += classInstance.load_data()  # type: ignore[abstract]
        return all_data


@dataclass
class GSM8K(Dataset):
    def load(self) -> Sequence[Question]:
        all_data: List[Question] = []
        for ProblemType in gsm_problem_types.values():
            classInstance = ProblemType("code")  # type: ignore
            all_data += classInstance.load_data()  # type: ignore[abstract]
        return all_data


@dataclass
class FG(Dataset):
    n: int = 10
    digits_list: List[int] = field(default_factory=lambda: [32])

    def load(self) -> Sequence[Question]:
        all_data: List[Question] = []
        for ProblemType in fg_problem_types.values():
            classInstance = ProblemType("code", self.n, self.digits_list)  # type: ignore
            all_data += classInstance.load_data()  # type: ignore[abstract]
        return all_data

    def load_subset(self, subset: List[str]) -> Sequence[Question]:
        for s in subset:
            assert s in list(fg_problem_types.keys()), "invalid subset"
        all_data: List[Question] = []
        for key, ProblemType in fg_problem_types.items():
            if key not in subset:
                continue
            classInstance = ProblemType("code", self.n, self.digits_list)  # type: ignore
            all_data += classInstance.load_data()  # type: ignore[abstract]
        return all_data


def make_dataset(
    kinds: Sequence[str],
    n: int = 3,
    digits_list: List[int] = [32],
    gsm_samples: int = 500,
    clrs_samples: int = 500,
) -> Sequence[Question]:
    """
    Build dataset deterministically and attach original positional order.
    """
    np = []
    clrs = False
    gsm = False
    fg = []
    for kind in kinds:
        if kind in list(problem_types.keys()):
            np.append(kind)
        if kind in list(clrs_problem_types.keys()):
            clrs = True
        if kind in list(gsm_problem_types.keys()):
            gsm = True
        if kind in list(fg_problem_types.keys()):
            fg.append(kind)

    all_data: List[Question] = []
    if clrs:
        all_data.extend(CLRS().load()[:clrs_samples])
    if gsm:
        all_data.extend(GSM8K().load()[:gsm_samples])
    if fg:
        all_data.extend(
            FG(
                n,
                digits_list,
            ).load_subset(fg)
        )
    if np:
        all_data.extend(NPHARD(n).load_subset(np))
    # attach stable original order
    for i, q in enumerate(all_data):
        setattr(q, "original_pos", i)
    return all_data
