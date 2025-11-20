from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import torch

from src.exps_performance.problems import Question
from src.exps_performance.problems.clrs import ClrsCheckAndFormat
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

try:
    from vllm import LLM as VLLMEngine
    from vllm import SamplingParams
except Exception as _vllm_import_err:
    VLLMEngine = None
    SamplingParams = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.deterministic = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# data interface

# runner interface -> track statistics


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
}

gsm_problem_types = {
    "gsm8k": Gsm8kCheckAndFormat,
}


class NPHARD(Dataset):
    def load(self) -> Sequence[Question]:
        all_data: List[Question] = []
        for ProblemType in problem_types.values():
            classInstance = ProblemType("code")  # type: ignore
            all_data += classInstance.load_data()  # type: ignore[abstract]
        return all_data

    def load_subset(self, subset: List[str]):
        for s in subset:
            assert s in list(problem_types.keys()), "invalid subset"
        all_data: List[Question] = []
        for key, ProblemType in problem_types.items():
            if key not in subset:
                continue
            classInstance = ProblemType("code")  # type: ignore
            all_data += classInstance.load_data()  # type: ignore[abstract]
        return all_data


class CLRS(Dataset):
    def load(self) -> Sequence[Question]:
        all_data: List[Question] = []
        for ProblemType in clrs_problem_types.values():
            classInstance = ProblemType("code")  # type: ignore
            all_data += classInstance.load_data()  # type: ignore[abstract]
        return all_data


class GSM8K(Dataset):
    def load(self) -> Sequence[Question]:
        all_data: List[Question] = []
        for ProblemType in gsm_problem_types.values():
            classInstance = ProblemType("code")  # type: ignore
            all_data += classInstance.load_data()  # type: ignore[abstract]
        return all_data


fg_problem_types: dict = {}


class FG(Dataset):
    n: int
    digits_list: List[int]
    kinds: List[str]
    seed: int = 1

    def load(self) -> Sequence[Question]:
        all_data: List[Question] = []
        for ProblemType in fg_problem_types.values():
            classInstance = ProblemType("code")  # type: ignore
            all_data += classInstance.load_data()  # type: ignore[abstract]
        return all_data
