from dataclasses import dataclass, field
from typing import List

import pytest

from src.exps_performance.llm import DummyClient, OpenAIChatClient, VLLMClient
from src.exps_performance.logger import Record


@dataclass
class CreateArgs:
    n: int = 100
    root: str = "."
    kinds: List[str] = field(default_factory=lambda: ["add"])
    seed: int = 1
    backend: str = "running"
    hf_dtype: str = "float16"
    sim_code_only: bool = True
    exec_code_only: bool = True
    controlled_sim: bool = True
    model: str = "google/gemma-2-9b-it"
    vllm_dtype: str = "float16"
    vllm_tensor_parallel: int = 8
    vllm_gpu_mem_util: float = 0.95
    vllm_max_model_len: int = 8192
    vllm_download_dir: str = "/nlpgpu/data/terry/ToolProj/src/models"
    hf_trust_remote_code: bool = True
    batch_size: int = 16
    max_tokens: int = 2048
    temperature: float = 0
    top_p: float = 1
    log_every: int = 50
    tb_text_chars: int = 10000
    digits_list: int = field(default_factory=lambda: [32])  # type: ignore


@pytest.fixture(scope="session")
def default_args():
    return CreateArgs()


@pytest.fixture(scope="session")
def mock_records():
    fake_record = Record(
        model="abc",  # answers depend on this
        seed=1,  # answers depend on this
        exp_id="abc",
        digit=1,
        kind="abc",
        question="abc",
        answer="abc",
        nl_question="abc",
        nl_answer="abc",
        nl_reasoning="abc",
        nl_correct=True,
        nl_parse_err=True,
        nl_err_msg="abc",  # defaults to "" if not err
        code_question="abc",
        code_answer="abc",  # (or err message)
        code_correct=True,
        code_parse_err=True,
        code_gen_err=True,
        code_err_msg="abc",
        sim_question="abc",
        sim_reasoning="abc",  # attempted reasoning
        sim_answer="abc",
        sim_correct=True,
        sim_parse_err=True,
        sim_err_msg="abc",
        controlsim_question="abc",
        controlsim_reasoning="abc",
        controlsim_answer="abc",
        controlsim_correct=True,
        controlsim_parse_err=True,
        controlsim_err_msg="abc",
    )
    return [fake_record for _ in range(EXAMPLES)]


@pytest.fixture(scope="session")
def mock_record_1():
    fake_record = Record(
        model="efg",  # answers depend on this
        seed=2,  # answers depend on this
        exp_id="efg",
        digit=2,
        kind="efg",
        question="efg",
        answer="efg",
        nl_question="efg",
        nl_answer="efg",
        nl_correct=False,
        nl_reasoning="abc",
        nl_parse_err=False,
        nl_err_msg="efg",  # defaults to "" if not err
        code_question="efg",
        code_answer="efg",  # (or err message)
        code_correct=False,
        code_parse_err=False,
        code_gen_err=False,
        code_err_msg="efg",
        sim_question="efg",
        sim_reasoning="efg",  # attempted reasoning
        sim_answer="efg",
        sim_correct=False,
        sim_parse_err=False,
        sim_err_msg="efg",
        controlsim_question="efg",
        controlsim_reasoning="efg",
        controlsim_answer="efg",
        controlsim_correct=False,
        controlsim_parse_err=False,
        controlsim_err_msg="efg",
    )
    return [fake_record for _ in range(EXAMPLES)]


@pytest.fixture(scope="session")
def llm(default_args):
    args = default_args
    if args.backend == "vllm":
        client = VLLMClient(
            model_name=args.model,
            dtype=args.vllm_dtype,
            tensor_parallel_size=args.vllm_tensor_parallel,
            gpu_memory_utilization=args.vllm_gpu_mem_util,
            max_model_len=args.vllm_max_model_len,
            download_dir=args.vllm_download_dir,
            trust_remote_code=args.hf_trust_remote_code,
            seed=args.seed,
        )
        return client
    elif args.backend == "dummy":
        return DummyClient()
    elif args.backend == "running":
        return OpenAIChatClient()


EXAMPLES = 5
RETRIES = 3


def check(arm, data, types):
    parsed_answer = arm.parsed_answer
    assert arm.parse_fail <= EXAMPLES * RETRIES - 1, "parse failed too much"
    pUtil = data[0].util_pointer(types)
    classtype = pUtil.PROB_TYPES[types]
    empties = 0
    for parsed in parsed_answer:
        assert type(parsed).__name__ == classtype.__name__, "no output, all wrong output types"
        if parsed == classtype():
            empties += 1
    assert empties < RETRIES - 1, "too many no parse"
