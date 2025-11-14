from dataclasses import dataclass

import pytest

from src.exps_performance.llm import VLLMClient


@dataclass
class CreateArgs:
    n: int = 1000
    digits: int = 2
    kinds: str = "add"
    seed: int = 1
    backend: str = "vllm"
    hf_dtype: str = "float16"
    sim_code_only: bool = True
    exec_code_only: bool = True
    controlled_sim: bool = True
    model: str = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    vllm_dtype: str = "float16"
    vllm_tensor_parallel: int = 8
    vllm_gpu_mem_util: float = 0.95
    vllm_max_model_len: int = 2048
    vllm_download_dir: str = "/nlpgpu/data/terry/ToolProj/src/models"
    hf_trust_remote_code: bool = True
    batch_size: int = 16
    max_tokens: int = 1024
    temperature: float = 0.95
    top_p: float = 0.1


@pytest.fixture()
def default_args():
    return CreateArgs()


@pytest.fixture()
def instantiate_llm(default_args):
    args = default_args
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
