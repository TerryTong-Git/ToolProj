from dataclasses import dataclass

import pytest

from src.exps_performance.dataset import NPHARD
from src.exps_performance.llm import DummyClient, OpenAIChatClient, VLLMClient


@dataclass
class CreateArgs:
    n: int = 1000
    digits: int = 2
    kinds: str = "add"
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
    vllm_max_model_len: int = 4096
    vllm_download_dir: str = "/nlpgpu/data/terry/ToolProj/src/models"
    hf_trust_remote_code: bool = True
    batch_size: int = 16
    max_tokens: int = 2048
    temperature: float = 0
    top_p: float = 1


@pytest.fixture(scope="session")
def default_args():
    return CreateArgs()


@pytest.fixture(scope="session")
def instantiate_data():
    return NPHARD().load()


@pytest.fixture(scope="session")
def subset_data():
    return NPHARD().load_subset


@pytest.fixture(scope="session")
def instantiate_llm(default_args):
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


# rather than booting one up, we can just send requests to a command line one? vllm serve Qwen/Qwen2.5-1.5B-Instruct

# from openai import OpenAI
# # Set OpenAI's API key and API base to use vLLM's API server.
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8000/v1"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

# chat_response = client.chat.completions.create(
#     model="Qwen/Qwen2.5-1.5B-Instruct",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Tell me a joke."},
#     ]
# )
# print("Chat response:", chat_response)
