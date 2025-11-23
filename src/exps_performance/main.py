#!/usr/bin/env python3
"""
NL-CoT vs Code-CoT on arithmetic, DP (LCS/Knapsack/Rod Cutting), and ILP tasks.

New kinds:
- lcs           : LCS length of two strings
- knap          : 0/1 knapsack max value
- rod           : rod-cutting max revenue
- ilp_assign    : assignment min cost (n x n)
- ilp_prod      : production planning (max profit with resource caps)
- ilp_partition : 2-way partition minimal difference

ILPs use PuLP if available; otherwise safe brute-force fallbacks (small sizes).
Code-CoT subprocess is constrained but allows imports.

Usage examples:
  python cot_general.py --backend hf --model google/gemma-2-9b-it \
    --n 60 --digits 8 9 10 --kinds add sub mul lcs knap rod ilp_assign ilp_prod ilp_partition \
    --exec_code --outdir out_hf
  tensorboard --logdir out_hf/tb
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from src.exps_performance.arms import Arm1, Arm2, Arm3, Arm4
from src.exps_performance.dataset import make_dataset
from src.exps_performance.llm import llm
from src.exps_performance.logger import create_dir, init_tensorboard, write_text_to_tensorboard, write_to_csv
from src.exps_performance.metrics import accuracy
from src.exps_performance.utils import seed_all_and_setup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Using __name__ is a common practice


def run(args):
    seed_all_and_setup(args)
    client = llm(args)
    logger.info("Making Dataset")
    data = make_dataset(args.kinds, args.n, args.digits_list)  # choose the data
    logger.info("Running Arm2")
    arm2 = Arm2(data, args, client)
    _, data = arm2.run()
    logger.info("Running Arm3")
    arm3 = Arm3(data, args, client)
    _, data = arm3.run()
    logger.info("Running Arm4")
    arm4 = Arm4(data, args, client)
    _, data = arm4.run()
    logger.info("Running Arm1")
    arm1 = Arm1(data, args, client)
    _, data = arm1.run()
    logger.info("Saving Results")
    records = [d.record for d in data]
    df = accuracy(records)
    # summarize here
    for kind in df.values:
        for val in kind:
            continue

    # serialize results
    exp_dir = create_dir(args, Path(args.root))
    writer = init_tensorboard(args, exp_dir)
    write_text_to_tensorboard(records, writer, args)
    csv_path = os.path.join(exp_dir, "res.csv")
    write_to_csv(csv_path, records)


# add simple parsing type checking to this.
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--n", type=int, default=1, help="total problems (balanced over kinds)")
    p.add_argument(
        "--digits_list",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Global hardness levels. For arithmetic: number magnitude. "
        "LCS: string length; knap: #items; rod: rod length; "
        "ilp_assign: n×n size; ilp_prod: scales products/resources/bounds; "
        "ilp_partition: #items.",
    )
    p.add_argument(
        "--kinds",
        type=str,
        nargs="+",
        default=[
            "add",
            "sub",
            "mul",
            "lcs",
            "knap",
            "rod",
            "ilp_assign",
            "ilp_prod",
            "ilp_partition",
        ],
    )
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--backend", type=str, default="dummy", choices=["dummy", "openai", "running", "vllm"])
    p.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model name or HF repo/path when --backend=hf",
    )
    p.add_argument(
        "--hf_dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    p.add_argument("--hf_device_map", type=str, default="auto")
    p.add_argument("--hf_trust_remote_code", action="store_true")

    p.add_argument("--max_tokens", type=int, default=4192)
    p.add_argument("--temperature", type=int, default=0.1)
    p.add_argument("--top_p", type=int, default=0.90)

    p.add_argument("--sim_code_only", action="store_true", help="Simulate only the generated code, not any NL input for fair comparison with arm 3")
    p.add_argument(
        "--exec_code",
        action="store_true",
        help="execute code-CoT in sandboxed subprocess (imports allowed)",
    )
    p.add_argument(
        "--controlled_sim",
        action="store_true",
        help="do fair controlled simulation w/o prompt",
    )
    p.add_argument("--log_every", type=int, default=50)

    # TensorBoard text limits
    p.add_argument("--tb_text_chars", type=int, default=10000)
    p.add_argument("--tb_disable", action="store_true")

    # vLLM options (kept minimal; defaults are conservative)
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for backends that support chat_many (vLLM).",
    )
    p.add_argument("--vllm_dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16"])
    p.add_argument("--vllm_tensor_parallel", type=int, default=8)
    p.add_argument("--vllm_gpu_mem_util", type=float, default=0.90)
    p.add_argument("--vllm_max_model_len", type=int, default=8192)
    p.add_argument("--vllm_download_dir", type=str, default="/nlpgpu/data/terry/ToolProj/src/models")
    return p.parse_args()


if __name__ == "__main__":
    start_time = time.perf_counter()
    args = parse_args()
    run(args)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")
