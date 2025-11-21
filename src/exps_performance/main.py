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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.exps_performance.dataset import make_dataset
from src.exps_performance.llm import llm
from src.exps_performance.logger import init_tensorboard, write_text_to_tensorboard, write_to_csv
from src.exps_performance.metrics import accuracy
from src.exps_performance.problems import Question
from src.exps_performance.runners import Arm1, Arm2, Arm3, Arm4
from src.exps_performance.utils import seed_all_and_setup


def run(args):
    seed_all_and_setup(args)
    client = llm(args)

    data = make_dataset(args.kinds, args.n, args.digits_list)  # choose the data
    arm2 = Arm2(data, args, client)
    arm2.run()
    problems_w_code = arm2.set_code()
    arm3 = Arm3(problems_w_code)
    arm3.run()
    data = arm3.edited_problems
    arm4 = Arm4(data, args, client)
    arm4.run()
    data = arm4.edited_problems
    arm1 = Arm1(data, args, client)
    arm1.run()
    data_subset: List[Question] = arm1.edited_problems

    # here should return a sequence of records
    # report summary metrics
    records = [d.record for d in data_subset]
    df = accuracy(records)
    # summarize here
    for kind in df.values:
        for val in kind:
            continue

    # serialize results
    outdir: str = args.model.split("/")[1]
    abs_outdir = os.path.join(Path(__name__), "results", outdir)
    writer, logdir = init_tensorboard(args, abs_outdir)
    write_text_to_tensorboard(records, writer, args)
    csv_path = os.path.join(logdir, "res.csv")
    write_to_csv(csv_path, records)


@dataclass
class ExpArgs:
    n: int = 1000  # number of examples per digit per kind
    digits: List[int] = [
        2  # Global hardness levels. For arithmetic: number magnitude. "LCS: string length; knap: #items; rod: rod length; ilp_assign: n×n siz ilp_prod: scales products/resources/bounds; ilp_partition: #items.
    ]
    kinds: List[str] = [
        "add"  # choices: "add","sub","mul","mix","lcs","knap","rod","ilp_assign","ilp_prod","ilp_partition","gsm8k","nphardeval","clrs30",
    ]
    seed: int = 1
    backend: str = "vllm"
    hf_dtype: str = "float16"
    sim_code_only: bool = True
    exec_code_only: bool = True
    controlled_sim: bool = True
    model: str = "google/gemma-2-9b-it"


@dataclass
class ModelHyperArgs:
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


@dataclass
class LogArgs:
    log_every: int = 50
    tb_text_chars: int = 10000


# add simple parsing type checking to this.
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(ExpArgs)
    p.add_argument(ModelHyperArgs)
    p.add_argument(LogArgs)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
