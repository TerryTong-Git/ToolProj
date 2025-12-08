#!/usr/bin/env python3


"""
Experiment: Coding Ability ↔ Tool-Use Ability Correlation (Batched + Tensor Parallel)

Benchmarks
- Coding: HumanEval pass@1, MBPP pass@1
- Tool-use: GSM8K accuracy with Python-REPL tool execution (PAL-style)

Models / Backends
- mock | hf | vllm | openai
- vLLM supports tensor parallelism (--vllm-tp-size)

Usage
python code_tool_correlation.py \
  --models "meta-llama/Llama-3.1-8B-Instruct,google/gemma-2-9b-it" \
  --engine vllm \
  --vllm-tp-size 2 \
  --batch-size 16 \
  --limit-humaneval 164 \
  --limit-mbpp 100 \
  --limit-gsm8k 250 \
  --out results_code_tool.csv

Dependencies
pip install datasets vllm transformers torch pandas numpy scipy tqdm
For OpenAI: pip install openai and set OPENAI_API_KEY
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

PAT = r"(0\.[0-9]+)"


def read_files() -> None:
    parent_dir = Path(__file__).parent.parent
    results_dir = os.path.join(parent_dir, "exps_performance")
    result_subdir = [f for f in Path(results_dir).iterdir() if f.is_dir()]
    pat = re.compile(PAT)
    record = {}
    for subdir in result_subdir:
        files = [f for f in Path(subdir).iterdir() if f.is_file() and f.name.endswith(".txt")]
        name = subdir.name
        for i, f in enumerate(files):  # i is seed num
            content = ""
            with open(f, "r") as file:
                content = file.read()
            content = "".join(content.split("\n")[:4])
            matches = pat.findall(content)
            record[name + "_seed_" + str(i)] = {"NL": matches[0], "CODE": matches[1], "EXEC": matches[2]}

    df = pd.DataFrame.from_dict(record)
    df.to_csv("compiled_res.csv")


"""
N=1319  exp_id=run_20250925_152208
Accuracy NL-CoT (overall):   0.4625
Accuracy Code-CoT (overall): 0.4754
Execution condition (subprocess): acc=0.6277
"""

if __name__ == "__main__":
    read_files()

# Log results to tensorboard to inspect?
