#!/usr/bin/env python3
# pre-commit.ci autofix
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

import argparse
import io
import math
import multiprocessing as mp
import os
import re
import sys
import textwrap
import traceback
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def seed_everything(seed: int):
    import random

    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ----------------------------
# Sandboxed execution (unchanged structure, minor cleanup)
# ----------------------------
def _sandbox_worker(payload, q):
    try:
        sys.stderr = io.StringIO()
        allowed_builtins = {
            "range": range,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "all": all,
            "any": any,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "map": map,
            "filter": filter,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "print": print,
        }
        g = {"__builtins__": allowed_builtins}
        loads = {}
        code = payload["code"]
        exec(code, g, loads)
        res = None
        if "entry" in payload:
            fn = loads.get(payload["entry"]) or g.get(payload["entry"])
            if callable(fn):
                res = fn(*payload.get("args", []), **payload.get("kwargs", {}))
        q.put((True, res, sys.stderr.getvalue()))
    except Exception:
        err = (sys.stderr.getvalue() if hasattr(sys, "stderr") else "") + "\n" + traceback.format_exc()
        q.put((False, None, err))


def _exec_in_subprocess(code: str, input_payload: Dict[str, Any], timeout: float = 3.0):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    payload = dict(input_payload)
    payload["code"] = code
    p = ctx.Process(target=_sandbox_worker, args=(payload, q))
    p.start()
    ok = False
    res = None
    err = ""
    try:
        ok, res, err = q.get(timeout=timeout)
    except Exception:
        err = "Timeout"
        try:
            p.terminate()
        except Exception:
            pass
    finally:
        p.join(timeout=0.1)
    return ok, res, err


# ----------------------------
# Backends (now with batch + TP for vLLM)
# ----------------------------
class LMEngine:
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> str:
        raise NotImplementedError

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> List[str]:
        # Default fallback: serial (override in subclasses)
        return [self.generate(p, max_new_tokens, temperature) for p in prompts]


class MockEngine(LMEngine):
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> str:
        if "Write a function" in prompt or "def " in prompt:
            m = re.search(r"def ([a-zA-Z_]\w*)\(", prompt)
            fn = m.group(1) if m else "solution"
            return f"```python\ndef {fn}(*args, **kwargs):\n    raise NotImplementedError\n```"
        if "FINAL_ANSWER" in prompt or "Solve the problem" in prompt:
            return "FINAL_ANSWER: 0"
        return "0"

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> List[str]:
        return [self.generate(p, max_new_tokens, temperature) for p in prompts]


class HFEngine(LMEngine):
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tk = AutoTokenizer.from_pretrained(model_name)
        self.m = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # shards layers across GPUs if available
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        )
        self.device = getattr(self.m, "device", "cpu")

    def _decode_new(self, input_ids, out_ids) -> str:
        # Decode only the generated continuation
        in_txt = self.tk.decode(input_ids, skip_special_tokens=True)
        full = self.tk.decode(out_ids, skip_special_tokens=True)
        return full[len(in_txt) :].strip()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> str:
        import torch

        toks = self.tk(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.m.generate(
                **toks,
                max_new_tokens=max_new_tokens,
                do_sample=bool(temperature and temperature > 0.0),
                temperature=temperature if temperature and temperature > 0.0 else None,
                pad_token_id=self.tk.eos_token_id,
            )
        return self._decode_new(toks["input_ids"][0], out[0])

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> List[str]:
        import torch

        toks = self.tk(prompts, return_tensors="pt", padding=True, truncation=False)
        toks = {k: v.to(self.device) for k, v in toks.items()}
        with torch.no_grad():
            outs = self.m.generate(
                **toks,
                max_new_tokens=max_new_tokens,
                do_sample=bool(temperature and temperature > 0.0),
                temperature=temperature if temperature and temperature > 0.0 else None,
                pad_token_id=self.tk.eos_token_id,
            )
        # outs shape [B, T+N]; need per-sample decode
        res = []
        for i in range(len(prompts)):
            res.append(self._decode_new(toks["input_ids"][i], outs[i]))
        return res


class VLLMEngine(LMEngine):
    def __init__(self, model_name: str, tp_size: int = 1):
        from vllm import LLM

        # tensor_parallel_size enables true TP across GPUs
        self.llm = LLM(
            model=model_name,
            dtype="float16",
            tensor_parallel_size=int(tp_size),
            max_model_len=4096,
            download_dir="../models",
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> str:
        from vllm import SamplingParams

        sp = SamplingParams(temperature=float(temperature), max_tokens=int(max_new_tokens), seed=seed, top_p=top_p)
        outs = self.llm.generate([prompt], sp)
        return outs[0].outputs[0].text.strip()

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> List[str]:
        from vllm import SamplingParams

        sp = SamplingParams(temperature=float(temperature), max_tokens=int(max_new_tokens), seed=seed, top_p=top_p)
        outs = self.llm.generate(prompts, sp)
        # vLLM returns in same order
        return [o.outputs[0].text.strip() if o.outputs else "" for o in outs]


class OpenAIEngine(LMEngine):
    def __init__(self, model_name: str):
        from openai import OpenAI  # type: ignore

        self.client = OpenAI()
        self.model = model_name

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature),
            max_tokens=int(max_new_tokens),
        )
        return resp.choices[0].message.content.strip()

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> List[str]:
        # Simple chunked loop to be nice to rate limits
        out = []
        for p in prompts:
            out.append(self.generate(p, max_new_tokens, temperature))
        return out


def make_engine(kind: str, model_name: str, vllm_tp_size: int) -> LMEngine:
    kind = kind.lower()
    if kind == "mock":
        return MockEngine()
    if kind == "hf":
        return HFEngine(model_name)
    if kind == "vllm":
        return VLLMEngine(model_name, tp_size=vllm_tp_size)
    if kind == "openai":
        return OpenAIEngine(model_name)
    raise ValueError(f"Unknown engine {kind}")


# ----------------------------
# Benchmarks: loaders
# ----------------------------
def load_humaneval(limit: Optional[int]) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    items = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        items.append(
            {
                "task_id": ex["task_id"],
                "prompt": ex["prompt"],
                "canonical_solution": ex["canonical_solution"],
                "test": ex["test"],
            }
        )
    return items


def load_mbpp(limit: Optional[int]) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("mbpp", split="test")
    items = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        items.append(
            {
                "task_id": f"mbpp-{i}",
                "text": ex["text"],
                "code": ex.get("code", ""),
                "test_list": ex.get("test_list", []),
            }
        )
    return items


def load_gsm8k(limit: Optional[int]) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        items.append(
            {
                "question": ex["question"],
                "answer": ex["answer"],
            }
        )
    return items


# ----------------------------
# Prompts / parsing
# ----------------------------
def strip_code_fence(txt: str) -> str:
    if txt is None:
        return ""
    m = re.search(r"```python(.*?)```", txt, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```(.*?)```", txt, flags=re.S)
    if m2:
        return m2.group(1).strip()
    return txt.strip()


def gsm8k_tool_prompt(q: str) -> str:
    return textwrap.dedent(f"""
    Solve the problem step by step. When computation is needed, write Python between a single ```python code block``` 
    and print the final integer result with:
    print("FINAL_ANSWER:", value)
    After the code block, state only one line: FINAL_ANSWER: <value>

    Problem:
    {q}
    """).strip()


def parse_gsm8k_gold(ans: str) -> Optional[int]:
    m = re.search(r"####\s*(-?\d+)", ans)
    return int(m.group(1)) if m else None


# ----------------------------
# Batched evaluation helpers
# ----------------------------
def _chunks(seq: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def evaluate_gsm8k_batched(
    engine: LMEngine,
    items: List[Dict[str, Any]],
    batch_size: int,
    temp: float = 0.0,
    max_new: int = 512,
    top_p: float = 1.0,
    seed: int = 0,
) -> float:
    prompts = [gsm8k_tool_prompt(ex["question"]) for ex in items]
    correct = 0
    for batch_idx, batch_prompts in enumerate(tqdm(list(_chunks(prompts, batch_size)), desc="GSM8K+Tool (batched)")):
        gens = engine.batch_generate(batch_prompts, max_new_tokens=max_new, temperature=temp, top_p=top_p, seed=seed)
        for j, gen in enumerate(gens):
            ex = items[batch_idx * batch_size + j]
            code = strip_code_fence(gen)
            final_ans = None
            if "FINAL_ANSWER" in code:
                _exec_in_subprocess(code, {"code": code}, timeout=5.0)  # ignore output; we parse text below
            m = re.findall(r"FINAL_ANSWER\s*[:=]\s*(-?\d+)", gen)
            if m:
                final_ans = int(m[-1])
            else:
                m2 = re.findall(r"-?\d+", gen)
                if m2:
                    final_ans = int(m2[-1])
            gold = parse_gsm8k_gold(ex["answer"])
            correct += int((final_ans is not None) and (gold is not None) and (final_ans == gold))
    return correct / max(1, len(items))


# ----------------------------
# Scoring and correlation
# ----------------------------
def corr_pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den != 0 else float("nan")


def corr_spearman(x: List[float], y: List[float]) -> float:
    from scipy.stats import spearmanr

    r, _ = spearmanr(x, y)
    return float(r)


# ----------------------------
# End-to-end
# ----------------------------
@dataclass
class Scores:
    model: str
    tool_gsm8k: float


def evaluate_model(
    engine: LMEngine,
    model_name: str,
    limits: Dict[str, int],
    batch_size: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 0,
) -> Scores:
    g8 = load_gsm8k(limits.get("gsm8k"))
    g8_acc = evaluate_gsm8k_batched(engine, g8, batch_size=batch_size, temp=temperature, max_new=512, top_p=top_p, seed=seed)

    return Scores(
        model=model_name,
        tool_gsm8k=g8_acc,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, required=True, help="Comma-separated model names")
    ap.add_argument("--engine", type=str, default="vllm", choices=["mock", "hf", "vllm", "openai"])
    ap.add_argument("--vllm-tp-size", type=int, default=8, help="Tensor parallel size for vLLM")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size for prompt generation")
    ap.add_argument("--limit-gsm8k", type=int, default=250)
    ap.add_argument("--out", type=str, default="results_code_tool.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=0.7)

    args = ap.parse_args()

    seed_everything(args.seed)

    limits = {"gsm8k": args.limit_gsm8k}
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = []

    for mname in model_names:
        eng = make_engine(args.engine, mname, vllm_tp_size=args.vllm_tp_size)
        s = evaluate_model(
            eng,
            mname,
            limits,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        rows.append(
            {
                "model": s.model,
                "tool_gsm8k_acc": s.tool_gsm8k,
            }
        )
        # Keep running partials per model
        partial_df = pd.DataFrame(rows)
        suffix = mname.split("/")[1]
        partial_df.to_csv(f"{suffix}_seed{args.seed}_" + args.out, index=False)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

# Log results to tensorboard to inspect?
