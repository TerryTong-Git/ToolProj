#!/usr/bin/env python3
"""
Experiment: Coding Ability ↔ Tool-Use Ability Correlation

Benchmarks
- Coding: HumanEval pass@1, MBPP pass@1
- Tool-use: GSM8K accuracy with Python-REPL tool execution (PAL-style)

Models
- Backends: mock | hf | vllm | openai
- Evaluate multiple models and compute Pearson/Spearman correlations between:
    coding_score (avg of HumanEval and MBPP pass@1)
    tool_score (GSM8K accuracy with tool)

Usage
python code_tool_correlation.py \
  --models "meta-llama/Llama-3.1-8B-Instruct,google/gemma-2-9b-it" \
  --engine vllm \
  --limit-humaneval 164 \
  --limit-mbpp 100 \
  --limit-gsm8k 250 \
  --out results_code_tool.csv

Dependencies
pip install datasets vllm transformers torch pandas numpy scipy tqdm
For OpenAI: pip install openai and set OPENAI_API_KEY
"""

from __future__ import annotations
import os, re, ast, math, time, json, signal, argparse, textwrap, multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from statistics import mean
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------------
# Backends
# ----------------------------
class LMEngine:
    def generate(self, prompt: str, max_new_tokens: int = 384, temperature: float = 0.0) -> str:
        raise NotImplementedError

class MockEngine(LMEngine):
    def generate(self, prompt: str, max_new_tokens: int = 384, temperature: float = 0.0) -> str:
        # trivial pattern: return minimal plausible code or number
        if "Write a function" in prompt or "def " in prompt:
            # emit pass-through solution that often fails
            m = re.search(r"def ([a-zA-Z_]\w*)\(", prompt)
            fn = m.group(1) if m else "solution"
            return f"```python\ndef {fn}(*args, **kwargs):\n    raise NotImplementedError\n```"
        if "FINAL_ANSWER" in prompt or "Solve the problem" in prompt:
            return "FINAL_ANSWER: 0"
        return "0"

class HFEngine(LMEngine):
    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tk = AutoTokenizer.from_pretrained(model_name)
        self.m = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        )
        self.device = self.m.device

    def generate(self, prompt: str, max_new_tokens: int = 384, temperature: float = 0.0) -> str:
        import torch
        toks = self.tk(prompt, return_tensors="pt").to(self.device)
        out = self.m.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=self.tk.eos_token_id,
        )
        txt = self.tk.decode(out[0], skip_special_tokens=True)
        return txt[len(self.tk.decode(toks["input_ids"][0], skip_special_tokens=True)) :].strip()

class VLLMEngine(LMEngine):
    def __init__(self, model_name: str):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_name, dtype='float16', download_dir="../models", max_model_len=4196)
        self.sp = SamplingParams(temperature=0.0, max_tokens=384)

    def generate(self, prompt: str, max_new_tokens: int = 384, temperature: float = 0.0) -> str:
        # vLLM SamplingParams already set; ignore per-call max_new_tokens/temperature for simplicity
        outs = self.llm.generate([prompt], self.sp)
        return outs[0].outputs[0].text.strip()

class OpenAIEngine(LMEngine):
    def __init__(self, model_name: str):
        import openai
        self.client = openai.OpenAI()
        self.model = model_name

    def generate(self, prompt: str, max_new_tokens: int = 384, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content.strip()

def make_engine(kind: str, model_name: str) -> LMEngine:
    kind = kind.lower()
    if kind == "mock": return MockEngine()
    if kind == "hf":   return HFEngine(model_name)
    if kind == "vllm": return VLLMEngine(model_name)
    if kind == "openai": return OpenAIEngine(model_name)
    raise ValueError(f"Unknown engine {kind}")

# ----------------------------
# Safe execution utilities
# ----------------------------
def _exec_in_subprocess(code: str, input_payload: Dict[str, Any], timeout: float = 3.0) -> Tuple[bool, Any, str]:
    """
    Execute Python code in a clean subprocess with minimal builtins.
    Return (ok, result, stderr_text).
    """
    def worker(payload, conn):
        try:
            import sys, io
            sys.stderr = io.StringIO()
            # Restricted globals
            allowed_builtins = {
                "range": range, "len": len, "min": min, "max": max, "sum": sum, "abs": abs, "all": all, "any": any,
                "enumerate": enumerate, "zip": zip, "sorted": sorted, "map": map, "filter": filter, "int": int,
                "float": float, "str": str, "list": list, "dict": dict, "set": set, "tuple": tuple
            }
            g = {"__builtins__": allowed_builtins}
            l = {}
            exec(payload["code"], g, l)
            res = None
            if "entry" in payload:
                fn = l.get(payload["entry"]) or g.get(payload["entry"])
                if callable(fn):
                    res = fn(*payload.get("args", []), **payload.get("kwargs", {}))
            conn.send((True, res, sys.stderr.getvalue()))
        except Exception as e:
            import traceback, sys
            conn.send((False, None, (sys.stderr.getvalue() if hasattr(sys, "stderr") else "") + "\n" + traceback.format_exc()))
        finally:
            conn.close()

    parent, child = mp.Pipe()
    p = mp.Process(target=worker, args=(input_payload, child))
    p.start()
    parent_conn = parent
    ok = False; res = None; err = ""
    if parent_conn.poll(timeout):
        ok, res, err = parent_conn.recv()
    else:
        try:
            p.terminate()
        except Exception:
            pass
        ok, res, err = (False, None, "Timeout")
    p.join(timeout=0.1)
    return ok, res, err

# ----------------------------
# Benchmarks: loaders
# ----------------------------
def load_humaneval(limit: Optional[int]) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    items = []
    for i, ex in enumerate(ds):
        if limit and i >= limit: break
        items.append({
            "task_id": ex["task_id"],
            "prompt": ex["prompt"],          # includes 'def f(...):'
            "canonical_solution": ex["canonical_solution"],
            "test": ex["test"],              # unit tests as code
        })
    return items

def load_mbpp(limit: Optional[int]) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("mbpp", split="test")  # test has held-out tasks with test_list field
    items = []
    for i, ex in enumerate(ds):
        if limit and i >= limit: break
        items.append({
            "task_id": f"mbpp-{i}",
            "text": ex["text"],                 # natural language description
            "code": ex.get("code", ""),         # reference solution (unused)
            "test_list": ex.get("test_list", []), # list of assert strings
        })
    return items

def load_gsm8k(limit: Optional[int]) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i, ex in enumerate(ds):
        if limit and i >= limit: break
        items.append({
            "question": ex["question"],
            "answer": ex["answer"],  # contains final number like "#### 42"
        })
    return items

# ----------------------------
# Benchmarks: prompts and evaluation
# ----------------------------
def strip_code_fence(txt: str) -> str:
    m = re.search(r"```python(.*?)```", txt, flags=re.S|re.I)
    if m: return m.group(1).strip()
    m2 = re.search(r"```(.*?)```", txt, flags=re.S)
    if m2: return m2.group(1).strip()
    return txt.strip()

def humaneval_prompt(ex: Dict[str, Any]) -> str:
    # Instruct to output pure code block with complete function
    return textwrap.dedent(f"""
    Write a correct and efficient solution as Python code block for the following specification.
    Return only a single ```python code block``` containing the full function implementation. No prose.

    Specification:
    {ex["prompt"]}
    """).strip()

def run_humaneval_case(engine: LMEngine, ex: Dict[str, Any]) -> bool:
    gen = engine.generate(humaneval_prompt(ex), max_new_tokens=512, temperature=0.0)
    code = strip_code_fence(gen)
    # Compose code + tests; run tests in subprocess
    test_code = ex["test"]
    payload_code = code + "\n\n" + test_code + "\n"
    ok, _, err = _exec_in_subprocess(payload_code, {"code": payload_code}, timeout=5.0)
    return ok

def mbpp_prompt(ex: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""
    Write a correct Python function that satisfies the description. Return only a single ```python code block``` with the function definition. No prose.

    Description:
    {ex["text"]}
    """).strip()

def run_mbpp_case(engine: LMEngine, ex: Dict[str, Any]) -> bool:
    gen = engine.generate(mbpp_prompt(ex), max_new_tokens=384, temperature=0.0)
    code = strip_code_fence(gen)
    test_snippets = "\n".join(ex.get("test_list", []))
    payload = code + "\n\n" + test_snippets + "\n"
    ok, _, err = _exec_in_subprocess(payload, {"code": payload}, timeout=5.0)
    return ok

def parse_gsm8k_gold(ans: str) -> Optional[int]:
    # gold has "#### 42" pattern
    m = re.search(r"####\s*(-?\d+)", ans)
    return int(m.group(1)) if m else None

def gsm8k_tool_prompt(q: str) -> str:
    return textwrap.dedent(f"""
    Solve the problem step by step. When computation is needed, write Python between a single ```python code block``` and print the final integer result with:
    print("FINAL_ANSWER:", value)
    After the code block, state only one line: FINAL_ANSWER: <value>

    Problem:
    {q}
    """).strip()

def run_gsm8k_case_with_tool(engine: LMEngine, ex: Dict[str, Any]) -> bool:
    gen = engine.generate(gsm8k_tool_prompt(ex["question"]), max_new_tokens=512, temperature=0.0)
    # Execute any python block found; capture FINAL_ANSWER
    code = strip_code_fence(gen)
    final_ans = None

    if "FINAL_ANSWER" in code:
        # Execute code; try to capture stdout text
        ok, _, err = _exec_in_subprocess(code, {"code": code}, timeout=5.0)
        # Regardless of execution, also parse from generated text after the code block
    # Parse FINAL_ANSWER from full text
    m = re.findall(r"FINAL_ANSWER\s*[:=]\s*(-?\d+)", gen)
    if m:
        final_ans = int(m[-1])
    else:
        # fallback: last integer in the output
        m2 = re.findall(r"-?\d+", gen)
        if m2:
            final_ans = int(m2[-1])

    gold = parse_gsm8k_gold(ex["answer"])
    return (final_ans is not None) and (gold is not None) and (final_ans == gold)

# ----------------------------
# Scoring and correlation
# ----------------------------
def corr_pearson(x: List[float], y: List[float]) -> float:
    import math
    if len(x) < 2: return float("nan")
    mx, my = mean(x), mean(y)
    num = sum((a-mx)*(b-my) for a,b in zip(x,y))
    den = math.sqrt(sum((a-mx)**2 for a in x) * sum((b-my)**2 for b in y))
    return num/den if den != 0 else float("nan")

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
    coding_humaneval: float
    coding_mbpp: float
    coding_avg: float
    tool_gsm8k: float

def evaluate_model(engine: LMEngine, model_name: str, limits: Dict[str,int]) -> Scores:
    # HumanEval
    he = load_humaneval(limits.get("humaneval"))
    he_correct = 0
    for ex in tqdm(he, desc=f"[{model_name}] HumanEval"):
        he_correct += int(run_humaneval_case(engine, ex))
    he_pass1 = he_correct / max(1, len(he))

    # MBPP
    mb = load_mbpp(limits.get("mbpp"))
    mb_correct = 0
    for ex in tqdm(mb, desc=f"[{model_name}] MBPP"):
        mb_correct += int(run_mbpp_case(engine, ex))
    mb_pass1 = mb_correct / max(1, len(mb))

    # GSM8K with tool
    g8 = load_gsm8k(limits.get("gsm8k"))
    g8_correct = 0
    for ex in tqdm(g8, desc=f"[{model_name}] GSM8K+Tool"):
        g8_correct += int(run_gsm8k_case_with_tool(engine, ex))
    g8_acc = g8_correct / max(1, len(g8))

    return Scores(
        model=model_name,
        coding_humaneval=he_pass1,
        coding_mbpp=mb_pass1,
        coding_avg=(he_pass1 + mb_pass1)/2.0,
        tool_gsm8k=g8_acc,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, required=True, help="Comma-separated model names")
    ap.add_argument("--engine", type=str, default="vllm", choices=["mock","hf","vllm","openai"])
    ap.add_argument("--limit-humaneval", type=int, default=164)   # full HumanEval=164
    ap.add_argument("--limit-mbpp", type=int, default=100)        # subset for speed
    ap.add_argument("--limit-gsm8k", type=int, default=250)       # subset for speed
    ap.add_argument("--out", type=str, default="results_code_tool.csv")
    args = ap.parse_args()

    limits = {"humaneval": args.limit_humaneval, "mbpp": args.limit_mbpp, "gsm8k": args.limit_gsm8k}

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    coding_scores = []
    tool_scores = []

    for mname in model_names:
        eng = make_engine(args.engine, mname)
        s = evaluate_model(eng, mname, limits)
        rows.append({
            "model": s.model,
            "coding_humaneval_pass1": s.coding_humaneval,
            "coding_mbpp_pass1": s.coding_mbpp,
            "coding_avg": s.coding_avg,
            "tool_gsm8k_acc": s.tool_gsm8k,
        })
        coding_scores.append(s.coding_avg)
        tool_scores.append(s.tool_gsm8k)

    pear = corr_pearson(coding_scores, tool_scores)
    spear = corr_spearman(coding_scores, tool_scores)

    df = pd.DataFrame(rows)
    df["pearson_code_tool"] = pear
    df["spearman_code_tool"] = spear
    df.to_csv(args.out, index=False)

    print("\n=== Model-wise Scores ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== Correlation Across Models ===")
    print(f"Pearson(coding_avg, tool_gsm8k): {pear:.4f}")
    print(f"Spearman(coding_avg, tool_gsm8k): {spear:.4f}")
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
