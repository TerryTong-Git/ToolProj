#!/usr/bin/env python3
"""
Experiment: Compare Natural-Language CoT vs Code CoT on arithmetic tasks without fine-tuning.
Primary comparison disallows execution for either mode (fair CoT-only).
Optional third condition executes a constrained arithmetic expression for code-CoT (quantifies tool advantage).

Reproducible, single-file, minimal dependencies.

Outputs:
- results.csv: per-item records (problem, answers, correctness)
- summary.txt: accuracies, McNemar exact binomial test, per-bin analysis by digit length

Backends:
- Implement your own LLM client by subclassing LLMClient (see DummyClient), or
- Use OpenAIChatClient (requires OPENAI_API_KEY env var and openai>=1.0), or
- Use VLLMHTTPClient (optional, simple /v1/chat/completions endpoint mimic).

Run:
  python cot_code_vs_nl.py --n 500 --digits 2 4 8 --backend openai --model gpt-4o
  python cot_code_vs_nl.py --n 1000 --digits 6 --backend dummy  # for dry-run without API

Fairness controls:
- Same base model, temperature=0, top_p=1, identical max_tokens and stop.
- Same problems per condition, randomized order, fixed seed.
- Same JSON output schema: {"rationale": ..., "answer": ...}.
- Primary metric uses 'answer' only (no execution) for both conditions.
- Token budget parity enforced by max_tokens and minimal prompt overhead.

Statistical test:
- McNemar exact binomial test on discordant pairs (code-correct, nl-incorrect) vs (code-incorrect, nl-correct).

Author: Research mode.
"""

from __future__ import annotations
import argparse, os, json, re, math, random, time, sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm

# ------------------------------- Dataset ------------------------------------

def sample_int(digits: int, rng: random.Random) -> int:
    if digits <= 0:
        raise ValueError("digits must be >=1")
    lo = 10**(digits-1)
    hi = 10**digits - 1
    return rng.randint(lo, hi)

@dataclass
class Problem:
    kind: str  # 'add' | 'sub' | 'mul' | 'mix'
    a: int
    b: int
    digits: int

    def text(self) -> str:
        if self.kind == 'add':
            return f"Compute: {self.a} + {self.b}"
        if self.kind == 'sub':
            return f"Compute: {self.a} - {self.b}"
        if self.kind == 'mul':
            return f"Compute: {self.a} * {self.b}"
        if self.kind == 'mix':
            # Simple two-op expression to avoid ambiguity
            # e.g., (a + b) * a
            return f"Compute: ({self.a} + {self.b}) * {self.a}"
        raise ValueError("unknown kind")

    def ground_truth(self) -> int:
        if self.kind == 'add':
            return self.a + self.b
        if self.kind == 'sub':
            return self.a - self.b
        if self.kind == 'mul':
            return self.a * self.b
        if self.kind == 'mix':
            return (self.a + self.b) * self.a
        raise ValueError("unknown kind")


def make_dataset(n: int, digits_list: List[int], kinds: List[str], seed: int=1) -> List[Problem]:
    rng = random.Random(seed)
    problems: List[Problem] = []
    for digits in digits_list:
        per = n // len(digits_list)
        for _ in range(per):
            kind = rng.choice(kinds)
            a = sample_int(digits, rng)
            b = sample_int(digits, rng)
            problems.append(Problem(kind=kind, a=a, b=b, digits=digits))
    rng.shuffle(problems)
    return problems

# ------------------------------- Prompts ------------------------------------

JSON_SCHEMA = (
    "Return only JSON with keys 'rationale' and 'answer'. "
    "'answer' must be a single integer. No extra keys, no text outside JSON."
)

NL_PROMPT = (
    "You are a careful math tutor. Solve the given arithmetic problem step by step "
    "in clear natural language sentences. Do not include any code or formulas in backticks.\n"
    + JSON_SCHEMA + "\n"
    "Problem: {problem}\n"
    "Output format example: {{\"rationale\": \"...natural language steps...\", \"answer\": 42}}"
)

CODE_PROMPT = (
    "You are a precise Python programmer. Write a single, minimal Python expression that computes the answer. "
    "Place the expression inside one fenced block using triple backticks. No imports, no functions, no variables."
    + JSON_SCHEMA + ""
    "Problem: {problem}"
    "Output format example: {{\"rationale\": \"```\n2+3\n```\", \"answer\": 5}}"
)

# ------------------------------- LLM Clients --------------------------------

class LLMClient:
    def chat(self, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]]=None) -> str:
        raise NotImplementedError

class DummyClient(LLMClient):
    """Deterministic stub for dry runs: echoes a trivial correct answer."""
    def chat(self, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]]=None) -> str:
        # Extract last problem numbers to compute trivially
        last = messages[-1]['content']
        m = re.search(r"Compute:\s*([0-9]+)\s*([+\-*])\s*([0-9]+)", last)
        if not m:
            m = re.search(r"Compute:\s*\((\d+)\s*\+\s*(\d+)\)\s*\*\s*(\d+)", last)
            if m:
                a, b, c = map(int, m.groups())
                ans = (a+b)*c
            else:
                ans = 0
                rationale = {"rationale":"could not parse","answer":ans}
                return json.dumps(rationale)
        else:
            a, op, b = m.groups()
            a = int(a); b = int(b)
            ans = a+b if op=='+' else (a-b if op=='-' else a*b)
        # Decide mode by presence of "math tutor" vs "Python programmer"
        if "math tutor" in messages[-1]['content']:
            rat = f"First, identify the operation. Then compute the result deterministically."
            out = {"rationale": rat, "answer": ans}
        else:
            rat = "```\n" + re.sub(r"Compute:\s*", "", last).strip() + "\n```"
            rat = rat.replace("Compute:", "").strip()
            out = {"rationale": rat, "answer": ans}
        return json.dumps(out)

class OpenAIChatClient(LLMClient):
    def __init__(self):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("pip install openai>=1.0 required") from e
        self.client = OpenAI()

    def chat(self, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]]=None) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )
        return resp.choices[0].message.content
class HFLocalClient(LLMClient):
    """Vanilla Hugging Face transformers inference (no NCCL/vLLM)."""
    def __init__(self, model_name: str, dtype: str = "auto", device_map: str = "auto", trust_remote_code: bool = False):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        _map = {"auto": None, "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = _map.get(dtype, None)
        cache_dir = "../models"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir
        )
        self.has_template = hasattr(self.tokenizer, "apply_chat_template") and (self.tokenizer.chat_template is not None)
    def chat(self, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]] = None) -> str:
        import torch
        if self.has_template:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            prompt = messages[-1]["content"]
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = (temperature is not None) and (float(temperature) > 0.0)
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            top_p=float(top_p) if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        out_ids = gen_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        if stop:
            idxs = [text.find(s) for s in stop if s in text]
            if idxs:
                cut = min(i for i in idxs if i >= 0)
                text = text[:cut]
        return text

# ------------------------------- Parsing ------------------------------------

def extract_json(s: str) -> Optional[Dict[str, Any]]:
    # grab the first {...} JSON object
    try:
        start = s.index('{')
        end = s.rindex('}')
        frag = s[start:end+1]
        return json.loads(frag)
    except Exception:
        return None

INT_RE = re.compile(r"[-+]?[0-9]+")

@dataclass
class Parsed:
    raw: str
    ok: bool
    answer: Optional[int]
    rationale: Optional[str]
    err: Optional[str]


def parse_response(raw: str) -> Parsed:
    obj = extract_json(raw)
    if obj is None:
        # fallback: try to find a number
        m = INT_RE.findall(raw)
        if not m:
            return Parsed(raw, False, None, None, "no-json-no-int")
        try:
            ans = int(m[-1])
            return Parsed(raw, True, ans, None, "json-missing")
        except Exception:
            return Parsed(raw, False, None, None, "int-parse-failed")
    # check keys
    if not (isinstance(obj, dict) and 'answer' in obj and 'rationale' in obj):
        return Parsed(raw, False, None, None, "bad-json-keys")
    ans = obj['answer']
    try:
        ans = int(ans)
    except Exception:
        # Try to parse last integer from string
        if isinstance(ans, str):
            m = INT_RE.findall(ans)
            if m:
                ans = int(m[-1])
            else:
                return Parsed(raw, False, None, obj.get('rationale'), "answer-not-int")
        else:
            return Parsed(raw, False, None, obj.get('rationale'), "answer-not-int-type")
    return Parsed(raw, True, ans, obj.get('rationale'), None)

# ----------------------- Optional code execution (constrained) --------------

import ast

ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Constant,  # ints
    ast.Paren if hasattr(ast, 'Paren') else ast.expr,  # py>=3.12 adds ast.Paren
)

class SafeEval(ast.NodeVisitor):
    def visit(self, node):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"disallowed node: {type(node).__name__}")
        return super().visit(node)
    def eval(self, src: str) -> int:
        tree = ast.parse(src, mode='eval')
        self.visit(tree)
        return int(self._eval(tree.body))
    def _eval(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int,)):
                return node.value
            raise ValueError("non-int constant")
        if isinstance(node, ast.UnaryOp):
            v = self._eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return +v
            raise ValueError("bad unary op")
        if isinstance(node, ast.BinOp):
            l = self._eval(node.left)
            r = self._eval(node.right)
            if isinstance(node.op, ast.Add): return l + r
            if isinstance(node.op, ast.Sub): return l - r
            if isinstance(node.op, ast.Mult): return l * r
            if isinstance(node.op, ast.Div): return int(l / r)
            if isinstance(node.op, ast.FloorDiv): return l // r
            if isinstance(node.op, ast.Mod): return l % r
            if isinstance(node.op, ast.Pow): return l ** r
            raise ValueError("bad binop")
        raise ValueError("bad node type")

FENCE_RE = re.compile(r"```+\n([\s\S]*?)\n```+", re.MULTILINE)

def try_exec_code_rationale(rationale: Optional[str]) -> Optional[int]:
    if not rationale:
        return None
    m = FENCE_RE.search(rationale)
    if not m:
        return None
    expr = m.group(1).strip()
    try:
        return SafeEval().eval(expr)
    except Exception:
        return None

# ------------------------------- Evaluation ---------------------------------

@dataclass
class Record:
    idx: int
    problem: str
    digits: int
    kind: str
    truth: int
    answer_nl: Optional[int]
    correct_nl: int
    answer_code: Optional[int]
    correct_code: int
    answer_code_exec: Optional[int]
    correct_code_exec: int
    raw_nl: str
    raw_code: str


def mcnemar_exact_p(b: int, c: int) -> float:
    # exact binomial two-sided on discordant pairs
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided p = 2 * P[X <= k] where X~Bin(n, 0.5), clipped at 1
    def binom_cdf_leq(n, k):
        s = 0.0
        for i in range(0, k+1):
            s += math.comb(n, i)
        return s * (0.5**n)
    p = 2.0 * binom_cdf_leq(n, k)
    return min(1.0, p)

# ------------------------------- Runner -------------------------------------

def run(args):
    # backend
    if args.backend == 'dummy':
        client: LLMClient = DummyClient()
    elif args.backend == 'openai':
        client = OpenAIChatClient()
    elif args.backend == 'hf':
        client = HFLocalClient(
            model_name=args.model,
            dtype=args.hf_dtype,
            device_map=args.hf_device_map,
            trust_remote_code=args.hf_trust_remote_code,
        )
    else:
        raise ValueError("backend must be one of {dummy, openai, hf}")

    problems = make_dataset(args.n, args.digits, args.kinds, seed=args.seed)

    records: List[Record] = []

    for i, pb in enumerate(tqdm(problems, total=len(problems), desc='eval')):
        problem_text = pb.text()
        truth = pb.ground_truth()

        # NL condition
        nl_prompt = NL_PROMPT.format(problem=problem_text)
        nl_raw = client.chat(
            model=args.model,
            messages=[{"role":"user","content": nl_prompt}],
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=None,
        )
        nl_parsed = parse_response(nl_raw)
        ans_nl = nl_parsed.answer
        correct_nl = int(ans_nl == truth)

        # Code condition (no execution)
        code_prompt = CODE_PROMPT.format(problem=problem_text)
        code_raw = client.chat(
            model=args.model,
            messages=[{"role":"user","content": code_prompt}],
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=None,
        )
        code_parsed = parse_response(code_raw)
        ans_code = code_parsed.answer
        correct_code = int(ans_code == truth)

        # Optional execution: parse fenced code expression and evaluate safely
        ans_code_exec = try_exec_code_rationale(code_parsed.rationale) if args.exec_code else None
        correct_code_exec = int(ans_code_exec == truth) if ans_code_exec is not None else 0

        records.append(Record(
            idx=i,
            problem=problem_text,
            digits=pb.digits,
            kind=pb.kind,
            truth=truth,
            answer_nl=ans_nl,
            correct_nl=correct_nl,
            answer_code=ans_code,
            correct_code=correct_code,
            answer_code_exec=ans_code_exec,
            correct_code_exec=correct_code_exec,
            raw_nl=nl_raw,
            raw_code=code_raw,
        ))
        if (i+1) % max(1, args.log_every) == 0:
            print(f"done {i+1}/{len(problems)}")
            sys.stdout.flush()

    # Write CSV
    import csv
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["idx","problem","digits","kind","truth",
                         "answer_nl","correct_nl",
                         "answer_code","correct_code",
                         "answer_code_exec","correct_code_exec"]) 
        for r in records:
            writer.writerow([r.idx, r.problem, r.digits, r.kind, r.truth,
                             r.answer_nl, r.correct_nl,
                             r.answer_code, r.correct_code,
                             r.answer_code_exec, r.correct_code_exec])

    # Summary
    def acc(xs):
        return sum(xs)/max(1,len(xs))

    acc_nl = acc([r.correct_nl for r in records])
    acc_code = acc([r.correct_code for r in records])

    # Paired discordances for McNemar
    b = sum(1 for r in records if r.correct_code==1 and r.correct_nl==0)
    c = sum(1 for r in records if r.correct_code==0 and r.correct_nl==1)
    p_mc = mcnemar_exact_p(b, c)

    # Per-bin by digits
    by_digits: Dict[int, List[Record]] = {}
    for r in records:
        by_digits.setdefault(r.digits, []).append(r)

    lines = []
    lines.append(f"N={len(records)}")
    lines.append(f"Accuracy NL-CoT:   {acc_nl:.4f}")
    lines.append(f"Accuracy Code-CoT: {acc_code:.4f}")
    lines.append(f"Discordant pairs b=code>nl: {b}, c=nl>code: {c}, McNemar exact p={p_mc:.4g}")
    lines.append("")
    lines.append("Per-digit bins:")
    for d in sorted(by_digits):
        recs = by_digits[d]
        a_nl = acc([x.correct_nl for x in recs])
        a_code = acc([x.correct_code for x in recs])
        b_d = sum(1 for x in recs if x.correct_code==1 and x.correct_nl==0)
        c_d = sum(1 for x in recs if x.correct_code==0 and x.correct_nl==1)
        p_d = mcnemar_exact_p(b_d, c_d)
        lines.append(f"  digits={d}: N={len(recs)} NL={a_nl:.4f} Code={a_code:.4f} McNemar p={p_d:.4g}")

    if any(r.answer_code_exec is not None for r in records):
        acc_code_exec = acc([r.correct_code_exec for r in records if r.answer_code_exec is not None])
        lines.append("")
        lines.append(f"Execution condition (constrained expression eval): acc={acc_code_exec:.4f}")

    summary_path = os.path.join(args.outdir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote: {csv_path}\nWrote: {summary_path}")

# ------------------------------- CLI ----------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=300, help='total problems (split across digits)')
    p.add_argument('--digits', type=int, nargs='+', default=[2,4,8])
    p.add_argument('--kinds', type=str, nargs='+', default=['add','sub','mul'], choices=['add','sub','mul','mix'])
    p.add_argument('--backend', type=str, default='dummy', choices=['dummy','openai','hf'])
    p.add_argument('--model', type=str, default='gpt-4o', help='OpenAI model name or HF repo/path when --backend=hf')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--hf_dtype', type=str, default='auto', choices=['auto','float16','bfloat16','float32'])
    p.add_argument('--hf_device_map', type=str, default='auto')
    p.add_argument('--hf_trust_remote_code', action='store_true')
    p.add_argument('--max_tokens', type=int, default=256)
    p.add_argument('--exec_code', action='store_true', help='evaluate fenced code expression safely (optional third condition)')
    p.add_argument('--outdir', type=str, default='out')
    p.add_argument('--log_every', type=int, default=50)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
