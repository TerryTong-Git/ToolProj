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
import argparse, os, json, re, math, random, time, sys, textwrap, tempfile, subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ------------------------------- Utilities ----------------------------------

INT_RE = re.compile(r"[-+]?[0-9]+")

def _try_import_pulp():
    try:
        import pulp  # type: ignore
        return pulp
    except Exception:
        return None

# ------------------------------- Dataset ------------------------------------

def sample_int(digits: int, rng: random.Random) -> int:
    if digits <= 0:
        raise ValueError("digits must be >=1")
    lo = 10**(digits-1)
    hi = 10**digits - 1
    return rng.randint(lo, hi)

def rand_string(rng: random.Random, alpha="abcd", lo=5, hi=12) -> str:
    n = rng.randint(lo, hi)
    return "".join(rng.choice(alpha) for _ in range(n))

@dataclass
class Problem:
    kind: str
    # arithmetic fields (optional)
    digits: int = 0
    a: int = 0
    b: int = 0
    # general payload for DP/ILP
    data: Optional[Dict[str, Any]] = None

    def text(self) -> str:
        k = self.kind
        if k in ("add","sub","mul","mix"):
            if k == 'add':
                return f"Compute: {self.a} + {self.b}"
            if k == 'sub':
                return f"Compute: {self.a} - {self.b}"
            if k == 'mul':
                return f"Compute: {self.a} * {self.b}"
            if k == 'mix':
                return f"Compute: ({self.a} + {self.b}) * {self.a}"
        elif k == "lcs":
            s = self.data["s"]; t = self.data["t"]
            return f"Compute the length of the Longest Common Subsequence (LCS) between strings:\nS = \"{s}\"\nT = \"{t}\""
        elif k == "knap":
            w = self.data["weights"]; v = self.data["values"]; C = self.data["capacity"]
            return ("0/1 Knapsack: Given item weights W and values V and capacity C, "
                    "compute the maximum total value.\n"
                    f"W = {w}\nV = {v}\nC = {C}")
        elif k == "rod":
            prices = self.data["prices"]
            N = len(prices)
            return ("Rod cutting: Given a rod of length N and price list P[1..N], "
                    "compute the maximum obtainable revenue.\n"
                    f"N = {N}\nP = {prices}")
        elif k == "ilp_assign":
            C = self.data["cost"]
            return ("Assignment problem: Given an n×n cost matrix C, assign each worker to one task "
                    "minimizing the total cost. Return the minimum total cost as an integer. \n"
                    f"C = {C}")
        elif k == "ilp_prod":
            prof = self.data["profit"]; cons = self.data["consumption"]; caps = self.data["capacity"]; ub = self.data["upper_bound"]
            return ("Production planning: Choose integer quantities x_j ≥ 0 to maximize total profit sum_j profit[j]*x_j, "
                    "subject to resource constraints sum_j consumption[i][j]*x_j ≤ capacity[i]. Return the max profit.\n"
                    f"profit = {prof}\nconsumption (rows=resources) = {cons}\ncapacity = {caps}\nupper_bounds = {ub}")
        elif k == "ilp_partition":
            w = self.data["weights"]
            return ("Partition: Split the items into two groups to minimize the absolute difference between the sums. "
                    "Return the minimum difference as an integer.\n"
                    f"weights = {w}")
        raise ValueError("unknown kind")

    # ---- Ground-truth evaluators ----
    def ground_truth(self) -> int:
        k = self.kind
        if k in ("add","sub","mul","mix"):
            if k == 'add': return self.a + self.b
            if k == 'sub': return self.a - self.b
            if k == 'mul': return self.a * self.b
            if k == 'mix': return (self.a + self.b) * self.a
        elif k == "lcs":
            return lcs_len(self.data["s"], self.data["t"])
        elif k == "knap":
            return knap_01_max_value(self.data["weights"], self.data["values"], self.data["capacity"])
        elif k == "rod":
            return rod_cut_max(self.data["prices"])
        elif k == "ilp_assign":
            return assignment_min_cost(self.data["cost"])
        elif k == "ilp_prod":
            return prodplan_max_profit(self.data)
        elif k == "ilp_partition":
            return partition_min_diff(self.data["weights"])
        raise ValueError("unknown kind")

# ---- DP helpers ----

def lcs_len(s: str, t: str) -> int:
    m, n = len(s), len(t)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if s[i] == t[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def knap_01_max_value(W: List[int], V: List[int], C: int) -> int:
    n = len(W)
    dp = [0]*(C+1)
    for i in range(n):
        w, val = W[i], V[i]
        for c in range(C, w-1, -1):
            dp[c] = max(dp[c], dp[c-w] + val)
    return dp[C]

def rod_cut_max(P: List[int]) -> int:
    n = len(P)
    dp = [0]*(n+1)
    for L in range(1, n+1):
        best = P[L-1]  # one piece of length L
        for k in range(1, L):
            best = max(best, dp[k] + dp[L-k])
        dp[L] = best
    return dp[n]

# ---- ILP / combinatorial helpers ----

def assignment_min_cost(C: List[List[int]]) -> int:
    n = len(C)
    pulp = _try_import_pulp()
    if pulp is not None:
        prob = pulp.LpProblem("assign", pulp.LpMinimize)
        x = [[pulp.LpVariable(f"x_{i}_{j}", 0, 1, cat="Binary") for j in range(n)] for i in range(n)]
        prob += pulp.lpSum(C[i][j]*x[i][j] for i in range(n) for j in range(n))
        for i in range(n):
            prob += pulp.lpSum(x[i][j] for j in range(n)) == 1
        for j in range(n):
            prob += pulp.lpSum(x[i][j] for i in range(n)) == 1
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        val = int(round(pulp.value(prob.objective)))
        return val
    # brute-force permutations (n small, e.g., <=5)
    import itertools
    best = None
    for perm in itertools.permutations(range(n)):
        cost = sum(C[i][perm[i]] for i in range(n))
        best = cost if best is None else min(best, cost)
    return int(best)

def prodplan_max_profit(d: Dict[str, Any]) -> int:
    profit: List[int] = d["profit"]
    consumption: List[List[int]] = d["consumption"]  # R x P
    capacity: List[int] = d["capacity"]              # R
    upper: List[int] = d["upper_bound"]              # P
    R = len(consumption)
    P = len(profit)
    pulp = _try_import_pulp()
    if pulp is not None:
        prob = pulp.LpProblem("prodplan", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=int(upper[j]), cat="Integer") for j in range(P)]
        prob += pulp.lpSum(profit[j]*x[j] for j in range(P))
        for i in range(R):
            prob += pulp.lpSum(consumption[i][j]*x[j] for j in range(P)) <= capacity[i]
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        val = int(round(pulp.value(prob.objective)))
        return val
    # bounded brute force (P<=4, bounds small)
    best = 0
    def dfs(j, cur_prof, use):
        nonlocal best
        if j == P:
            best = max(best, cur_prof); return
        for q in range(0, upper[j]+1):
            ok = True
            for i in range(R):
                if use[i] + consumption[i][j]*q > capacity[i]:
                    ok = False; break
            if not ok: break
            for i in range(R): use[i] += consumption[i][j]*q
            dfs(j+1, cur_prof + profit[j]*q, use)
            for i in range(R): use[i] -= consumption[i][j]*q
    dfs(0, 0, [0]*R)
    return int(best)

def partition_min_diff(weights: List[int]) -> int:
    total = sum(weights)
    target = total//2
    possible = 1  # bitset; bit k means sum k achievable
    for w in weights:
        possible = possible | (possible << w)
    # scan for achievable sum closest to target
    best = None
    for s in range(target, -1, -1):
        if (possible >> s) & 1:
            best = s
            break
    if best is None:
        return total
    return int(total - 2*best)

# ------------------------------- Utilities ----------------------------------

def rand_string(rng: random.Random, alpha="abcd", n: Optional[int]=None, lo=5, hi=12) -> str:
    if n is None:
        n = rng.randint(lo, hi)
    return "".join(rng.choice(alpha) for _ in range(n))

# ------------------------------- Generators ---------------------------------

def make_problem(rng: random.Random, kind: str, digits: Optional[int]=None) -> Problem:
    """
    Use `digits` as a single hardness knob:
      - add/sub/mul/mix    : same as before (number magnitude ~ 10^digits)
      - lcs                : |S| ≈ |T| ≈ digits
      - knap               : n_items ≈ digits; weights/values scale with digits
      - rod                : rod length N ≈ digits
      - ilp_assign         : n x n with n ≈ min(digits, 7)   (cap for runtime)
      - ilp_prod           : P≈min(2+digits//3, 6), R≈min(2+digits//4, 4); bounds scale
      - ilp_partition      : n_items ≈ min(digits, 24); magnitudes scale with digits
    """
    d = digits if digits is not None else rng.choice([2,4,8])

    if kind in ("add","sub","mul","mix"):
        a = sample_int(d, rng); b = sample_int(d, rng)
        if kind == "sub" and b > a: a, b = b, a
        return Problem(kind=kind, digits=d, a=a, b=b, data=None)

    if kind == "lcs":
        n = max(2, int(d))
        s = rand_string(rng, alpha="abcd", n=n)
        t = rand_string(rng, alpha="abcd", n=n + rng.randint(-1, 1))
        return Problem(kind="lcs", digits=d, data={"s": s, "t": t})

    if kind == "knap":
        n_items = max(3, int(d))
        # scale magnitudes gently with d to keep runtimes sane
        w_max = max(5, 2*d)
        v_max = max(10, 4*d)
        weights = [rng.randint(1, w_max) for _ in range(n_items)]
        values  = [rng.randint(1, v_max) for _ in range(n_items)]
        capacity = max(1, int(0.5 * sum(weights)))
        return Problem(kind="knap", digits=d, data={"weights": weights, "values": values, "capacity": capacity})

    if kind == "rod":
        N = max(2, int(d))
        price_max = max(5, 3*d)
        prices = [rng.randint(1, price_max) for _ in range(N)]
        return Problem(kind="rod", digits=d, data={"prices": prices})

    if kind == "ilp_assign":
        n = max(2, min(int(d), 7))  # cap n for brute-force fallback safety
        C = [[rng.randint(1, max(6, 3*d)) for _ in range(n)] for __ in range(n)]
        return Problem(kind="ilp_assign", digits=d, data={"cost": C})

    if kind == "ilp_prod":
        # scale #products/#resources and magnitudes with d, but cap to keep fallback feasible
        P = max(2, min(2 + d // 3, 6))
        R = max(2, min(2 + d // 4, 4))
        profit = [rng.randint(3, max(8, 3*d)) for _ in range(P)]
        consumption = [[rng.randint(1, max(3, d)) for _ in range(P)] for __ in range(R)]
        # capacity scaled so some slack exists; upper bounds smallish (<= 10)
        capacity = [rng.randint(max(6, 2*d), max(10, 4*d)) for _ in range(R)]
        upper = []
        for j in range(P):
            ub_j = min(10, min((capacity[i] // max(1, consumption[i][j]) for i in range(R)), default=10))
            upper.append(int(max(3, ub_j)))
        return Problem(kind="ilp_prod", digits=d, data={
            "profit": profit, "consumption": consumption, "capacity": capacity, "upper_bound": upper
        })

    if kind == "ilp_partition":
        n_items = max(4, min(int(d), 24))
        w_max   = max(6, 3*d)
        weights = [rng.randint(1, w_max) for _ in range(n_items)]
        return Problem(kind="ilp_partition", digits=d, data={"weights": weights})

    raise ValueError(f"unknown kind: {kind}")

def make_dataset(n: int, digits_list: List[int], kinds: List[str], seed: int=1) -> List[Problem]:
    """
    Balance over (kind × digits) so MI/acc buckets are well-populated.
    """
    rng = random.Random(seed)
    problems: List[Problem] = []
    K = max(1, len(kinds)); D = max(1, len(digits_list))
    per = max(1, n // (K * D))
    for d in digits_list:
        for k in kinds:
            for _ in range(per):
                problems.append(make_problem(rng, k, d))
    while len(problems) < n:
        k = rng.choice(kinds); d = rng.choice(digits_list)
        problems.append(make_problem(rng, k, d))
    rng.shuffle(problems)
    return problems[:n]


# ------------------------------- Prompts ------------------------------------

JSON_SCHEMA = (
    "Return only JSON with keys 'rationale' and 'answer'. "
    "'answer' must be a single integer. No extra keys, no text outside JSON."
)

# IMPORTANT: double braces {{ }} for format literals
NL_PROMPT = (
    "You are a careful and accurate problem solver. Solve the given problem step by step "
    "in clear natural language sentences. Do not include any code or formulas in backticks.\n"
    + JSON_SCHEMA + "\n"
    "Problem: {problem}\n"
    "Output format example: {{\"rationale\": \"...natural language steps...\", \"answer\": 42}}"
)

CODE_PROMPT = (
    "You are a precise and accurate Python programmer. Put ALL your reasoning as executable Python in a single fenced block.\n"
    "You may use imports and helper functions: in the python environment, you have access to scipy, numpy, and PuLP. The LAST line MUST be a print(...) of the final integer.\n"
    "Do not include prose outside the fenced block.\n"
    + JSON_SCHEMA + "\n"
    "Problem: {problem}\n"
    "Output format example: {{\"rationale\": \"```python\\n# your code here\\nprint(42)\\n```\", \"answer\": 42}}"
)

# ------------------------------- LLM Clients --------------------------------

class LLMClient:
    def chat(self, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]]=None) -> str:
        raise NotImplementedError

class DummyClient(LLMClient):
    """Deterministic stub: returns correct integer for known templates; else 0."""
    def chat(self, model: str, messages: List[Dict[str, str]], max_tokens: int,
             temperature: float, top_p: float, stop: Optional[List[str]] = None) -> str:
        last = messages[-1]["content"]
        ans = 0
        # arithmetic quick parse
        m = re.search(r"Compute:\s*(\d+)\s*([+\-*])\s*(\d+)", last)
        if m:
            a, op, b = m.groups(); a, b = int(a), int(b)
            ans = a + b if op == '+' else (a - b if op == '-' else a * b)
        else:
            m2 = re.search(r"Compute:\s*\((\d+)\s*\+\s*(\d+)\)\s*\*\s*(\d+)", last)
            if m2: a,b,c = map(int, m2.groups()); ans = (a+b)*c
        # for other kinds, just 0 to keep it simple for dry-run
        is_nl = "problem solver" in last.lower()
        if is_nl:
            out = {"rationale": "Solve deterministically.", "answer": ans}
        else:
            out = {"rationale": f"```python\nprint({ans})\n```", "answer": ans}
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
            model=model, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop,
        )
        return resp.choices[0].message.content

class HFLocalClient(LLMClient):
    """Vanilla Hugging Face transformers inference (no vLLM)."""
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
            pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id),
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
    try:
        start = s.index('{'); end = s.rindex('}')
        frag = s[start:end+1]
        return json.loads(frag)
    except Exception:
        return None

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
        m = INT_RE.findall(raw)
        if not m:
            return Parsed(raw, False, None, None, "no-json-no-int")
        try:
            ans = int(m[-1])
            return Parsed(raw, True, ans, None, "json-missing")
        except Exception:
            return Parsed(raw, False, None, None, "int-parse-failed")
    if not (isinstance(obj, dict) and 'answer' in obj and 'rationale' in obj):
        return Parsed(raw, False, None, None, "bad-json-keys")
    ans = obj['answer']
    try:
        ans = int(ans)
    except Exception:
        if isinstance(ans, str):
            m = INT_RE.findall(ans); ans = int(m[-1]) if m else None
        else:
            ans = None
    return Parsed(raw, ans is not None, ans, obj.get('rationale'), None if ans is not None else "answer-not-int")

# ----------------------- Code execution (subprocess sandbox) ----------------

FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\s*\n([\s\S]*?)\n```", re.MULTILINE)

def extract_fenced_code(rationale: Optional[str]) -> Optional[str]:
    if not rationale: return None
    m = FENCE_RE.search(rationale)
    if not m: return None
    return m.group(1).strip()
def run_code_subprocess(code: str, timeout_s: float = 3.0, allow_imports: bool = True,
                        exec_prefix: Optional[List[str]] = None,
                        exec_python: Optional[str] = None) -> Dict[str, Any]:
    """Run code and always return a dict with stable keys."""
    import time as _time
    code = textwrap.dedent(code).strip()
    t0 = _time.time()
    stdout = ""; stderr = ""; value = None
    ok = False; timeout = False; retcode = None; exc = None

    with tempfile.TemporaryDirectory(prefix="cot_exec_") as td:
        pyfile = os.path.join(td, "main.py")
        with open(pyfile, "w") as f:
            f.write(code + "\n")
        pybin = exec_python or sys.executable
        core = [pybin, pyfile] if allow_imports else [pybin, "-I", "-S", pyfile]
        cmd = (exec_prefix or []) + core

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        preexec = None
        try:
            import resource
            def _limit():
                resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
                mem = 1_000 * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                resource.setrlimit(resource.RLIMIT_FSIZE, (2_000_000, 2_000_000))
            preexec = _limit
        except Exception:
            preexec = None

        try:
            res = subprocess.run(
                cmd, cwd=td, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=timeout_s, check=False, text=True, env=env, preexec_fn=preexec
            )
            stdout = (res.stdout or "")
            stderr = (res.stderr or "")
            retcode = res.returncode
            # extract last integer printed
            nums = INT_RE.findall(stdout)
            if nums:
                try:
                    value = int(nums[-1])
                    ok = True
                except Exception:
                    value = None
        except subprocess.TimeoutExpired:
            timeout = True
            stderr = "TimeoutExpired"
        except Exception as e:
            exc = repr(e)
            stderr = (stderr + "\n" + exc) if stderr else exc

    return {
        "ok": bool(ok),
        "value": value,
        "stdout": stdout,
        "stderr": stderr,
        "retcode": retcode,
        "timeout": bool(timeout),
        "duration_s": time.time() - t0,
    }

def exec_from_rationale(rationale: Optional[str], allow_imports: bool = True,
                        exec_prefix: Optional[List[str]] = None,
                        exec_python: Optional[str] = None) -> Dict[str, Any]:
    code = extract_fenced_code(rationale)
    if not code:
        return {"ok": False, "value": None, "stdout": "", "stderr": "no_fenced_code",
                "retcode": None, "timeout": False, "duration_s": 0.0}
    return run_code_subprocess(code, timeout_s=3.0, allow_imports=allow_imports,
                               exec_prefix=exec_prefix, exec_python=exec_python)


# ------------------------------- Evaluation ---------------------------------

@dataclass
class Record:
    idx: int
    problem: str
    kind: str
    digits: int
    truth: int
    answer_nl: Optional[int]
    correct_nl: int
    answer_code: Optional[int]
    correct_code: int
    answer_code_exec: Optional[int]
    correct_code_exec: int
    raw_nl: str
    raw_code: str
    
    exec_ok: Optional[int] = None
    exec_retcode: Optional[int] = None
    exec_timeout: Optional[int] = None
    exec_stdout: Optional[str] = None
    exec_stderr: Optional[str] = None

def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    def binom_cdf_leq(n, k):
        s = 0.0
        for i in range(0, k+1): s += math.comb(n, i)
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

    # TensorBoard
    os.makedirs(args.outdir, exist_ok=True)
    exp_id = time.strftime("run_%Y%m%d_%H%M%S")
    tb = None if args.tb_disable else SummaryWriter(log_dir=os.path.join(args.outdir, "tb", exp_id))

    def tb_text(tag: str, title: str, body: str, step: int = 0):
        if tb is None: return
        body = body or ""
        n = args.tb_text_chars
        body_show = body if len(body) <= n else (body[:max(0, n-3)] + "...")
        tb.add_text(tag, f"**{title}**\n\n```\n{body_show}\n```", global_step=step)

    records: List[Record] = []

    for i, pb in enumerate(tqdm(problems, total=len(problems), desc='eval')):
        problem_text = pb.text()
        truth = pb.ground_truth()

        # NL
        nl_prompt = NL_PROMPT.format(problem=problem_text)
        nl_raw = client.chat(
            model=args.model,
            messages=[{"role":"user","content": nl_prompt}],
            max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, stop=None,
        )
        nl_parsed = parse_response(nl_raw)
        ans_nl = nl_parsed.answer
        correct_nl = int(ans_nl == truth)

        base_nl   = f"{args.model}/nl/d{pb.digits}/{pb.kind}/i{i}"
        tb_text(f"{base_nl}/prompt", "Prompt (NL-CoT)", nl_prompt)
        tb_text(f"{base_nl}/rationale", "NL Rationale", nl_parsed.rationale or "")
        tb_text(f"{base_nl}/raw_json", "Raw NL JSON", nl_parsed.raw)
        tb_text(f"{base_nl}/answer", "Final Answer (NL)", "" if ans_nl is None else str(ans_nl))

        # Code
        code_prompt = CODE_PROMPT.format(problem=problem_text)
        code_raw = client.chat(
            model=args.model,
            messages=[{"role":"user","content": code_prompt}],
            max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, stop=None,
        )
        code_parsed = parse_response(code_raw)
        ans_code = code_parsed.answer
        correct_code = int(ans_code == truth)

        base_code = f"{args.model}/code/d{pb.digits}/{pb.kind}/i{i}"
        tb_text(f"{base_code}/prompt", "Prompt (Code-CoT)", code_prompt)
        tb_text(f"{base_code}/rationale", "Code Rationale (fenced)", code_parsed.rationale or "")
        tb_text(f"{base_code}/raw_json", "Raw Code JSON", code_parsed.raw)
        tb_text(f"{base_code}/answer", "Final Answer (Code)", "" if ans_code is None else str(ans_code))

        # Optional execution via constrained subprocess (imports allowed)
        ans_code_exec = {"ok": False, "value": None, "stdout": "", "stderr": "", "retcode": None, "timeout": False}
        if args.exec_code:
            ans_code_exec = exec_from_rationale(code_parsed.rationale, allow_imports=True) if args.exec_code else None
        correct_code_exec = int(ans_code_exec == truth) if ans_code_exec is not None else 0
        tb_text(f"{base_code}/exec_answer", "Executed Answer (subprocess)", "" if ans_code_exec is None else str(ans_code_exec))
        tb_text(f"{base_code}/exec_stdout", "Executed STDOUT", ans_code_exec.get("stdout",""))
        tb_text(f"{base_code}/exec_stderr", "Executed STDERR", ans_code_exec.get("stderr",""))
        tb_text(f"{base_code}/exec_meta",   "Exec Meta", f"retcode={ans_code_exec.get('retcode')} timeout={ans_code_exec.get('timeout')} ok={ans_code_exec.get('ok')} value={ans_code_exec}")
        records.append(Record(
            idx=i, problem=problem_text, kind=pb.kind, digits=pb.digits, truth=truth,
            answer_nl=ans_nl, correct_nl=correct_nl,
            answer_code=ans_code, correct_code=correct_code,
            answer_code_exec=ans_code_exec, correct_code_exec=correct_code_exec,
            raw_nl=nl_raw, raw_code=code_raw,
        ))

    # CSV
    import csv
    csv_path = os.path.join(args.outdir, f'{exp_id}_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["idx","kind","digits","truth",
                    "answer_nl","correct_nl",
                    "answer_code","correct_code",
                    "answer_code_exec","correct_code_exec","problem"])
        for r in records:
            w.writerow([r.idx, r.kind, r.digits, r.truth,
                        r.answer_nl, r.correct_nl,
                        r.answer_code, r.correct_code,
                        r.answer_code_exec, r.correct_code_exec, r.problem])

    # Summary (overall + per kind)
    def acc(xs): return sum(xs)/max(1,len(xs))
    acc_nl = acc([r.correct_nl for r in records])
    acc_code = acc([r.correct_code for r in records])
    has_exec = any(r.answer_code_exec is not None for r in records)
    acc_exec = acc([r.correct_code_exec for r in records if r.answer_code_exec is not None]) if has_exec else float('nan')

    b = sum(1 for r in records if r.correct_code==1 and r.correct_nl==0)
    c = sum(1 for r in records if r.correct_code==0 and r.correct_nl==1)
    p_mc = mcnemar_exact_p(b, c)

    by_kind: Dict[str, List[Record]] = {}
    for r in records: by_kind.setdefault(r.kind, []).append(r)

    lines: List[str] = []
    lines.append(f"N={len(records)}  exp_id={exp_id}")
    lines.append(f"Accuracy NL-CoT (overall):   {acc_nl:.4f}")
    lines.append(f"Accuracy Code-CoT (overall): {acc_code:.4f}")
    lines.append(f"Discordant pairs b=code>nl: {b}, c=nl>code: {c}, McNemar exact p={p_mc:.4g}")
    lines.append("Per-kind bins:")
    for k in sorted(by_kind):
        recs = by_kind[k]
        a_nl   = acc([x.correct_nl   for x in recs])
        a_code = acc([x.correct_code for x in recs])
        # Exec accuracy only over items that actually executed
        exec_vals = [x.correct_code_exec for x in recs if x.answer_code_exec is not None]
        a_exec = (sum(exec_vals) / len(exec_vals)) if exec_vals else float('nan')

        lines.append(f"  kind={k:12s}: N={len(recs):3d}  NL={a_nl:.4f}  Code={a_code:.4f}  Exec={a_exec:.4f}")
        if tb is not None:
            tb.add_scalar(f"{args.model}/acc_nl/{k}",   a_nl)
            tb.add_scalar(f"{args.model}/acc_code/{k}", a_code)
            if exec_vals:  # only log if anything executed
                tb.add_scalar(f"{args.model}/acc_exec/{k}", a_exec)

    if has_exec:
        lines.append("")
        lines.append(f"Execution condition (subprocess): acc={acc_exec:.4f}")

    summary_path = os.path.join(args.outdir, f'{exp_id}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote: {csv_path}\nWrote: {summary_path}")
    if tb is not None:
        tb.flush(); tb.close()

# ------------------------------- CLI ----------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100, help='total problems (balanced over kinds)')
    p.add_argument('--digits', type=int, nargs='+', default=[2,4,8],
               help='Global hardness levels. For arithmetic: number magnitude. '
                    'LCS: string length; knap: #items; rod: rod length; '
                    'ilp_assign: n×n size; ilp_prod: scales products/resources/bounds; '
                    'ilp_partition: #items.')
    p.add_argument('--kinds', type=str, nargs='+',
                   default=['add','sub','mul','lcs','knap','rod','ilp_assign','ilp_prod','ilp_partition'],
                   choices=['add','sub','mul','mix','lcs','knap','rod','ilp_assign','ilp_prod','ilp_partition'])
    p.add_argument('--seed', type=int, default=1)

    p.add_argument('--backend', type=str, default='dummy', choices=['dummy','openai','hf'])
    p.add_argument('--model', type=str, default='gpt-4o', help='OpenAI model name or HF repo/path when --backend=hf')
    p.add_argument('--hf_dtype', type=str, default='blfloat16', choices=['auto','float16','bfloat16','float32'])
    p.add_argument('--hf_device_map', type=str, default='auto')
    p.add_argument('--hf_trust_remote_code', action='store_true')

    p.add_argument('--max_tokens', type=int, default=512)
    p.add_argument('--exec_code', action='store_true', help='execute code-CoT in sandboxed subprocess (imports allowed)')
    p.add_argument('--outdir', type=str, default='out')
    p.add_argument('--log_every', type=int, default=50)

    # TensorBoard text limits
    p.add_argument('--tb_text_chars', type=int, default=6000)
    p.add_argument('--tb_disable', action='store_true')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
