#!/usr/bin/env python3
"""
Compare Natural-Language CoT vs Code CoT on arithmetic, dynamic programming, and ILP tasks.
- DP: LCS (length), 0/1 Knapsack (max value), Rod Cutting (max revenue)
- ILP: Assignment (min cost), Production Planning (max profit), Balanced Partition (min diff)
Code-CoT executes in a sandboxed subprocess that ALLOWS imports (e.g., PuLP).

Examples:
  # Arithmetic only (unchanged)
  python cot_tb_ext.py --backend hf --model google/gemma-2-9b-it \
    --n 50 --digits 8 9 10 --kinds add sub mul --exec_code --outdir out_hf

  # Mix arithmetic + DP + ILP (use --kinds to pick)
  python cot_tb_ext.py --backend hf --model google/gemma-2-9b-it \
    --n 60 --digits 6 8 --kinds add lcs knap rod ilp_assign ilp_prod ilp_partition \
    --exec_code --outdir out_mix

TensorBoard:
  tensorboard --logdir <outdir>/tb
"""

from __future__ import annotations
import argparse, os, json, re, math, random, time, sys, textwrap, tempfile, subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ------------------------------- Helpers (safe ints) -------------------------------

def sample_int(digits: int, rng: random.Random) -> int:
    if digits <= 0:
        raise ValueError("digits must be >=1")
    lo = 10**(digits-1)
    hi = 10**digits - 1
    return rng.randint(lo, hi)

def randint(rng: random.Random, a:int, b:int) -> int:
    return rng.randint(a, b)

def choice(rng: random.Random, xs: List[Any]) -> Any:
    return xs[rng.randint(0, len(xs)-1)]

# ------------------------------- Problem definitions -------------------------------

@dataclass
class Problem:
    """
    kind:
      Arithmetic: 'add' | 'sub' | 'mul' | 'mix'
      DP:         'lcs' | 'knap' | 'rod'
      ILP:        'ilp_assign' | 'ilp_prod' | 'ilp_partition'
    digits: integer "size" bucket (kept for TB grouping / CSV)
    a,b: used for arithmetic; ignored otherwise
    payload: problem-specific data dict
    """
    kind: str
    digits: int
    a: int = 0
    b: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)

    # ---------- human-readable statement for the LLM ----------
    def text(self) -> str:
        k = self.kind
        if k == 'add':  return f"Compute: {self.a} + {self.b}"
        if k == 'sub':  return f"Compute: {self.a} - {self.b}"
        if k == 'mul':  return f"Compute: {self.a} * {self.b}"
        if k == 'mix':  return f"Compute: ({self.a} + {self.b}) * {self.a}"

        if k == 'lcs':
            s = self.payload["s"]; t = self.payload["t"]
            return (f"LCS length problem: Given two strings s='{s}' and t='{t}', "
                    f"return the integer length of their longest common subsequence.")
        if k == 'knap':
            w = self.payload["weights"]; v = self.payload["values"]; C = self.payload["capacity"]
            return (f"0/1 Knapsack: weights={w}, values={v}, capacity={C}. "
                    f"Return the integer maximum achievable value.")
        if k == 'rod':
            P = self.payload["prices"]; L = self.payload["length"]
            return (f"Rod Cutting: rod length={L}, prices for lengths 1..{len(P)} are {P}. "
                    f"Return the integer maximum revenue.")
        if k == 'ilp_assign':
            C = self.payload["costs"]
            return (f"Assignment ILP (min cost): cost matrix={C}. "
                    f"Assign each job to a unique machine to minimize total cost. "
                    f"Return the integer minimum cost.")
        if k == 'ilp_prod':
            prof = self.payload["profits"]; R = self.payload["resources"]; cap = self.payload["capacity"]; ub = self.payload["ubound"]
            return (f"Production Planning ILP (max profit): profits={prof}, resource_matrix={R}, capacity={cap}, upper_bounds={ub}. "
                    f"Choose integer units per product to maximize profit subject to resource constraints. "
                    f"Return the integer maximum profit.")
        if k == 'ilp_partition':
            pops = self.payload["values"]
            return (f"Balanced Partition ILP: populations={pops}. Partition into two groups to minimize absolute difference "
                    f"of group sums. Return the integer minimum difference.")
        raise ValueError(f"unknown kind: {k}")

    # ---------- exact ground-truth (small instances) ----------
    def ground_truth(self) -> int:
        k = self.kind
        if k == 'add':  return self.a + self.b
        if k == 'sub':  return self.a - self.b
        if k == 'mul':  return self.a * self.b
        if k == 'mix':  return (self.a + self.b) * self.a

        if k == 'lcs':
            s = self.payload["s"]; t = self.payload["t"]
            return lcs_len(s, t)
        if k == 'knap':
            w = self.payload["weights"]; v = self.payload["values"]; C = self.payload["capacity"]
            return knapsack_01_max_value(w, v, C)
        if k == 'rod':
            P = self.payload["prices"]; L = self.payload["length"]
            return rod_cut_max(P, L)
        if k == 'ilp_assign':
            C = self.payload["costs"]
            return assignment_min_cost_bruteforce(C)
        if k == 'ilp_prod':
            prof = self.payload["profits"]; R = self.payload["resources"]; cap = self.payload["capacity"]; ub = self.payload["ubound"]
            return prodplan_max_profit_bruteforce(prof, R, cap, ub)
        if k == 'ilp_partition':
            vals = self.payload["values"]
            return balanced_partition_diff(vals)
        raise ValueError(f"unknown kind: {k}")

# ------------------------------- DP solvers (truth) -------------------------------

def lcs_len(s: str, t: str) -> int:
    n, m = len(s), len(t)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            if s[i] == t[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def knapsack_01_max_value(weights: List[int], values: List[int], capacity: int) -> int:
    n = len(weights)
    dp = [0]*(capacity+1)
    for i in range(n):
        w, val = weights[i], values[i]
        for c in range(capacity, w-1, -1):
            dp[c] = max(dp[c], dp[c-w] + val)
    return dp[capacity]

def rod_cut_max(prices: List[int], L: int) -> int:
    dp = [0]*(L+1)
    for x in range(1, L+1):
        best = 0
        for cut in range(1, min(x, len(prices)) + 1):
            best = max(best, prices[cut-1] + dp[x-cut])
        dp[x] = best
    return dp[L]

# ------------------------------- ILP (truth via small brute force) -------------------------------

def assignment_min_cost_bruteforce(C: List[List[int]]) -> int:
    # Square or rectangular cost matrix; assign each job to a unique machine (min rows, cols <= 5).
    import itertools
    m = len(C)       # jobs
    n = len(C[0])    # machines
    # If rectangular, we assign min(m,n) pairs; here keep m == n in generation to simplify.
    best = math.inf
    for perm in itertools.permutations(range(n), r=m):
        cost = sum(C[i][perm[i]] for i in range(m))
        if cost < best:
            best = cost
    return int(best)

def prodplan_max_profit_bruteforce(profits: List[int], R: List[List[int]], cap: List[int], ub: List[int]) -> int:
    # Small integer search on x_i in [0, ub_i]
    best = -10**9
    n = len(profits)
    def ok(x):
        # Resource constraints: for each resource r: sum_j R[r][j]*x[j] <= cap[r]
        for r in range(len(R)):
            s = sum(R[r][j] * x[j] for j in range(n))
            if s > cap[r]: return False
        return True
    def value(x):
        return sum(profits[j] * x[j] for j in range(n))
    # naive nested loops (n <= 4; ub small)
    def dfs(i, cur):
        nonlocal best
        if i == n:
            if ok(cur):
                best = max(best, value(cur))
            return
        for v in range(0, ub[i]+1):
            cur.append(v)
            dfs(i+1, cur)
            cur.pop()
    dfs(0, [])
    return int(best if best != -10**9 else 0)

def balanced_partition_diff(vals: List[int]) -> int:
    # Minimize |sum(S) - sum(~S)| == |total - 2*subset|
    total = sum(vals)
    target = total // 2
    dp = [False]*(target+1)
    dp[0] = True
    for x in vals:
        for s in range(target, x-1, -1):
            dp[s] = dp[s] or dp[s-x]
    for s in range(target, -1, -1):
        if dp[s]:
            return int(abs(total - 2*s))
    return int(total)

# ------------------------------- Dataset synthesis -------------------------------

def rand_string(rng: random.Random, length: int, alphabet: str="abcdxyz") -> str:
    return "".join(choice(rng, list(alphabet)) for _ in range(length))

def make_dataset(n: int, digits_list: List[int], kinds: List[str], seed: int=1) -> List[Problem]:
    rng = random.Random(seed)
    problems: List[Problem] = []
    if not kinds:
        return problems
    per = max(1, n // len(digits_list))
    for digits in digits_list:
        for _ in range(per):
            kind = choice(rng, kinds)

            if kind in ('add','sub','mul','mix'):
                a = sample_int(digits, rng)
                b = sample_int(digits, rng)
                problems.append(Problem(kind=kind, digits=digits, a=a, b=b))

            elif kind == 'lcs':
                L1 = max(2, min(20, digits))     # keep it small
                L2 = max(2, min(20, digits-1 if digits>2 else digits))
                s = rand_string(rng, L1)
                t = rand_string(rng, L2)
                problems.append(Problem(kind='lcs', digits=digits, payload=dict(s=s, t=t)))

            elif kind == 'knap':
                n_items = max(3, min(10, digits))
                weights = [randint(rng, 1, 10) for _ in range(n_items)]
                values  = [randint(rng, 1, 20) for _ in range(n_items)]
                capacity = max(5, int(sum(weights) * 0.4))
                problems.append(Problem(kind='knap', digits=digits, payload=dict(weights=weights, values=values, capacity=capacity)))

            elif kind == 'rod':
                L = max(3, min(20, digits))
                prices = [randint(rng, 1, 15) for _ in range(L)]
                problems.append(Problem(kind='rod', digits=digits, payload=dict(prices=prices, length=L)))

            elif kind == 'ilp_assign':
                n_jobs = max(2, min(5, digits % 6 or 4))  # 2..5
                C = [[randint(rng, 1, 30) for _ in range(n_jobs)] for __ in range(n_jobs)]
                problems.append(Problem(kind='ilp_assign', digits=n_jobs, payload=dict(costs=C)))

            elif kind == 'ilp_prod':
                # small < 5 products, 2 resources; bounds small
                n_prod = max(2, min(4, digits % 5 or 3))
                profits = [randint(rng, 2, 15) for _ in range(n_prod)]
                R = [[randint(rng, 1, 6) for _ in range(n_prod)] for __ in range(2)]
                cap = [randint(rng, 8, 20) for _ in range(2)]
                ub = [randint(rng, 2, 6) for _ in range(n_prod)]
                problems.append(Problem(kind='ilp_prod', digits=n_prod, payload=dict(profits=profits, resources=R, capacity=cap, ubound=ub)))

            elif kind == 'ilp_partition':
                m = max(4, min(14, digits))
                vals = [randint(rng, 1, 40) for _ in range(m)]
                problems.append(Problem(kind='ilp_partition', digits=m, payload=dict(values=vals)))

            else:
                # fallback: default to addition if unknown label sneaks in
                a = sample_int(digits, rng); b = sample_int(digits, rng)
                problems.append(Problem(kind='add', digits=digits, a=a, b=b))

    rng.shuffle(problems)
    return problems[:n]

# ------------------------------- Prompts ------------------------------------

JSON_SCHEMA = (
    "Return only JSON with keys 'rationale' and 'answer'. "
    "'answer' must be a single integer. No extra keys, no text outside JSON."
)

NL_PROMPT = (
    "You are a careful math tutor. Solve the given problem step by step "
    "in clear natural language sentences. Do NOT include fenced code.\n"
    + JSON_SCHEMA + "\n"
    "Problem: {problem}\n"
    "Output format example: {\"rationale\": \"...reasoning...\", \"answer\": 42}"
)

# Code CoT: allow imports (including PuLP), helper functions, variables; require final print(int)
CODE_PROMPT = (
    "You are a precise Python programmer. Put ALL your reasoning as executable Python in a single fenced block.\n"
    "You may import standard libraries and third-party packages (e.g., pulp) and define helper functions.\n"
    "The LAST line MUST be a print(...) of the final integer answer.\n"
    "Do not include prose outside the fenced block.\n"
    + JSON_SCHEMA + "\n"
    "Problem: {problem}\n"
    "Output format example: {\"rationale\": \"```python\\nimport math\\nprint(2+3)\\n```\", \"answer\": 5}"
)

# ------------------------------- LLM Clients --------------------------------

class LLMClient:
    def chat(self, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]]=None) -> str:
        raise NotImplementedError

class DummyClient(LLMClient):
    """Deterministic stub for dry runs: arithmetic correct; other kinds return 0 (flow check)."""
    INT_RE = re.compile(r"Compute:\s*(\d+)\s*([+\-*])\s*(\d+)")
    MIX_RE = re.compile(r"Compute:\s*\((\d+)\s*\+\s*(\d+)\)\s*\*\s*(\d+)")
    def chat(self, model: str, messages: List[Dict[str, str]], max_tokens: int,
             temperature: float, top_p: float, stop: Optional[List[str]] = None) -> str:
        last = messages[-1]["content"]
        ans = 0
        m = self.INT_RE.search(last)
        if m:
            a, op, b = m.groups(); a,b = int(a), int(b)
            ans = a + b if op == '+' else (a - b if op == '-' else a * b)
        else:
            m2 = self.MIX_RE.search(last)
            if m2:
                a,b,c = map(int, m2.groups()); ans = (a+b)*c
            else:
                ans = 0
        is_math_tutor = "math tutor" in last
        if is_math_tutor:
            out = {"rationale": "Compute deterministically in NL.", "answer": ans}
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

def run_code_subprocess(code: str, timeout_s: float = 4.0) -> Optional[int]:
    """
    Sandbox with resource limits. ALLOWS imports (including pulp).
    We keep '-I' (isolated) for safety, but DO NOT use '-S' so site-packages remain discoverable.
    """
    code = textwrap.dedent(code).strip()
    with tempfile.TemporaryDirectory(prefix="cot_exec_") as td:
        pyfile = os.path.join(td, "main.py")
        with open(pyfile, "w") as f:
            f.write(code + "\n")
        # NOTE: allow site-packages for pulp by not passing '-S'
        cmd = [sys.executable, "-I", pyfile]
        env = {"PYTHONHASHSEED": "0"}
        preexec = None
        try:
            import resource
            def _limit():
                # ~2s CPU, 1GB address space, 2MB file writes
                resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
                mem = 1024 * 1024 * 1024
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
        except subprocess.TimeoutExpired:
            return None
        out = (res.stdout or "").strip()
        nums = INT_RE.findall(out)
        if not nums:
            return None
        try:
            return int(nums[-1])
        except Exception:
            return None

def exec_from_rationale(rationale: Optional[str]) -> Optional[int]:
    code = extract_fenced_code(rationale)
    if not code: return None
    return run_code_subprocess(code, timeout_s=4.0)

# ------------------------------- Eval structs ---------------------------------

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

        base_nl = f"{args.model}/nl/sz{pb.digits}/{pb.kind}/i{i}"
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

        base_code = f"{args.model}/code/sz{pb.digits}/{pb.kind}/i{i}"
        tb_text(f"{base_code}/prompt", "Prompt (Code-CoT)", code_prompt)
        tb_text(f"{base_code}/rationale", "Code Rationale (fenced)", code_parsed.rationale or "")
        tb_text(f"{base_code}/raw_json", "Raw Code JSON", code_parsed.raw)
        tb_text(f"{base_code}/answer", "Final Answer (Code)", "" if ans_code is None else str(ans_code))

        # Optional execution via constrained subprocess (supports imports, incl. pulp)
        ans_code_exec = exec_from_rationale(code_parsed.rationale) if args.exec_code else None
        correct_code_exec = int(ans_code_exec == truth) if ans_code_exec is not None else 0
        if args.exec_code:
            tb_text(f"{base_code}/exec_answer", "Executed Answer (subprocess)", "" if ans_code_exec is None else str(ans_code_exec))

        records.append(Record(
            idx=i, problem=problem_text, digits=pb.digits, kind=pb.kind, truth=truth,
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
        w.writerow(["idx","problem","size","kind","truth",
                    "answer_nl","correct_nl",
                    "answer_code","correct_code",
                    "answer_code_exec","correct_code_exec"])
        for r in records:
            w.writerow([r.idx, r.problem, r.digits, r.kind, r.truth,
                        r.answer_nl, r.correct_nl,
                        r.answer_code, r.correct_code,
                        r.answer_code_exec, r.correct_code_exec])

    # Summary
    def acc(xs): return sum(xs)/max(1,len(xs))
    acc_nl = acc([r.correct_nl for r in records])
    acc_code = acc([r.correct_code for r in records])
    has_exec = any(r.answer_code_exec is not None for r in records)
    acc_exec = acc([r.correct_code_exec for r in records if r.answer_code_exec is not None]) if has_exec else float('nan')

    b = sum(1 for r in records if r.correct_code==1 and r.correct_nl==0)
    c = sum(1 for r in records if r.correct_code==0 and r.correct_nl==1)
    p_mc = mcnemar_exact_p(b, c)

    # Per-kind summary (since tasks vary widely)
    by_kind: Dict[str, List[Record]] = {}
    for r in records: by_kind.setdefault(r.kind, []).append(r)

    lines: List[str] = []
    lines.append(f"N={len(records)}  exp_id={exp_id}")
    lines.append(f"Accuracy NL-CoT:   {acc_nl:.4f}")
    lines.append(f"Accuracy Code-CoT: {acc_code:.4f}")
    lines.append(f"Discordant pairs b=code>nl: {b}, c=nl>code: {c}, McNemar exact p={p_mc:.4g}")
    lines.append("")
    lines.append("Per-kind bins:")
    for k in sorted(by_kind.keys()):
        recs = by_kind[k]
        a_nl = acc([x.correct_nl for x in recs])
        a_code = acc([x.correct_code for x in recs])
        b_k = sum(1 for x in recs if x.correct_code==1 and x.correct_nl==0)
        c_k = sum(1 for x in recs if x.correct_code==0 and x.correct_nl==1)
        p_k = mcnemar_exact_p(b_k, c_k)
        lines.append(f"  kind={k:12s}: N={len(recs):3d}  NL={a_nl:.4f}  Code={a_code:.4f}  McNemar p={p_k:.4g}")
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
    p.add_argument('--n', type=int, default=100, help='total problems (split across sizes)')
    p.add_argument('--digits', type=int, nargs='+', default=[6,8,10],
                   help='For arithmetic: number of digits. For DP/ILP: used as size seed (len, items, etc.).')
    p.add_argument('--kinds', type=str, nargs='+',
                   default=['add','sub','mul'],
                   choices=['add','sub','mul','mix','lcs','knap','rod','ilp_assign','ilp_prod','ilp_partition'],
                   help='Which task types to sample from.')
    p.add_argument('--seed', type=int, default=1)

    p.add_argument('--backend', type=str, default='dummy', choices=['dummy','openai','hf'])
    p.add_argument('--model', type=str, default='gpt-4o', help='OpenAI model name or HF repo/path when --backend=hf')
    p.add_argument('--hf_dtype', type=str, default='auto', choices=['auto','float16','bfloat16','float32'])
    p.add_argument('--hf_device_map', type=str, default='auto')
    p.add_argument('--hf_trust_remote_code', action='store_true')

    p.add_argument('--max_tokens', type=int, default=512)
    p.add_argument('--exec_code', action='store_true', help='execute code-CoT in sandboxed subprocess (allows imports incl. pulp)')
    p.add_argument('--outdir', type=str, default='out')
    p.add_argument('--log_every', type=int, default=50)

    # TensorBoard text limits
    p.add_argument('--tb_text_chars', type=int, default=6000)
    p.add_argument('--tb_disable', action='store_true')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
