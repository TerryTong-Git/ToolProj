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
from transformers import AutoTokenizer
try:
    from vllm import LLM as VLLMEngine, SamplingParams
except Exception as _vllm_import_err:
    VLLMEngine = None
    SamplingParams = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.deterministic = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ------------------------------- Utilities ----------------------------------

INT_RE  = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')


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

    # perhap save this sometwhere, make bins more fine-grained. Submit PR and review it before merging. 
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
"""
You are tasked with solving an algorithmic problem by reasoning through it step by step using a chain-of-thought approach expressed in clear, natural language. Begin by thoroughly analyzing the problem, breaking it down into manageable parts, and explaining your thought process in detail. The problem is given after <|Problem|>. You should fill in <|Response|>. You are never allowed to use code. After fully reasoning through the problem in natural language, output <|Final Answer|> and consolidate your final thoughts into a JSON dictionary containing two keys:

- "rationale": a comprehensive explanation summarizing your reasoning and approach to the problem.
- "answer": give the final requested answer as an integer.

Ensure your explanation is clear, logically structured, and leads naturally to the final answer provided in the JSON output. 

Examples:

===========================================================================

<|Problem|>
Minimum Vertex Cover: Given an undirected graph G=(V,E), choose the smallest set of vertices that touches every edge. Return the cover size.
V = {{0,1,2,3}}
E = {{(0,1), (1,2), (2,3)}}

<|Response|>
The edges form a simple path: (0 1 2 3). To cover every edge, choose vertices so that each edge has at least one endpoint selected. Picking {{1,2}} covers (0,1) via 1, (1,2) via either 1 or 2, and (2,3) via 2. That uses 2 vertices. Any single vertex cannot cover all three edges, so 2 is minimal.

<|Final Answer|>
{{
\"rationale\": \"The graph is a path of length 3. Selecting the middle two vertices {{1,2}} covers all edges, and a cover of size 1 is impossible, so the minimum cover size is 2.\",
\"answer\": 2
}}

============================================================================

<|Problem|>
Coin Change (min coins): Given coin denominations and a target amount, compute the minimum number of coins to make the amount (use -1 if impossible).
coins = [1, 3, 4]
amount = 6

<|Response|>
I consider the smallest number of coins to total 6. Using 4 first, the remainder is 2, which needs two 1s for a total of 3 coins. Using 3 first, the remainder is 3; another 3 makes 6 with 2 coins total. Using only 1s needs 6 coins. The best among these is 2 coins (3 + 3).

<|Final Answer|>
{{
\"rationale\": \"Compare constructions: 4+1+1 uses 3 coins; 3+3 uses 2 coins; six 1s uses 6 coins. The minimum is achieved by 3+3 with 2 coins.\",
\"answer\": 2
}}

============================================================================

<|Problem|>
Compute: 84 ÷ 6

<|Response|>
Divide 84 by 6. Since 6 × 14 = 84, the quotient is 14.

<|Final Answer|>
{{
\"rationale\": \"84 divided by 6 equals 14 because 6 × 14 = 84.\",
\"answer\": 14
}}

============================================================================

<|Problem|>
{problem}

<|Response|>
"""
)

CODE_PROMPT = (
"""
You are an expert algorithm problem solver who reasons entirely in Python code by generating a piece of code, then simulating its execution after <|Execution Simulation|>. When you generate code, write clear, executable code. You can use Math, Numpy, Torch, PuLP, Scipy, and Pandas. Ensure proper syntax, indentation, definitions, and variable instantiation. When simulating execution, write out the line number, the variable state, and behave like a single-cycle cpu.  The last line of the program must print the final result. The problem is given after <|Problem|>. You should fill in <|Response|>. After response, give a piece of code, then give <|Execution Simulation|> and try to simulate the code.  You MUST write out [BEGIN] and [DONE] before the execution simulation. Moreover, after simulating the program, report out the program that you attempted to simulate and the answer you got from simulation. Do this by first outputting <|Final Answer|>, then by producing a JSON dictionary with two keys:

- "rationale": the complete Python code solution, enclosed in a code block.
- "answer": the result from executing the code, should be an integer.

After generating a short Python program that defines a single function `f(...)` and then calls it as `output = f(<args>)`, produce ONLY the execution trace in the exact format below.

## Required format
[BEGIN]
state: {{}}
line: def f(<params>):
state: {{"f": "<callable_object f>"}}
line: output = f(<args>)
state: {{<callee_locals_after_param_binding>}}
line: <next_source_line_or_guard>
state: {{<locals_after_effect_or_after_guard_eval>}}
... (repeat for every executed line, including each re-check of while/if guards)
line: return <expr>
state: {{"f": "<callable_object f>", "output": <final_integer>}}
[DONE]

## Tracing rules (must follow exactly)
- “line:” shows the exact source line being executed or the guard being evaluated (e.g., `while v4 > 0:`), logged EACH time it is checked.
- “state:” is a JSON-like snapshot RIGHT AFTER that line’s effect (or after a guard is evaluated), with variables from the **current scope** only.
- Scopes:
  - Before the call (outer scope): show only {{"f":"<callable_object f>"}}.
  - At the call line, switch to the callee scope and show only the callee’s locals (e.g., {{"v0": 6, "v4": 2}}).
  - On `return`, switch back to outer scope and show {{"f":"<callable_object f>", "output": <value>}}.
- While/if guards: log the guard line every time; its “state” reflects locals unchanged by the guard itself.
- Aug-assignments (like `v0 += 2`) and assignments mutate the state immediately.
- Preserve concrete values verbatim (e.g., lists, dicts, integers, strings); render function objects exactly as "<callable_object f>".
- Output NOTHING except the trace block. No code fences, no prose.

Examples:

==================================================================

<|Problem|>:
Minimum Vertex Cover (ILP): Given an undirected graph G=(V,E), choose the smallest set of vertices covering all edges. Return the cover size.
V = {{0,1,2,3}}
E = {{(0,1), (1,2), (2,3)}}

<|Response|>:
```python
# ILP: minimize sum x_v, s.t. x_u + x_v >= 1 for each edge (u,v); x_v in {{0,1}}
def f(V, E):
    import pulp
    x = {{v: pulp.LpVariable(f"x_{{v}}", lowBound=0, upBound=1, cat=pulp.LpBinary) for v in V}}
    prob = pulp.LpProblem("min_vertex_cover", pulp.LpMinimize)
    # objective
    prob += pulp.lpSum([x[v] for v in V])
    # cover constraints
    for (u, v) in E:
        prob += x[u] + x[v] >= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    val = int(round(pulp.value(pulp.lpSum([x[v] for v in V]))))
    return val

output = f({{0,1,2,3}}, {{(0,1),(1,2),(2,3)}})
print(output)
```

<|Execution Simulation|>
[BEGIN]
state: {{}}
line: def f(V, E):
state: {{"f": "<callable_object f>"}}
line: output = f({{0,1,2,3}}, {{(0,1),(1,2),(2,3)}})
state: {{"V": {{0,1,2,3}}, "E": {{(0,1),(1,2),(2,3)}}}}
line: import pulp
state: {{"V": {{0,1,2,3}}, "E": {{(0,1),(1,2),(2,3)}}, "pulp": "<module pulp>"}}
line: x = {{v: pulp.LpVariable(f"x_{{v}}", lowBound=0, upBound=1, cat=pulp.LpBinary) for v in V}}
state: {{"V": {{0,1,2,3}}, "E": {{(0,1),(1,2),(2,3)}}, "pulp": "<module pulp>", "x": {{"0": "<binvar>", "1": "<binvar>", "2": "<binvar>", "3": "<binvar>"}}}}
line: prob = pulp.LpProblem("min_vertex_cover", pulp.LpMinimize)
state: {{"V": {{0,1,2,3}}, "E": {{(0,1),(1,2),(2,3)}}, "pulp": "<module pulp>", "x": {{"0": "<binvar>", "1": "<binvar>", "2": "<binvar>", "3": "<binvar>"}}, "prob": "<LpProblem>"}}
line: prob += pulp.lpSum([x[v] for v in V])
state: {{"V": {{0,1,2,3}}, "E": {{(0,1),(1,2),(2,3)}}, "pulp": "<module pulp>", "x": {{"0": "<binvar>", "1": "<binvar>", "2": "<binvar>", "3": "<binvar>"}}, "prob": "<LpProblem>"}}
line: for (u, v) in E:
state: {{"V": {{0,1,2,3}}, "E": {{(0,1),(1,2),(2,3)}}, "pulp": "<module pulp>", "x": {{"0": "<binvar>", "1": "<binvar>", "2": "<binvar>", "3": "<binvar>"}}, "prob": "<LpProblem>", "u": 0, "v": 1}}
line: prob += x[u] + x[v] >= 1
state: {{"V": {{0,1,2,3}}, "E": {{(0,1),(1,2),(2,3)}}, "pulp": "<module pulp>", "x": {{"0": "<binvar>", "1": "<binvar>", "2": "<binvar>", "3": "<binvar>"}}, "prob": "<LpProblem>", "u": 0, "v": 1}}
line: for (u, v) in E:
state: {{"V": {{0,1,2,3}}, "E": {{(0,1),(1,2),(2,3)}}, "pulp": "<module pulp>", "x": {{"0": "<binvar>", "1": "<binvar>", "2": "<binvar>", "3": "<binvar>"}}, "prob": "<LpProblem>", "u": 1, "v": 2}}
line: prob += x[u] + x[v] >= 1
state: {{"V": {{0,1,2,3}}, "E": {{(0,1),(1,2),(2,3)}}, "pulp": "<module pulp>", "x": {{"0": "<binvar>", "1": "<binvar>", "2": "<binvar>", "3": "<binvar>"}}, "prob": "<LpProblem>", "u": 1, "v": 2}}
line: for (u, v) in E:
state: {{"V": {{0,1,2,3}}, "E": {{(0,1),(1,2),(2,3)}}, "pulp": "<module pulp>", "x": {{"0": "<binvar>", "1": "<binvar>", "2": "<binvar>", "3": "<binvar>"}}, "prob": "<LpProblem>", "u": 2, "v": 3}}
line: prob += x[u] + x[v] >= 1
state: {{"V": {{0,1,2,3}}, "E": {{(0,1),(1,2),(2,3)}}, "pulp": "<module pulp>", "x": {{"0": "<binvar>", "1": "<binvar>", "2": "<binvar>", "3": "<binvar>"}}, "prob": "<LpProblem>", "u": 2, "v": 3}}
line: prob.solve(pulp.PULP_CBC_CMD(msg=False))
state: {{"V": {{0,1,2,3}}, "E": {{(0,1),(1,2),(2,3)}}, "pulp": "<module pulp>", "x": {{"0": 0, "1": 1, "2": 1, "3": 0}}, "prob": "<LpProblem>"}}
line: val = int(round(pulp.value(pulp.lpSum([x[v] for v in V]))))
state: {{"V": {{0,1,2,3}}, "E": {{(0,1),(1,2),(2,3)}}, "pulp": "<module pulp>", "x": {{"0": 0, "1": 1, "2": 1, "3": 0}}, "prob": "<LpProblem>", "val": 2}}
line: return val
state: {{"f": "<callable_object f>", "output": 2}}
[DONE]

<|Final Answer|>
{{"rationale":"```python
# ILP: minimize sum x_v, s.t. x_u + x_v >= 1 for each edge (u,v); x_v in {{0,1}}
def f(V, E):
    import pulp
    x = {{v: pulp.LpVariable(f"x_{{v}}", lowBound=0, upBound=1, cat=pulp.LpBinary) for v in V}}
    prob = pulp.LpProblem("min_vertex_cover", pulp.LpMinimize)
    # objective
    prob += pulp.lpSum([x[v] for v in V])
    # cover constraints
    for (u, v) in E:
        prob += x[u] + x[v] >= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    val = int(round(pulp.value(pulp.lpSum([x[v] for v in V]))))
    return val
output = f({{0,1,2,3}}, {{(0,1),(1,2),(2,3)}})
print(output)
```","answer":2}}

============================================================================

<|Problem|>:
Coin Change (DP, min coins): Given coin denominations and a target amount, compute the minimum number of coins to make the amount (use -1 if impossible).
coins = [1, 2]
amount = 3

<|Response|>:
```python
# Copy code
def f(coins, amount):
    INF = 10**9
    dp = [INF] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        best = INF
        for c in coins:
            if a - c >= 0 and dp[a - c] + 1 < best:
                best = dp[a - c] + 1
        dp[a] = best
    return -1 if dp[amount] >= INF else dp[amount]

output = f([1,2], 3)
print(output)
```

<|Execution Simulation|>
[BEGIN]
state: {{}}
line: def f(coins, amount):
state: {{"f": "<callable_object f>"}}
line: output = f([1,2], 3)
state: {{"coins": [1,2], "amount": 3}}
line: INF = 10**9
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000}}
line: dp = [INF] * (amount + 1)
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [1000000000,1000000000,1000000000,1000000000]}}
line: dp[0] = 0
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1000000000,1000000000,1000000000]}}
line: for a in range(1, amount + 1):
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1000000000,1000000000,1000000000], "a": 1}}
line: best = INF
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1000000000,1000000000,1000000000], "a": 1, "best": 1000000000}}
line: for c in coins:
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1000000000,1000000000,1000000000], "a": 1, "best": 1000000000, "c": 1}}
line: if a - c >= 0 and dp[a - c] + 1 < best:
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1000000000,1000000000,1000000000], "a": 1, "best": 1000000000, "c": 1}}
line: best = dp[a - c] + 1
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1000000000,1000000000,1000000000], "a": 1, "best": 1, "c": 1}}
line: for c in coins:
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1000000000,1000000000,1000000000], "a": 1, "best": 1, "c": 2}}
line: if a - c >= 0 and dp[a - c] + 1 < best:
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1000000000,1000000000,1000000000], "a": 1, "best": 1, "c": 2}}
line: dp[a] = best
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1000000000,1000000000], "a": 1, "best": 1}}
line: for a in range(1, amount + 1):
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1000000000,1000000000], "a": 2}}
line: best = INF
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1000000000,1000000000], "a": 2, "best": 1000000000}}
line: for c in coins:
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1000000000,1000000000], "a": 2, "best": 1000000000, "c": 1}}
line: if a - c >= 0 and dp[a - c] + 1 < best:
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1000000000,1000000000], "a": 2, "best": 1000000000, "c": 1}}
line: best = dp[a - c] + 1
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1000000000,1000000000], "a": 2, "best": 2, "c": 1}}
line: for c in coins:
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1000000000,1000000000], "a": 2, "best": 2, "c": 2}}
line: if a - c >= 0 and dp[a - c] + 1 < best:
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1000000000,1000000000], "a": 2, "best": 2, "c": 2}}
line: best = dp[a - c] + 1
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1000000000,1000000000], "a": 2, "best": 1, "c": 2}}
line: dp[a] = best
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1,1000000000], "a": 2, "best": 1}}
line: for a in range(1, amount + 1):
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1,1000000000], "a": 3}}
line: best = INF
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1,1000000000], "a": 3, "best": 1000000000}}
line: for c in coins:
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1,1000000000], "a": 3, "best": 1000000000, "c": 1}}
line: if a - c >= 0 and dp[a - c] + 1 < best:
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1,1000000000], "a": 3, "best": 1000000000, "c": 1}}
line: best = dp[a - c] + 1
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1,1000000000], "a": 3, "best": 2, "c": 1}}
line: for c in coins:
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1,1000000000], "a": 3, "best": 2, "c": 2}}
line: if a - c >= 0 and dp[a - c] + 1 < best:
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1,1000000000], "a": 3, "best": 2, "c": 2}}
line: dp[a] = best
state: {{"coins": [1,2], "amount": 3, "INF": 1000000000, "dp": [0,1,1,2], "a": 3, "best": 2}}
line: return -1 if dp[amount] >= INF else dp[amount]
state: {{"f": "<callable_object f>", "output": 2}}
[DONE]

<|Final Answer|>
{{"rationale":"```python
def f(coins, amount):
    INF = 10**9
    dp = [INF] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        best = INF
        for c in coins:
            if a - c >= 0 and dp[a - c] + 1 < best:
                best = dp[a - c] + 1
        dp[a] = best
    return -1 if dp[amount] >= INF else dp[amount]
output = f([1,3,4], 6)
print(output)
```","answer":2}}

============================================================================

<|Problem|>:
Compute: 84 ÷ 6

<|Response|>:
```python
def f(a, b):
    # integer division
    return a // b

output = f(84, 6)
print(output)
```

<|Execution Simulation|>
[BEGIN]
state: {{}}
line: def f(a, b):
state: {{"f": "<callable_object f>"}}
line: output = f(84, 6)
state: {{"a": 84, "b": 6}}
line: return a // b
state: {{"f": "<callable_object f>", "output": 14}}
[DONE]

<|Final Answer|>
{{"rationale":"```python
def f(a, b):
    # integer division
    return a // b
output = f(84, 6)
print(output)
```","answer":14}}

============================================================================

<|Problem|>:
{problem}

<|Response|>:

"""
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
    
class VLLMClient(LLMClient):
    """
    vLLM-powered local inference with the same .chat(...) signature you use
    everywhere else. Reuses a single engine; applies a chat template if the
    model provides one; otherwise falls back to the last user message content.
    """
    def __init__(self, model_name: str,
                 dtype: str = "auto",
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: Optional[int] = None,
                 download_dir: Optional[str] = None,
                 trust_remote_code: bool = False, 
                 seed: int=0):
        assert dtype == "float16", "Wrong dtype"
        if VLLMEngine is None:
            raise RuntimeError(
                "vLLM is not installed. Install a CUDA-matching vLLM wheel "
                "(e.g. vllm-cu121) or build from source."
            )
        # vLLM engine (persistent)
        self.seed = seed
        self.llm = VLLMEngine(
            model=model_name,
            dtype=dtype,                               # "auto" | "float16"
            tensor_parallel_size=int(tensor_parallel_size),
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len) if max_model_len else None,
            trust_remote_code=bool(trust_remote_code),
            download_dir=download_dir,
            seed=seed, 
        )
        # Use HF tokenizer to format chat prompts if available
        self.tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.has_template = hasattr(self.tok, "apply_chat_template") and (self.tok.chat_template is not None)

    def _to_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.has_template:
            # Mirrors your HFLocalClient behavior
            return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # Fallback: use the last user content (same as HFLocalClient fallback)
        return messages[-1]["content"]

    def chat(self, model: str, messages: List[Dict[str, str]], max_tokens: int,
             temperature: float, top_p: float, stop: Optional[List[str]] = None) -> str:
        prompt = self._to_prompt(messages)
        sp = SamplingParams(
            max_tokens=int(max_tokens),                # new tokens
            temperature=float(temperature) if temperature is not None else 0.0,
            top_p=float(top_p) if top_p is not None else 1.0,
            stop=stop or None,
            seed=self.seed
        )
        # vLLM can batch; here we keep semantics identical (one request per call)
        outs = self.llm.generate([prompt], sp)
        # outs is a List[RequestOutput]; take first, first candidate
        return outs[0].outputs[0].text
    def chat_many(self, model: str, messages_list: List[List[Dict[str, str]]], max_tokens: int,
                  temperature: float, top_p: float, stop: Optional[List[str]] = None) -> List[str]:
        prompts = [self._to_prompt(msgs) for msgs in messages_list]
        sp = SamplingParams(
            max_tokens=int(max_tokens),
            temperature=float(temperature) if temperature is not None else 0.0,
            top_p=float(top_p) if top_p is not None else 1.0,
            stop=stop or None,
            seed=self.seed,
        )
        outs = self.llm.generate(prompts, sp)
        # preserve order, one candidate per request
        return [o.outputs[0].text for o in outs]

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

JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")

def _repair_json_candidate(s: str) -> Optional[str]:
    m = JSON_OBJ_RE.search(s)
    if not m:
        return None
    frag = m.group(0)

    # If rationale is a quoted string containing raw newlines, escape them.
    # This targets the first "rationale": " ... " occurrence.
    def _escape_newlines_in_rationale(mo: re.Match) -> str:
        head, body, tail = mo.group(1), mo.group(2), mo.group(3)
        body = body.replace('\\', '\\\\').replace('"', '\\"').replace('\r', '\\r').replace('\n', '\\n')
        return head + body + tail

    frag = re.sub(
        r'("rationale"\s*:\s*")([\s\S]*?)(")',
        _escape_newlines_in_rationale,
        frag,
        count=1,
    )
    return frag

def extract_json(s: str) -> Optional[Dict[str, Any]]:
    # Try clean parse
    try:
        start = s.index('{'); end = s.rindex('}')
        frag = s[start:end+1]
        return json.loads(frag)
    except Exception:
        pass
    # Try repaired parse
    try:
        repaired = _repair_json_candidate(s)
        if repaired is not None:
            return json.loads(repaired)
    except Exception:
        return None
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
        # salvage from fenced code
        code = extract_fenced_code(raw)
        if code:
            # Try to recover the printed int from the raw text
            nums = INT_RE.findall(raw)
            ans = None
            try:
                ans = int(nums[-1])
            except Exception:
                if nums and float(nums[-1]) != float('inf'):
                    ans = int(float(nums[-1]))
            if ans is not None:
                obj = {"rationale": f"```python\n{code}\n```", "answer": ans}
                return Parsed(json.dumps(obj), True, ans, obj["rationale"], "salvaged")
        # fallback: last int in whole text
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
            m = INT_RE.findall(ans); 
            try:
                ans = int(m[-1])
            except Exception:
                if m and float(m[-1]) != float('inf'):
                    ans = int(float(m[-1]))
                else:
                    None
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
                    ok=True
                except Exception:
                    try:
                        value = int(float(nums[-1]))
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
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
    elif args.backend == 'vllm':
        print("Instantiating VLLM")
        client = VLLMClient(
            model_name=args.model,
            dtype=args.vllm_dtype,
            tensor_parallel_size=args.vllm_tensor_parallel,
            gpu_memory_utilization=args.vllm_gpu_mem_util,
            max_model_len=args.vllm_max_model_len,
            download_dir=args.vllm_download_dir,
            trust_remote_code=args.hf_trust_remote_code,  
            seed=args.seed
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
        
    nl_msgs = [[{"role": "user", "content": NL_PROMPT.format(problem=pb.text())}] for pb in problems]
    code_msgs = [[{"role": "user", "content": CODE_PROMPT.format(problem=pb.text())}] for pb in problems]

    def run_batch(messages_list):
        if hasattr(client, "chat_many") and callable(getattr(client, "chat_many")) and args.batch_size > 1:
            outs = []
            for start in range(0, len(messages_list), args.batch_size):
                chunk = messages_list[start:start+args.batch_size]
                outs.extend(client.chat_many(args.model, chunk,
                                             max_tokens=args.max_tokens,
                                             temperature=args.temperature, top_p=args.top_p, stop=None))
            return outs
        else:
            return [
                client.chat(args.model, m, max_tokens=args.max_tokens, temperature=0.0, top_p=1.0, stop=None)
                for m in messages_list
            ]

    # === Generate all NL outputs, then all Code outputs (order preserved) ===
    nl_raw_all = run_batch(nl_msgs)
    code_raw_all = run_batch(code_msgs)
    
    records: List[Record] = []
    for i, pb in enumerate(tqdm(problems, total=len(problems), desc='eval')):
        problem_text = pb.text()
        truth = pb.ground_truth()

        nl_raw = nl_raw_all[i]
        nl_parsed = parse_response(nl_raw)
        ans_nl = nl_parsed.answer
        correct_nl = int(ans_nl == truth)

        base_nl = f"{args.model}/nl/d{pb.digits}/{pb.kind}/i{i}"
        tb_text(f"{base_nl}/prompt", "Prompt (NL-CoT)", NL_PROMPT.format(problem=problem_text))
        tb_text(f"{base_nl}/rationale", "NL Rationale", nl_parsed.rationale or "")
        tb_text(f"{base_nl}/raw_json", "Raw NL JSON", nl_parsed.raw)
        tb_text(f"{base_nl}/answer", "Final Answer (NL)", "" if ans_nl is None else str(ans_nl))

        code_raw = code_raw_all[i]
        code_parsed = parse_response(code_raw)
        ans_code = code_parsed.answer
        correct_code = int(ans_code == truth)

        base_code = f"{args.model}/code/d{pb.digits}/{pb.kind}/i{i}"
        tb_text(f"{base_code}/prompt", "Prompt (Code-CoT)", CODE_PROMPT.format(problem=problem_text))
        tb_text(f"{base_code}/rationale", "Code Rationale (fenced)", code_parsed.rationale or "")
        tb_text(f"{base_code}/raw_json", "Raw Code JSON", code_parsed.raw)
        tb_text(f"{base_code}/answer", "Final Answer (Code)", "" if ans_code is None else str(ans_code))

        ans_code_exec = {"ok": False, "value": None, "stdout": "", "stderr": "", "retcode": None, "timeout": False}
        if args.exec_code:
            ans_code_exec = exec_from_rationale(code_parsed.rationale, allow_imports=True)
            tb_text(f"{base_code}/exec_answer", "Executed Answer (subprocess)", "" if ans_code_exec is None else str(ans_code_exec))
            tb_text(f"{base_code}/exec_stdout", "Executed STDOUT", ans_code_exec.get("stdout",""))
            tb_text(f"{base_code}/exec_stderr", "Executed STDERR", ans_code_exec.get("stderr",""))
            tb_text(f"{base_code}/exec_meta", "Exec Meta",
                    f"retcode={ans_code_exec.get('retcode')} timeout={ans_code_exec.get('timeout')} ok={ans_code_exec.get('ok')} value={ans_code_exec.get('value')}")
        correct_code_exec = int(ans_code_exec.get("value") == truth) if ans_code_exec is not None else 0

        records.append(Record(
            idx=i, problem=problem_text, kind=pb.kind, digits=pb.digits, truth=truth,
            answer_nl=ans_nl, correct_nl=correct_nl,
            answer_code=ans_code, correct_code=correct_code,
            answer_code_exec=ans_code_exec, correct_code_exec=correct_code_exec,
            raw_nl=nl_raw, raw_code=code_raw,
        ))


    # CSV
    import csv
    csv_path = os.path.join(args.outdir, f'{exp_id}_results_seed_{args.seed}.csv')
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
    if has_exec:
        lines.append(f"Execution condition (subprocess): acc={acc_exec:.4f}")
    lines.append(f"Discordant pairs b=code>nl: {b}, c=nl>code: {c}, McNemar exact p={p_mc:.4g}")
    lines.append("Per-kind bins:")
    # --- Per-kind × digit breakdown (printed) ---
    by_kd: Dict[Tuple[str, int], List[Record]] = {}
    for r in records:
        by_kd.setdefault((r.kind, r.digits), []).append(r)

    def _acc(lst): 
        return sum(lst) / max(1, len(lst))

    lines.append("")
    lines.append("Per-kind × digit bins:")
    for kind in sorted({k for (k, _) in by_kd.keys()}):
        lines.append(f"  kind={kind}")
        for d in sorted({d for (k, d) in by_kd.keys() if k == kind}):
            grp = by_kd[(kind, d)]
            N = len(grp)
            acc_nl   = _acc([x.correct_nl   for x in grp])
            acc_code = _acc([x.correct_code for x in grp])

            # Exec accuracy only for items that actually executed (and if exec requested)
            if args.exec_code:
                exec_vals = [x.correct_code_exec for x in grp if x.answer_code_exec is not None]
                acc_exec = (sum(exec_vals) / len(exec_vals)) if exec_vals else float('nan')
            else:
                acc_exec = float('nan')

            lines.append(f"    digits={d:2d}: N={N:3d}  NL={acc_nl:.4f}  Code={acc_code:.4f}  Exec={acc_exec:.4f}")

            # Optional: log to TensorBoard as kind/digit tags
            if tb is not None:
                tb.add_scalar(f"{args.model}/acc_nl/{kind}/d{d}",   acc_nl)
                tb.add_scalar(f"{args.model}/acc_code/{kind}/d{d}", acc_code)
                if args.exec_code and not math.isnan(acc_exec):
                    tb.add_scalar(f"{args.model}/acc_exec/{kind}/d{d}", acc_exec)
    csv_kd_path = os.path.join(args.outdir, f'{exp_id}_by_kind_digit.csv')
    with open(csv_kd_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["kind", "digits", "N", "acc_nl", "acc_code", "acc_exec"])
        for kind in sorted({k for (k, _) in by_kd.keys()}):
            for d in sorted({d for (k, d) in by_kd.keys() if k == kind}):
                grp = by_kd[(kind, d)]
                N = len(grp)
                acc_nl   = sum(x.correct_nl   for x in grp) / N if N else float('nan')
                acc_code = sum(x.correct_code for x in grp) / N if N else float('nan')
                if args.exec_code:
                    exec_vals = [x.correct_code_exec for x in grp if x.answer_code_exec is not None]
                    acc_exec = (sum(exec_vals) / len(exec_vals)) if exec_vals else float('nan')
                else:
                    acc_exec = float('nan')
                w.writerow([
                    kind, d, N, f"{acc_nl:.6f}", f"{acc_code:.6f}",
                    "" if math.isnan(acc_exec) else f"{acc_exec:.6f}"
                ])

    

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

    p.add_argument('--backend', type=str, default='dummy', choices=['dummy','openai','hf', 'vllm'])
    p.add_argument('--model', type=str, default='gpt-4o', help='OpenAI model name or HF repo/path when --backend=hf')
    p.add_argument('--hf_dtype', type=str, default='bfloat16', choices=['auto','float16','bfloat16','float32'])
    p.add_argument('--hf_device_map', type=str, default='auto')
    p.add_argument('--hf_trust_remote_code', action='store_true')

    p.add_argument('--max_tokens', type=int, default=4192)
    p.add_argument('--temperature', type=int, default=0.1)
    p.add_argument('--top_p', type=int, default=0.90)

    p.add_argument('--exec_code', action='store_true', help='execute code-CoT in sandboxed subprocess (imports allowed)')
    p.add_argument('--outdir', type=str, default='out')
    p.add_argument('--log_every', type=int, default=50)

    # TensorBoard text limits
    p.add_argument('--tb_text_chars', type=int, default=10000)
    p.add_argument('--tb_disable', action='store_true')
    
    # vLLM options (kept minimal; defaults are conservative)
    p.add_argument('--batch_size', type=int, default=8, help='Batch size for backends that support chat_many (vLLM).')
    p.add_argument('--vllm_dtype', type=str, default='float16',
                   choices=['auto','float16','bfloat16'])
    p.add_argument('--vllm_tensor_parallel', type=int, default=8)
    p.add_argument('--vllm_gpu_mem_util', type=float, default=0.80)
    p.add_argument('--vllm_max_model_len', type=int, default=None)
    p.add_argument('--vllm_download_dir', type=str, default='../models')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
