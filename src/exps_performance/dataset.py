from __future__ import annotations

import random
from typing import List, MutableSequence, Optional, Sequence

import torch
from problems import Problem, load_CLRS30, load_gsm8k, load_NPHARDEVAL
from utils import rand_string

try:
    from vllm import LLM as VLLMEngine
    from vllm import SamplingParams
except Exception as _vllm_import_err:
    VLLMEngine = None
    SamplingParams = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.deterministic = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def sample_int(digits: int, rng: random.Random) -> int:
    if digits <= 0:
        raise ValueError("digits must be >=1")
    lo = 10 ** (digits - 1)
    hi = 10**digits - 1
    return rng.randint(lo, hi)


def make_problem(rng: random.Random, kind: str, digits: Optional[int] = None) -> Problem:
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
    d = digits if digits is not None else rng.choice([2, 4, 8])

    if kind in ("add", "sub", "mul", "mix"):
        a = sample_int(d, rng)
        b = sample_int(d, rng)
        if kind == "sub" and b > a:
            a, b = b, a
        return Problem(kind=kind, digits=d, a=a, b=b)

    if kind == "lcs":
        n = max(2, int(d))
        s = rand_string(rng, alpha="abcd", n=n)
        t = rand_string(rng, alpha="abcd", n=n + rng.randint(-1, 1))
        return Problem(kind="lcs", digits=d, data={"s": s, "t": t})

    if kind == "knap":
        n_items = max(3, int(d))
        # scale magnitudes gently with d to keep runtimes sane
        w_max = max(5, 2 * d)
        v_max = max(10, 4 * d)
        weights = [rng.randint(1, w_max) for _ in range(n_items)]
        values = [rng.randint(1, v_max) for _ in range(n_items)]
        capacity1 = max(1, int(0.5 * sum(weights)))
        return Problem(kind="knap", digits=d, data={"weights": weights, "values": values, "capacity": capacity1})

    if kind == "rod":
        N = max(2, int(d))
        price_max = max(5, 3 * d)
        prices = [rng.randint(1, price_max) for _ in range(N)]
        return Problem(kind="rod", digits=d, data={"prices": prices})

    if kind == "ilp_assign":
        n = max(2, min(int(d), 7))  # cap n for brute-force fallback safety
        C = [[rng.randint(1, max(6, 3 * d)) for _ in range(n)] for __ in range(n)]
        return Problem(kind="ilp_assign", digits=d, data={"cost": C})

    if kind == "ilp_prod":
        # scale #products/#resources and magnitudes with d, but cap to keep fallback feasible
        P = max(2, min(2 + d // 3, 6))
        R = max(2, min(2 + d // 4, 4))
        profit = [rng.randint(3, max(8, 3 * d)) for _ in range(P)]
        consumption = [[rng.randint(1, max(3, d)) for _ in range(P)] for __ in range(R)]
        # capacity scaled so some slack exists; upper bounds smallish (<= 10)
        capacity = [rng.randint(max(6, 2 * d), max(10, 4 * d)) for _ in range(R)]
        upper = []
        for j in range(P):
            ub_j = min(10, min((capacity[i] // max(1, consumption[i][j]) for i in range(R)), default=10))
            upper.append(int(max(3, ub_j)))
        return Problem(
            kind="ilp_prod",
            digits=d,
            data={
                "profit": profit,
                "consumption": consumption,
                "capacity": capacity,
                "upper_bound": upper,
            },
        )

    # perhap save this sometwhere, make bins more fine-grained. Submit PR and review it before merging.
    if kind == "ilp_partition":
        n_items = max(4, min(int(d), 24))
        w_max = max(6, 3 * d)
        weights = [rng.randint(1, w_max) for _ in range(n_items)]
        return Problem(kind="ilp_partition", digits=d, data={"weights": weights})

    raise ValueError(f"unknown kind: {kind}")


def make_dataset(n: int, digits_list: List[int], kinds: List[str], seed: int = 1) -> Sequence[Problem]:
    """
    Balance over (kind × digits) so MI/acc buckets are well-populated.
    """
    if kinds[0] == "gsm8k":
        return load_gsm8k()
    if kinds[0] == "nphardeval":
        return load_NPHARDEVAL()
    if kinds[0] == "clrs30":
        return load_CLRS30()
    rng = random.Random(seed)
    problems: MutableSequence[Problem] = []
    K = max(1, len(kinds))
    D = max(1, len(digits_list))
    per = max(1, n // (K * D))
    for d in digits_list:
        for k in kinds:
            for _ in range(per):
                problems.append(make_problem(rng, k, d))
    while len(problems) < n:
        k = rng.choice(kinds)
        d = rng.choice(digits_list)
        problems.append(make_problem(rng, k, d))
    rng.shuffle(problems)
    return problems[:n]
