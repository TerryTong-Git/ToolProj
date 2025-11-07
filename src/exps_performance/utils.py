import random
import re
from typing import Optional

INT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\s*\n([\s\S]*?)\n```", re.MULTILINE)


def rand_string(rng: random.Random, alpha="abcd", n: Optional[int] = None, lo=5, hi=12) -> str:
    if n is None:
        n = rng.randint(lo, hi)
    return "".join(rng.choice(alpha) for _ in range(n))


def extract_fenced_code(rationale: Optional[str]) -> Optional[str]:
    if not rationale:
        return None
    m = FENCE_RE.search(rationale)
    if not m:
        return None
    return m.group(1).strip()


def sample_int(digits: int, rng: random.Random) -> int:
    if digits <= 0:
        raise ValueError("digits must be >=1")
    lo = 10 ** (digits - 1)
    hi = 10**digits - 1
    return rng.randint(lo, hi)
