import random
import re
from typing import Any, Dict, Optional, Set, Tuple

INT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\s*\n([\s\S]*?)\n```", re.MULTILINE)


def remove_python_triple_quote(input: str) -> str:
    """Not accepted by langchain parsing, so remove"""
    return input.replace('"""', "")


def cast_float_to_int(obj: Any) -> Any:
    if isinstance(obj, float):
        if obj == float("inf") or obj == float("-inf"):
            return -9999999999
        return int(obj)
    if isinstance(obj, list):
        return [cast_float_to_int(o) for o in obj]
    if isinstance(obj, dict):
        return {cast_float_to_int(k): cast_float_to_int(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(cast_float_to_int(o) for o in obj)
    return obj


def clean_code_llm(code: str) -> str:
    pat = r"\`\`\`python(.*)\`\`\`"
    match = re.search(pat, code, flags=re.DOTALL)
    if match:
        code = str(match.group(1))
    code = code.replace("```", "")
    code = code.replace("python", "")
    code = code.replace('"""', "")
    return code


def remove_json_backticks(code: str) -> str:
    pat = r"\`\`\`json(.*)\`\`\`"
    match = re.search(pat, code, flags=re.DOTALL)
    if match:
        return match.group(1)
    else:
        code = code.replace("```", "")
        code = code.replace("json", "")
        return code


def rand_string(rng: random.Random, alpha: str = "abcd", n: Optional[int] = None, lo: int = 5, hi: int = 12) -> str:
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


def seed_all_and_setup(args: Any) -> None:
    random.seed(args.seed)
    import numpy as np
    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def read_dimacs_format(dimacs_str: str) -> Tuple[int, Dict[int, Set[int]]]:
    lines = dimacs_str.strip().split("\n")
    p_line = next(line for line in lines if line.startswith("p"))
    _, _, num_vertices_str, num_edges_str = p_line.split()
    num_vertices, _ = int(num_vertices_str), int(num_edges_str)

    adjacency_list: Dict[int, Set[int]] = {i: set() for i in range(1, num_vertices + 1)}
    for line in lines:
        if line.startswith("e"):
            _, vertex1_str, vertex2_str = line.split()
            vertex1, vertex2 = int(vertex1_str), int(vertex2_str)
            if vertex1 in adjacency_list and vertex2 in adjacency_list:
                adjacency_list[vertex1].add(vertex2)
                adjacency_list[vertex2].add(vertex1)

    return num_vertices, adjacency_list
