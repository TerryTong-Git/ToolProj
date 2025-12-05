#!/usr/bin/env python3
"""Parsers for extracting structured information from problem texts."""

import re
from typing import List, Optional, Tuple

# ------------------------------------------------------------------------------
# Regex patterns
# ------------------------------------------------------------------------------

INT_RE = re.compile(r"[-+]?\d+")

# Primary wrapper markers used in NL_PROMPT / CODE_PROMPT
PROBLEM_SLICE_RE = re.compile(
    r"Here\s+is\s+the\s+actual\s+problem:\s*(.*?)\s*Give\s+the\s+solution:",
    re.DOTALL | re.IGNORECASE,
)

# Code-fence stripper
FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*\n([\s\S]*?)\n```", re.MULTILINE)

# Known per-kind headers from Problem.text()
KNOWN_KIND_HEADERS = [
    r"^Compute:\s*\(",  # mix
    r"^Compute:\s*\d+\s*[\+\-\*]\s*\d+",  # add/sub/mul
    r"^Compute the length of the Longest Common Subsequence",
    r"^0/1 Knapsack:",
    r"^Rod cutting:",
    r"^Assignment problem:",
    r"^Production planning:",
    r"^Partition:",
]
KIND_START_RE = re.compile("|".join(KNOWN_KIND_HEADERS), re.MULTILINE)

# Arithmetic patterns
AR_ADD = re.compile(r"^\s*Compute:\s*(\d+)\s*\+\s*(\d+)\s*$", re.MULTILINE | re.IGNORECASE)
AR_SUB = re.compile(r"^\s*Compute:\s*(\d+)\s*-\s*(\d+)\s*$", re.MULTILINE | re.IGNORECASE)
AR_MUL = re.compile(r"^\s*Compute:\s*(\d+)\s*\*\s*(\d+)\s*$", re.MULTILINE | re.IGNORECASE)
AR_MIX = re.compile(r"^\s*Compute:\s*\(\s*(\d+)\s*\+\s*(\d+)\s*\)\s*\*\s*(\d+)\s*$", re.MULTILINE | re.IGNORECASE)


# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------


def strip_md_code_fences(s: str) -> str:
    """Remove markdown code fences, keeping inner text."""
    if not s:
        return ""
    return FENCE_RE.sub(lambda m: m.group(1), s)


def maybe_strip_fences(text: str) -> str:
    """Strip code fence markers from text."""
    return (text or "").replace("```python", " ").replace("```", " ")


def parse_list_of_ints(s: str) -> List[int]:
    """Parse a string to extract a list of integers."""
    return [int(x) for x in INT_RE.findall(s or "")]


def parse_list_of_list_ints(s: str) -> List[List[int]]:
    """Parse a string to extract a list of lists of integers."""
    rows = []
    for row in re.findall(r"\[([^\[\]]*)\]", s or ""):
        vals = [int(x) for x in INT_RE.findall(row)]
        if vals:
            rows.append(vals)
    return rows


def safe_len(x) -> int:
    """Safe length calculation."""
    try:
        return len(x)
    except Exception:
        return 0


# ------------------------------------------------------------------------------
# Problem text extraction
# ------------------------------------------------------------------------------


def extract_problem_text(full_prompt: str) -> str:
    """
    Extract ONLY the {problem} block from Problem.text().

    Works with:
      - NL/CODE templates (between 'Here is the actual problem:' and 'Give the solution:')
      - Raw {problem} text (no wrapper)
      - Prompts wrapped in code fences
      - Extra text before/after (falls back to scanning for known headers)
    """
    s = full_prompt or ""
    s = strip_md_code_fences(s).strip()

    # Ideal path: inside the NL/CODE wrapper
    m = PROBLEM_SLICE_RE.search(s)
    if m:
        return m.group(1).strip()

    # Fallback: find first line that looks like a Problem.text() header
    m2 = KIND_START_RE.search(s)
    if m2:
        start = m2.start()
        tail = s[start:].strip()
        parts = re.split(r"\n\s*\n", tail, maxsplit=1)
        return parts[0].strip()

    # Last resort: if the prompt is already just the problem, return it
    return s


# ------------------------------------------------------------------------------
# Kind-specific parsers
# ------------------------------------------------------------------------------


def parse_arithmetic_operands(kind: str, text: str, d: int) -> Optional[Tuple[int, int]]:
    """Parse arithmetic operands from problem text."""
    if kind == "add":
        m = AR_ADD.search(text)
        return (int(m.group(1)), int(m.group(2))) if m else None
    if kind == "sub":
        m = AR_SUB.search(text)
        return (int(m.group(1)), int(m.group(2))) if m else None
    if kind == "mul":
        m = AR_MUL.search(text)
        return (int(m.group(1)), int(m.group(2))) if m else None
    if kind == "mix":
        m = AR_MIX.search(text)
        if m:
            a, b, _a2 = map(int, m.groups())
            return a, b
    return None


def parse_lcs_lengths(text: str, d: int) -> Tuple[int, int]:
    """Parse LCS string lengths from problem text."""
    Sm = re.search(r'S\s*=\s*"([^"]*)"', text)
    Tm = re.search(r'T\s*=\s*"([^"]*)"', text)
    Ls = len(Sm.group(1)) if Sm else max(2, d)
    Lt = len(Tm.group(1)) if Tm else max(2, d)
    return Ls, Lt


def parse_knap_stats(text: str, d: int) -> Tuple[int, float]:
    """Parse knapsack statistics from problem text."""
    Wm = re.search(r"W\s*=\s*\[([^\]]*)\]", text)
    Vm = re.search(r"V\s*=\s*\[([^\]]*)\]", text)
    Cm = re.search(r"C\s*=\s*([0-9]+)", text)

    if Wm and Vm:
        W = parse_list_of_ints(Wm.group(1))
        n_items = len(W)
        C = int(Cm.group(1)) if Cm else max(1, int(0.5 * sum(W)))
        cap_ratio = C / max(1, sum(W))
    else:
        n_items = max(3, d)
        cap_ratio = 0.5
    return n_items, cap_ratio


def parse_rod_N(text: str, d: int) -> int:
    """Parse rod length from problem text."""
    Nm = re.search(r"\bN\s*=\s*([0-9]+)", text)
    if Nm:
        return int(Nm.group(1))
    Pm = re.search(r"P\s*=\s*\[([^\]]*)\]", text)
    if Pm:
        return len(parse_list_of_ints(Pm.group(1)))
    return max(2, d)


def parse_ilp_assign_n(text: str, d: int) -> int:
    """Parse ILP assignment matrix size from problem text."""
    Cm = re.search(r"C\s*=\s*(\[[\s\S]*\])\s*$", text, re.MULTILINE)
    if Cm:
        mat = parse_list_of_list_ints(Cm.group(1))
        if safe_len(mat) > 0:
            return len(mat)
    return min(max(2, d), 7)


def parse_ilp_prod_PR(text: str, d: int) -> Tuple[int, int]:
    """Parse ILP production problem dimensions from problem text."""
    Pm = re.search(r"profit\s*=\s*\[([^\]]*)\]", text)
    Cm = re.search(r"consumption\s*\(rows=resources\)\s*=\s*(\[[\s\S]*\])", text)

    if Pm:
        P = len(parse_list_of_ints(Pm.group(1)))
    else:
        P = min(2 + d // 3, 6)

    if Cm:
        cons = parse_list_of_list_ints(Cm.group(1))
        R = len(cons) if safe_len(cons) > 0 else min(2 + d // 4, 4)
    else:
        R = min(2 + d // 4, 4)

    return int(P), int(R)


def parse_ilp_partition_n(text: str, d: int) -> int:
    """Parse ILP partition problem size from problem text."""
    Wm = re.search(r"weights\s*=\s*\[([^\]]*)\]", text)
    if Wm:
        return len(parse_list_of_ints(Wm.group(1)))
    return min(max(4, d), 24)
