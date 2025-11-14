from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from src.exps_performance.utils import INT_RE, extract_fenced_code


# try just execute with exec
class CodeReasoning(BaseModel):
    code: str = Field(description="The code that solves the problem, printing out the solution")
    simulation: str = Field(description="The attempt at simulating the code")
    answer: str = Field(description="The answer the code outputs")


class NLReasoning(BaseModel):
    reasoning: str = Field(description="Step by step reasoning chain used to solve the problem")
    answer: str = Field(description="The answer arrived at after reasoning")


class JustCode(BaseModel):
    answer: str = Field(description="The answer the code outputs")


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
        body = body.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "\\r").replace("\n", "\\n")
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
        start = s.index("{")
        end = s.rindex("}")
        frag = s[start : end + 1]
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
    answer: Optional[int | float]
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
            ans: int | float = -float("inf")
            try:
                ans = int(nums[-1])
            except Exception:
                if nums and float(nums[-1]) != float("inf"):
                    ans = int(float(nums[-1]))
            if ans is not -float("inf"):
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
    if not (isinstance(obj, dict) and "answer" in obj and "rationale" in obj):
        return Parsed(raw, False, None, None, "bad-json-keys")
    ans = obj["answer"]
    try:
        ans = int(ans)
    except Exception:
        if isinstance(ans, str):
            m = INT_RE.findall(ans)
            try:
                ans = int(m[-1])
            except Exception:
                if m and float(m[-1]) != float("inf"):
                    ans = int(float(m[-1]))
                else:
                    None
        else:
            ans = -float("inf")

    return Parsed(
        raw,
        ans is not None,
        ans,
        obj.get("rationale"),
        None if ans is not None else "answer-not-int",
    )
