from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any, Dict, List, Optional

import torch
from utils import INT_RE, extract_fenced_code

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
# ----------------------- Code execution (subprocess sandbox) ----------------


def run_code_subprocess(
    code: str,
    timeout_s: float = 3.0,
    allow_imports: bool = True,
    exec_prefix: Optional[List[str]] = None,
    exec_python: Optional[str] = None,
) -> Dict[str, Any]:
    """Run code and always return a dict with stable keys."""
    import time as _time

    code = textwrap.dedent(code).strip()
    t0 = _time.time()
    stdout = ""
    stderr = ""
    value = None
    ok = False
    timeout = False
    retcode = None
    exc = None

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
                cmd,
                cwd=td,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                check=False,
                text=True,
                env=env,
                preexec_fn=preexec,
            )
            stdout = res.stdout or ""
            stderr = res.stderr or ""
            retcode = res.returncode
            # extract last integer printed
            nums = INT_RE.findall(stdout)
            if nums:
                try:
                    value = int(nums[-1])
                    ok = True
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


def exec_from_rationale(
    rationale: Optional[str],
    allow_imports: bool = True,
    exec_prefix: Optional[List[str]] = None,
    exec_python: Optional[str] = None,
) -> Dict[str, Any]:
    code = extract_fenced_code(rationale)
    if not code:
        return {
            "ok": False,
            "value": None,
            "stdout": "",
            "stderr": "no_fenced_code",
            "retcode": None,
            "timeout": False,
            "duration_s": 0.0,
        }
    return run_code_subprocess(
        code,
        timeout_s=3.0,
        allow_imports=allow_imports,
        exec_prefix=exec_prefix,
        exec_python=exec_python,
    )
