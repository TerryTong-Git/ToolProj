from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from src.exps_performance.rich_ui import progress_manager

load_dotenv()


@dataclass
class RLMExecutionResult:
    response: str
    err: str
    execution_time: float = 0.0
    metadata: Optional[dict[str, Any]] = None


class RecursiveLMExecutor:
    """
    Thin wrapper around the external `rlm` package so the benchmark can treat
    RLM execution as another answer-producing stage.

    The import is intentionally lazy so the rest of the project does not depend
    on the package unless an RLM arm is explicitly enabled.
    """

    def __init__(self, args: Any):
        self.args = args
        self._rlm: Any | None = None

    def _ensure_repo_path(self) -> None:
        repo_path = getattr(self.args, "rlm_repo_path", None)
        if not repo_path:
            return
        resolved = str(Path(repo_path).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)

    def _load_rlm_class(self) -> Any:
        self._ensure_repo_path()
        try:
            from rlm import RLM
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Could not import `rlm`. Install the `rlms` package or point "
                "--rlm_repo_path at a checkout of https://github.com/alexzhang13/rlm."
            ) from exc
        return RLM

    def _load_rlm_logger_class(self) -> Any:
        self._ensure_repo_path()
        try:
            from rlm.logger import RLMLogger
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Could not import `RLMLogger` from the external rlm package."
            ) from exc
        return RLMLogger

    def _resolve_backend(self) -> str:
        backend = getattr(self.args, "rlm_backend", None) or getattr(self.args, "backend", "openrouter")
        if backend == "running":
            return "openai"
        if backend == "dummy":
            raise RuntimeError("RLM execution is not compatible with backend=dummy.")
        return backend

    def _build_backend_kwargs(self) -> dict[str, Any]:
        model_name = getattr(self.args, "rlm_model", None) or getattr(self.args, "model", None)
        if not model_name:
            raise RuntimeError("RLM execution requires a model name.")

        kwargs: dict[str, Any] = {"model_name": model_name}
        backend = self._resolve_backend()

        explicit_api_key = getattr(self.args, "rlm_api_key", None)
        if explicit_api_key:
            kwargs["api_key"] = explicit_api_key

        explicit_base_url = getattr(self.args, "rlm_base_url", None)
        if explicit_base_url:
            kwargs["base_url"] = explicit_base_url
            return kwargs

        if backend == "openrouter":
            kwargs["base_url"] = getattr(self.args, "openrouter_base_url", None) or os.getenv(
                "OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"
            )
            api_key = getattr(self.args, "openrouter_api_key", None) or os.getenv("OPENROUTER_API_KEY")
            if api_key and "api_key" not in kwargs:
                kwargs["api_key"] = api_key
        elif backend == "vllm":
            vllm_base = os.getenv("OPENAI_API_BASE")
            if vllm_base:
                kwargs["base_url"] = vllm_base
        elif backend == "openai":
            openai_base = os.getenv("OPENAI_API_BASE")
            if openai_base:
                kwargs["base_url"] = openai_base

        return kwargs

    def _get_rlm(self) -> Any:
        if self._rlm is not None:
            return self._rlm

        RLM = self._load_rlm_class()
        RLMLogger = self._load_rlm_logger_class()
        self._rlm = RLM(
            backend=self._resolve_backend(),
            backend_kwargs=self._build_backend_kwargs(),
            environment=getattr(self.args, "rlm_environment", "local"),
            max_depth=max(1, int(getattr(self.args, "rlm_max_depth", 2))),
            max_iterations=max(1, int(getattr(self.args, "rlm_max_iterations", 8))),
            max_timeout=getattr(self.args, "rlm_max_timeout", None),
            logger=RLMLogger(),
            verbose=bool(getattr(self.args, "rlm_verbose", False)),
        )
        return self._rlm

    def _serialize_args(self) -> dict[str, Any]:
        raw_args = dict(vars(self.args))
        serialized: dict[str, Any] = {}
        for key, value in raw_args.items():
            if isinstance(value, Path):
                serialized[key] = str(value)
            else:
                serialized[key] = value
        return serialized

    def run(self, prompt: str) -> RLMExecutionResult:
        try:
            completion = self._get_rlm().completion(prompt)
            return RLMExecutionResult(
                response=str(getattr(completion, "response", "") or ""),
                err="ok",
                execution_time=float(getattr(completion, "execution_time", 0.0) or 0.0),
                metadata=getattr(completion, "metadata", None),
            )
        except Exception as exc:  # noqa: BLE001
            return RLMExecutionResult(response="", err=str(exc))

    async def arun(self, prompt: str) -> RLMExecutionResult:
        payload = {"args": self._serialize_args(), "prompt": prompt}
        input_path = ""
        output_path = ""
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as input_file:
                json.dump(payload, input_file)
                input_path = input_file.name
            output_path = f"{input_path}.out.json"
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "src.exps_performance.core.rlm_worker",
                input_path,
                output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if output_path and Path(output_path).exists():
                try:
                    result_payload = json.loads(Path(output_path).read_text())
                    return RLMExecutionResult(
                        response=str(result_payload.get("response", "") or ""),
                        err=str(result_payload.get("err", "ok") or "ok"),
                        execution_time=float(result_payload.get("execution_time", 0.0) or 0.0),
                        metadata=result_payload.get("metadata"),
                    )
                except Exception as exc:  # noqa: BLE001
                    err_msg = (
                        f"worker_output_parse_failed: {exc}; "
                        f"stdout={stdout.decode(errors='ignore')[:500]}; "
                        f"stderr={stderr.decode(errors='ignore')[:500]}"
                    )
                    return RLMExecutionResult(response="", err=err_msg)
            err_text = stderr.decode(errors="ignore").strip() or stdout.decode(errors="ignore").strip()
            if proc.returncode != 0:
                err_text = err_text or f"RLM worker exited with code {proc.returncode}"
            return RLMExecutionResult(response="", err=err_text or "RLM worker produced no output")
        except Exception as exc:  # noqa: BLE001
            return RLMExecutionResult(response="", err=str(exc))
        finally:
            for path in [input_path, output_path]:
                if path:
                    with contextlib.suppress(OSError):
                        Path(path).unlink()

    async def arun_many(
        self,
        prompts: list[str],
        max_concurrent: int = 4,
        progress_desc: str | None = None,
        parent_task_id: int | None = None,
    ) -> list[RLMExecutionResult]:
        semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))
        results: list[Optional[RLMExecutionResult]] = [None] * len(prompts)

        async def _one(index: int, prompt: str) -> tuple[int, RLMExecutionResult]:
            async with semaphore:
                return index, await self.arun(prompt)

        tasks = [asyncio.create_task(_one(i, prompt)) for i, prompt in enumerate(prompts)]
        subtask_desc = progress_desc or "rlm"
        subtask_id = progress_manager.add_task(f"  [sub] {subtask_desc}", total=len(tasks))
        try:
            for fut in asyncio.as_completed(tasks):
                index, result = await fut
                results[index] = result
                progress_manager.update(subtask_id, advance=1)
                if parent_task_id is not None:
                    progress_manager.update(parent_task_id, advance=1)
        finally:
            progress_manager.remove_task(subtask_id)
            for task in tasks:
                if not task.done():
                    task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        return [result if result is not None else RLMExecutionResult(response="", err="missing_result") for result in results]
