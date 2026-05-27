from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Optional, cast

from dotenv import load_dotenv

from src.exps_performance.rich_ui import progress_manager

load_dotenv()

PROGRESS_PREFIX = "__RLM_PROGRESS__"
_ACTIVE_PROGRESS_REPORTER: "WorkerProgressReporter | None" = None
_INSTRUMENTED_RLM_CLASS: Any | None = None


@dataclass
class RLMExecutionResult:
    response: str
    err: str
    execution_time: float = 0.0
    cost_usd: float = 0.0
    metadata: Optional[dict[str, Any]] = None


@dataclass
class WorkerProgressReporter:
    stream: Any

    def __post_init__(self) -> None:
        self._lock = Lock()
        self.llm_calls = 0
        self.max_depth_seen = 0

    def emit(self, event_type: str, *, depth: int | None = None, **payload: Any) -> None:
        with self._lock:
            if event_type == "lm_call_start":
                self.llm_calls += 1
            if depth is not None:
                self.max_depth_seen = max(self.max_depth_seen, int(depth))
            event: dict[str, Any] = {
                "type": event_type,
                "llm_calls": self.llm_calls,
                "max_depth_seen": self.max_depth_seen,
            }
            if depth is not None:
                event["depth"] = int(depth)
            event.update(payload)
            print(f"{PROGRESS_PREFIX}{json.dumps(event, ensure_ascii=False)}", file=self.stream, flush=True)


@dataclass
class LiveRLMTaskState:
    llm_calls: int = 0
    max_depth_seen: int = 0
    last_phase: str = "queued"
    active_subcalls: int = 0


def set_active_progress_reporter(reporter: WorkerProgressReporter | None) -> None:
    global _ACTIVE_PROGRESS_REPORTER
    _ACTIVE_PROGRESS_REPORTER = reporter


def _emit_progress_event(event_type: str, *, depth: int | None = None, **payload: Any) -> None:
    if _ACTIVE_PROGRESS_REPORTER is None:
        return
    _ACTIVE_PROGRESS_REPORTER.emit(event_type, depth=depth, **payload)


def _install_instrumented_rlm_class() -> Any:
    global _INSTRUMENTED_RLM_CLASS
    if _INSTRUMENTED_RLM_CLASS is not None:
        return _INSTRUMENTED_RLM_CLASS

    import rlm as rlm_pkg
    import rlm.core.rlm as rlm_core

    base_cls = rlm_core.RLM
    if getattr(base_cls, "__name__", "") == "InstrumentedRLM":
        _INSTRUMENTED_RLM_CLASS = base_cls
        return base_cls

    class InstrumentedRLM(base_cls):  # type: ignore[misc, valid-type]
        def completion(self, prompt: str | dict[str, Any], root_prompt: str | None = None) -> Any:
            _emit_progress_event("task_start", depth=getattr(self, "depth", 0), max_depth=getattr(self, "max_depth", 0))
            result = super().completion(prompt, root_prompt=root_prompt)
            _emit_progress_event("task_complete", depth=getattr(self, "depth", 0))
            return result

        def _completion_turn(self, prompt: Any, lm_handler: Any, environment: Any) -> Any:
            _emit_progress_event("lm_call_start", depth=getattr(self, "depth", 0), phase="iteration")
            result = super()._completion_turn(prompt=prompt, lm_handler=lm_handler, environment=environment)
            _emit_progress_event(
                "lm_call_complete",
                depth=getattr(self, "depth", 0),
                phase="iteration",
                duration=float(getattr(result, "iteration_time", 0.0) or 0.0),
            )
            return result

        def _default_answer(self, message_history: list[dict[str, Any]], lm_handler: Any) -> str:
            _emit_progress_event("lm_call_start", depth=getattr(self, "depth", 0), phase="default_answer")
            result = super()._default_answer(message_history, lm_handler)
            _emit_progress_event("lm_call_complete", depth=getattr(self, "depth", 0), phase="default_answer")
            return cast(str, result)

        def _fallback_answer(self, message: str | dict[str, Any]) -> str:
            _emit_progress_event("lm_call_start", depth=getattr(self, "depth", 0), phase="fallback")
            result = super()._fallback_answer(message)
            _emit_progress_event("lm_call_complete", depth=getattr(self, "depth", 0), phase="fallback")
            return cast(str, result)

        def _compact_history(
            self, lm_handler: Any, environment: Any, message_history: list[dict[str, Any]], compaction_count: int = 1
        ) -> list[dict[str, Any]]:
            _emit_progress_event("lm_call_start", depth=getattr(self, "depth", 0), phase="compaction")
            result = super()._compact_history(lm_handler, environment, message_history, compaction_count)
            _emit_progress_event("lm_call_complete", depth=getattr(self, "depth", 0), phase="compaction")
            return cast(list[dict[str, Any]], result)

    _INSTRUMENTED_RLM_CLASS = InstrumentedRLM
    rlm_core.RLM = InstrumentedRLM
    if hasattr(rlm_pkg, "RLM"):
        rlm_pkg.RLM = InstrumentedRLM
    return InstrumentedRLM


def _format_live_stats(state: LiveRLMTaskState, max_depth: int) -> str:
    stats = f"depth={state.max_depth_seen}/{max_depth} calls={state.llm_calls}"
    if state.active_subcalls > 0:
        stats += f" subcalls={state.active_subcalls}"
    if state.last_phase:
        stats += f" phase={state.last_phase}"
    return stats


def _consume_live_event(state: LiveRLMTaskState, event: dict[str, Any]) -> None:
    state.llm_calls = int(event.get("llm_calls", state.llm_calls) or state.llm_calls)
    state.max_depth_seen = max(state.max_depth_seen, int(event.get("max_depth_seen", 0) or 0))

    event_type = str(event.get("type", ""))
    depth = int(event.get("depth", state.max_depth_seen) or state.max_depth_seen)
    if event_type == "subcall_start":
        state.active_subcalls += 1
        state.max_depth_seen = max(state.max_depth_seen, depth)
        state.last_phase = f"subcall@d{depth}"
    elif event_type == "subcall_complete":
        state.active_subcalls = max(0, state.active_subcalls - 1)
        state.max_depth_seen = max(state.max_depth_seen, depth)
        state.last_phase = f"return@d{depth}"
    elif event_type == "lm_call_start":
        phase = str(event.get("phase", "iteration") or "iteration")
        state.last_phase = f"{phase}@d{depth}"
    elif event_type == "lm_call_complete":
        phase = str(event.get("phase", "iteration") or "iteration")
        state.last_phase = f"{phase}_done@d{depth}"
    elif event_type == "task_complete":
        state.last_phase = f"done@d{depth}"
    elif event_type == "task_start":
        state.last_phase = f"started@d{depth}"


def _format_aggregate_stats(done: int, total: int, active: int, max_concurrent: int, total_cost_usd: float) -> str:
    return f"batch={done}/{total} " f"active={active}/{max_concurrent} " f"cost=${total_cost_usd:,.4f}"


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
            _install_instrumented_rlm_class()
            from rlm import RLM
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Could not import `rlm`. Install the `rlms` package or point " "--rlm_repo_path at a checkout of https://github.com/alexzhang13/rlm."
            ) from exc
        return RLM

    def _load_rlm_logger_class(self) -> Any:
        self._ensure_repo_path()
        try:
            from rlm.logger import RLMLogger
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Could not import `RLMLogger` from the external rlm package.") from exc
        return RLMLogger

    def _resolve_backend(self) -> str:
        backend = str(getattr(self.args, "rlm_backend", None) or getattr(self.args, "backend", "openrouter"))
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
            kwargs["base_url"] = getattr(self.args, "openrouter_base_url", None) or os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
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
            on_subcall_start=lambda depth, model, _prompt_preview: _emit_progress_event(
                "subcall_start",
                depth=depth,
                model=str(model),
            ),
            on_subcall_complete=lambda depth, model, duration, error_or_none: _emit_progress_event(
                "subcall_complete",
                depth=depth,
                model=str(model),
                duration=float(duration),
                error=str(error_or_none) if error_or_none else "",
            ),
        )
        return self._rlm

    def _serialize_args(self) -> dict[str, Any]:
        raw_args = dict(vars(self.args))
        serialized: dict[str, Any] = {}
        for key, value in raw_args.items():
            if key.startswith("_"):
                continue
            if isinstance(value, Path):
                serialized[key] = str(value)
                continue
            try:
                json.dumps(value)
                serialized[key] = value
            except TypeError:
                continue
        return serialized

    def run(self, prompt: str) -> RLMExecutionResult:
        try:
            completion = self._get_rlm().completion(prompt)
            usage_summary = getattr(completion, "usage_summary", None)
            total_cost = float(getattr(usage_summary, "total_cost", 0.0) or 0.0)
            return RLMExecutionResult(
                response=str(getattr(completion, "response", "") or ""),
                err="ok",
                execution_time=float(getattr(completion, "execution_time", 0.0) or 0.0),
                cost_usd=total_cost,
                metadata=getattr(completion, "metadata", None),
            )
        except Exception as exc:  # noqa: BLE001
            return RLMExecutionResult(response="", err=str(exc))

    async def arun(
        self,
        prompt: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> RLMExecutionResult:
        payload = {"args": self._serialize_args(), "prompt": prompt}
        input_path = ""
        output_path = ""
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as input_file:
                json.dump(payload, input_file)
                input_path = input_file.name
            output_path = f"{input_path}.out.json"
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                "-m",
                "src.exps_performance.core.rlm_worker",
                input_path,
                output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async def _read_stdout() -> None:
                if proc.stdout is None:
                    return
                while True:
                    raw = await proc.stdout.readline()
                    if not raw:
                        break
                    line = raw.decode(errors="ignore").strip()
                    if not line:
                        continue
                    if line.startswith(PROGRESS_PREFIX):
                        payload_str = line[len(PROGRESS_PREFIX) :]
                        with contextlib.suppress(json.JSONDecodeError):
                            event = json.loads(payload_str)
                            if progress_callback is not None:
                                progress_callback(event)
                        continue
                    stdout_lines.append(line)

            async def _read_stderr() -> None:
                if proc.stderr is None:
                    return
                while True:
                    raw = await proc.stderr.readline()
                    if not raw:
                        break
                    line = raw.decode(errors="ignore").strip()
                    if line:
                        stderr_lines.append(line)

            await asyncio.gather(_read_stdout(), _read_stderr())
            await proc.wait()
            if output_path and Path(output_path).exists():
                try:
                    result_payload = json.loads(Path(output_path).read_text())
                    return RLMExecutionResult(
                        response=str(result_payload.get("response", "") or ""),
                        err=str(result_payload.get("err", "ok") or "ok"),
                        execution_time=float(result_payload.get("execution_time", 0.0) or 0.0),
                        cost_usd=float(result_payload.get("cost_usd", 0.0) or 0.0),
                        metadata=result_payload.get("metadata"),
                    )
                except Exception as exc:  # noqa: BLE001
                    err_msg = (
                        f"worker_output_parse_failed: {exc}; " f"stdout={' '.join(stdout_lines)[:500]}; " f"stderr={' '.join(stderr_lines)[:500]}"
                    )
                    return RLMExecutionResult(response="", err=err_msg)
            err_text = "\n".join(stderr_lines).strip() or "\n".join(stdout_lines).strip()
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
        on_completed: Callable[[int, RLMExecutionResult], Any] | None = None,
    ) -> list[RLMExecutionResult]:
        max_concurrent = max(1, int(max_concurrent))
        semaphore = asyncio.Semaphore(max_concurrent)
        results: list[Optional[RLMExecutionResult]] = [None] * len(prompts)
        max_depth = max(1, int(getattr(self.args, "rlm_max_depth", 2)))
        total_tasks = len(prompts)
        aggregate = {
            "done": 0,
            "active": 0,
            "cost_usd": 0.0,
        }

        def _aggregate_stats() -> str:
            return _format_aggregate_stats(
                int(aggregate["done"]),
                total_tasks,
                int(aggregate["active"]),
                max_concurrent,
                float(aggregate["cost_usd"]),
            )

        async def _one(index: int, prompt: str) -> tuple[int, RLMExecutionResult]:
            async with semaphore:
                aggregate["active"] += 1
                current_stats = _aggregate_stats()
                progress_manager.update(subtask_id, stats=current_stats)
                if parent_task_id is not None:
                    progress_manager.update(parent_task_id, stats=current_stats)
                state = LiveRLMTaskState()
                task_id = progress_manager.add_task(
                    f"    [rlm] {(progress_desc or 'rlm')} #{index + 1:03d}",
                    total=max_depth,
                    stats=_format_live_stats(state, max_depth),
                )

                def _on_event(event: dict[str, Any]) -> None:
                    _consume_live_event(state, event)
                    progress_manager.update(
                        task_id,
                        completed=min(max_depth, state.max_depth_seen),
                        stats=_format_live_stats(state, max_depth),
                    )

                try:
                    result = await self.arun(prompt, progress_callback=_on_event)
                    state.last_phase = "done" if result.err == "ok" else "error"
                    progress_manager.update(
                        task_id,
                        completed=min(max_depth, state.max_depth_seen),
                        stats=_format_live_stats(state, max_depth),
                    )
                    return index, result
                finally:
                    aggregate["active"] = max(0, aggregate["active"] - 1)
                    current_stats = _aggregate_stats()
                    progress_manager.update(subtask_id, stats=current_stats)
                    if parent_task_id is not None:
                        progress_manager.update(parent_task_id, stats=current_stats)
                    progress_manager.remove_task(task_id)

        tasks = [asyncio.create_task(_one(i, prompt)) for i, prompt in enumerate(prompts)]
        subtask_desc = progress_desc or "rlm"
        subtask_id = progress_manager.add_task(
            f"  [sub] {subtask_desc}",
            total=len(tasks),
            stats=_format_aggregate_stats(0, total_tasks, 0, max_concurrent, 0.0),
        )
        if parent_task_id is not None:
            progress_manager.update(
                parent_task_id,
                stats=_format_aggregate_stats(0, total_tasks, 0, max_concurrent, 0.0),
            )
        try:
            for fut in asyncio.as_completed(tasks):
                index, result = await fut
                results[index] = result
                aggregate["done"] += 1
                aggregate["cost_usd"] += float(getattr(result, "cost_usd", 0.0) or 0.0)
                if on_completed is not None:
                    maybe_awaitable = on_completed(index, result)
                    if inspect.isawaitable(maybe_awaitable):
                        await maybe_awaitable
                stats = _aggregate_stats()
                progress_manager.update(subtask_id, advance=1, stats=stats)
                if parent_task_id is not None:
                    progress_manager.update(parent_task_id, advance=1, stats=stats)
        finally:
            progress_manager.remove_task(subtask_id)
            for task in tasks:
                if not task.done():
                    task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        return [result if result is not None else RLMExecutionResult(response="", err="missing_result") for result in results]
