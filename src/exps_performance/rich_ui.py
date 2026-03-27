from __future__ import annotations

import logging
from threading import RLock

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()


def setup_rich_logging() -> None:
    root = logging.getLogger()
    if any(isinstance(handler, RichHandler) for handler in root.handlers):
        return

    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=False,
        show_path=False,
        show_time=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.handlers = [handler]
    root.setLevel(logging.INFO)


class RichProgressManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(compact=True),
            console=console,
            transient=False,
            expand=True,
        )
        self._started = False
        self._live_tasks = 0

    def _start_if_needed(self) -> None:
        if not self._started:
            self._progress.start()
            self._started = True

    def add_task(self, description: str, total: int) -> TaskID:
        with self._lock:
            self._start_if_needed()
            self._live_tasks += 1
            return self._progress.add_task(description, total=total)

    def update(self, task_id: TaskID, advance: int = 0, completed: float | None = None) -> None:
        with self._lock:
            if self._started:
                self._progress.update(task_id, advance=advance, completed=completed)

    def remove_task(self, task_id: TaskID) -> None:
        with self._lock:
            if not self._started:
                return
            self._progress.remove_task(task_id)
            self._live_tasks = max(0, self._live_tasks - 1)
            if self._live_tasks == 0:
                self._progress.stop()
                self._started = False


progress_manager = RichProgressManager()
