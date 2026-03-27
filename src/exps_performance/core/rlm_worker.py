from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from src.exps_performance.core.rlm_executor import (
    RecursiveLMExecutor,
    WorkerProgressReporter,
    set_active_progress_reporter,
)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m src.exps_performance.core.rlm_worker <input_json> <output_json>")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    payload = json.loads(input_path.read_text())
    args = SimpleNamespace(**payload["args"])
    prompt = str(payload["prompt"])

    set_active_progress_reporter(WorkerProgressReporter(sys.stdout))
    try:
        result = RecursiveLMExecutor(args).run(prompt)
    finally:
        set_active_progress_reporter(None)
    output = {
        "response": result.response,
        "err": result.err,
        "execution_time": result.execution_time,
        "cost_usd": result.cost_usd,
        "metadata": result.metadata,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
