"""Frontier structured-reasoning reproduction entrypoint."""

from __future__ import annotations

from src.reasoning_benchmark.scripts.reproduce_frontier_structured import *  # noqa: F401,F403
from src.reasoning_benchmark.scripts.reproduce_frontier_structured import main

if __name__ == "__main__":
    raise SystemExit(main())
