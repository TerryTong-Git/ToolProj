"""Command-line entrypoint for the reasoning benchmark."""

from __future__ import annotations

import logging
import time

from src.reasoning_benchmark.runner import parse_args, run

logger = logging.getLogger(__name__)


def main() -> int:
    start_time = time.perf_counter()
    args = parse_args()
    run(args)
    elapsed_time = time.perf_counter() - start_time
    logger.info("Elapsed time: %.4f seconds", elapsed_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
