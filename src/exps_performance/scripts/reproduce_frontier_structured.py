#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_RESULTS_ROOT = REPO_ROOT / "src" / "exps_performance" / "results"
RESULTS_ROOT = REPO_ROOT / "results"
FRONTIER_STRUCTURED_REFERENCE = RESULTS_ROOT / "frontier_structured_seed1_reference.jsonl"

SEED1_KINDS = [
    "spp",
    "bsp",
    "edp",
    "gcp",
    "gcp_d",
    "tsp",
    "tsp_d",
    "ksp",
    "msp",
    "clrs30",
    "add",
    "sub",
    "mul",
    "lcs",
    "rod",
    "knap",
    "ilp_assign",
    "ilp_partition",
    "ilp_prod",
]


@dataclass(frozen=True)
class FrontierConfig:
    name: str
    n: int
    digits_list: list[int]
    gsm_samples: int
    clrs_samples: int
    kinds: list[str]
    reference_path: Path
    expected_rows: int
    run_slug: str
    openrouter_max_concurrency: int
    batch_size: int
    checkpoint_every: int
    request_timeout: float


@dataclass(frozen=True)
class ModelPreset:
    alias: str
    label: str
    model: str
    reasoning_effort: Optional[str] = None
    reasoning_max_tokens: Optional[int] = None
    verbosity: Optional[str] = None

    @property
    def outdir(self) -> str:
        return f"{self.model.split('/')[-1]}_seed1"


FRONTIER_STRUCTURED = FrontierConfig(
    name="frontier_structured",
    n=60,
    digits_list=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    gsm_samples=0,
    clrs_samples=500,
    kinds=SEED1_KINDS,
    reference_path=FRONTIER_STRUCTURED_REFERENCE,
    expected_rows=350,
    run_slug="frontier_structured",
    openrouter_max_concurrency=350,
    batch_size=350,
    checkpoint_every=350,
    request_timeout=1200.0,
)

MODEL_PRESETS = {
    "gpt": ModelPreset(
        alias="gpt",
        label="GPT-5.4",
        model="openai/gpt-5.4",
        reasoning_effort="xhigh",
    ),
    "opus": ModelPreset(
        alias="opus",
        label="Claude Opus 4.6",
        model="anthropic/claude-opus-4.6",
        reasoning_max_tokens=95000,
        verbosity="max",
    ),
}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def ensure_repo_root() -> None:
    os.chdir(REPO_ROOT)
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def best_effort_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def validate_frontier_config(config: FrontierConfig) -> None:
    actual_rows = len(best_effort_jsonl(config.reference_path))
    if actual_rows != config.expected_rows:
        raise SystemExit(
            f"frontier config size mismatch for {config.name}: expected {config.expected_rows}, got {actual_rows} from {config.reference_path}"
        )


def latest_resumable_exp_id(preset: ModelPreset) -> Optional[str]:
    tb_root = SRC_RESULTS_ROOT / preset.outdir / "tb"
    if not tb_root.exists():
        return None
    run_dirs = [path for path in tb_root.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return run_dirs[0].name


def build_frontier_args(preset: ModelPreset, exp_id: Optional[str], *, resume: bool, config: FrontierConfig = FRONTIER_STRUCTURED) -> Any:
    ensure_repo_root()
    from src.exps_performance.main import Args

    kwargs: dict[str, Any] = {
        "root": str(SRC_RESULTS_ROOT.parent),
        "n": config.n,
        "digits_list": list(config.digits_list),
        "kinds": list(config.kinds),
        "gsm_samples": config.gsm_samples,
        "clrs_samples": config.clrs_samples,
        "seed": 1,
        "backend": "openrouter",
        "model": preset.model,
        "tb_disable": True,
        "exp_id": exp_id,
        "subset_reference_jsonl": str(config.reference_path),
        "openrouter_structured_outputs": True,
        "openrouter_structured_strict": True,
        "openrouter_response_healing": True,
        "openrouter_reasoning_enabled": True,
        "openrouter_reasoning_exclude": True,
        "openrouter_retry_attempts": 6,
        "openrouter_max_concurrency": config.openrouter_max_concurrency,
        "stage_batch_retry_attempts": 3,
        "max_tokens": 100000,
        "batch_size": config.batch_size,
        "checkpoint_every": config.checkpoint_every,
        "request_timeout": config.request_timeout,
        "exec_code": True,
        "controlled_sim": False,
        "resume": resume,
    }
    if preset.reasoning_effort:
        kwargs["openrouter_reasoning_effort"] = preset.reasoning_effort
    if preset.reasoning_max_tokens:
        kwargs["openrouter_reasoning_max_tokens"] = preset.reasoning_max_tokens
    if preset.verbosity:
        kwargs["openrouter_verbosity"] = preset.verbosity
    return Args(**kwargs)


def run_model(alias: str, exp_id: Optional[str], *, resume: bool) -> int:
    load_dotenv(REPO_ROOT / ".env")
    validate_frontier_config(FRONTIER_STRUCTURED)
    ensure_repo_root()
    from src.exps_performance.main import run

    preset = MODEL_PRESETS[alias]
    args = build_frontier_args(preset, exp_id, resume=resume)
    print(
        json.dumps(
            {
                "event": "worker_start",
                "model": preset.model,
                "exp_id": exp_id,
                "frontier_config": FRONTIER_STRUCTURED.name,
                "resume": resume,
                "reference_path": str(FRONTIER_STRUCTURED.reference_path),
                "expected_rows": FRONTIER_STRUCTURED.expected_rows,
                "openrouter_retry_attempts": args.openrouter_retry_attempts,
                "stage_batch_retry_attempts": args.stage_batch_retry_attempts,
                "openrouter_max_concurrency": args.openrouter_max_concurrency,
                "request_timeout": args.request_timeout,
                "max_tokens": args.max_tokens,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    start = time.time()
    run(args)
    print(
        json.dumps(
            {
                "event": "worker_done",
                "model": preset.model,
                "elapsed_seconds": round(time.time() - start, 2),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def default_exp_id(alias: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = FRONTIER_STRUCTURED
    return f"run_{stamp}_{config.run_slug}_{alias}_seed1_rt{int(config.request_timeout)}_c{config.openrouter_max_concurrency}_or6_sr3"


def run_frontier(model: str, *, fresh: bool, exp_id: Optional[str]) -> int:
    load_dotenv(REPO_ROOT / ".env")
    validate_frontier_config(FRONTIER_STRUCTURED)
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set after loading repo .env", file=sys.stderr)
        return 2
    selected = list(MODEL_PRESETS) if model == "both" else [model]
    if exp_id and len(selected) > 1:
        print("--exp-id can only be used with --model gpt or --model opus", file=sys.stderr)
        return 2

    for alias in selected:
        preset = MODEL_PRESETS[alias]
        selected_exp_id = exp_id
        resume = not fresh
        if selected_exp_id is None and fresh:
            selected_exp_id = default_exp_id(alias)
        elif selected_exp_id is None and resume:
            selected_exp_id = latest_resumable_exp_id(preset)
        print(
            json.dumps(
                {
                    "event": "launch",
                    "model": preset.model,
                    "frontier_config": FRONTIER_STRUCTURED.name,
                    "exp_id": selected_exp_id,
                    "resume": resume,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        run_model(alias, selected_exp_id, resume=resume)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frontier structured reproduction.")
    parser.add_argument("--model", choices=["gpt", "opus", "both"], default="both")
    parser.add_argument("--fresh", action="store_true", help="Start a fresh run instead of resuming the latest checkpointed run.")
    parser.add_argument("--exp-id", help="Resume or launch a specific exp_id. Only valid with a single model.")
    parser.add_argument("--worker", choices=["gpt", "opus"], help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_model(args.worker, args.exp_id, resume=bool(args.resume))
    return run_frontier(args.model, fresh=bool(args.fresh), exp_id=args.exp_id)


if __name__ == "__main__":
    raise SystemExit(main())
