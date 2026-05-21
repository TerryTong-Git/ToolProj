#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_RESULTS_ROOT = REPO_ROOT / "src" / "exps_performance" / "results"
RESULTS_ROOT = REPO_ROOT / "results"
SUBSET_REFERENCE = RESULTS_ROOT / "seed1_subset_reference.jsonl"
SUBSET_REFERENCE_350 = RESULTS_ROOT / "seed1_subset_reference_350.jsonl"
SUBSET_REFERENCE_30 = RESULTS_ROOT / "seed1_subset_reference_30.jsonl"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

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
class DatasetProfile:
    name: str
    label: str
    n: int
    digits_list: list[int]
    gsm_samples: int
    clrs_samples: int
    kinds: list[str]
    subset_reference_jsonl: Optional[Path]
    expected_rows: int
    exp_suffix: str
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


@dataclass
class LaunchSpec:
    preset: ModelPreset
    exp_id: str
    run_dir: Path
    log_path: Path
    process: subprocess.Popen[str]
    started_at: float
    log_handle: Any
    resumed: bool


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


DATASET_PROFILES = {
    "full": DatasetProfile(
        name="full",
        label="Full Seed1 Reference",
        n=60,
        digits_list=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        gsm_samples=0,
        clrs_samples=500,
        kinds=SEED1_KINDS,
        subset_reference_jsonl=SUBSET_REFERENCE,
        expected_rows=686,
        exp_suffix="full686",
        openrouter_max_concurrency=350,
        batch_size=350,
        checkpoint_every=350,
        request_timeout=1200.0,
    ),
    "quarter": DatasetProfile(
        name="quarter",
        label="Quarter Seed1 Slice",
        n=15,
        digits_list=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        gsm_samples=0,
        clrs_samples=125,
        kinds=SEED1_KINDS,
        subset_reference_jsonl=None,
        expected_rows=350,
        exp_suffix="quarter350",
        openrouter_max_concurrency=350,
        batch_size=350,
        checkpoint_every=350,
        request_timeout=1200.0,
    ),
    "subset350": DatasetProfile(
        name="subset350",
        label="350-row Seed1 Reference Subset",
        n=60,
        digits_list=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        gsm_samples=0,
        clrs_samples=500,
        kinds=SEED1_KINDS,
        subset_reference_jsonl=SUBSET_REFERENCE_350,
        expected_rows=350,
        exp_suffix="subset350",
        openrouter_max_concurrency=350,
        batch_size=350,
        checkpoint_every=350,
        request_timeout=1200.0,
    ),
    "subset30": DatasetProfile(
        name="subset30",
        label="30-row Seed1 Reference Subset",
        n=60,
        digits_list=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        gsm_samples=0,
        clrs_samples=500,
        kinds=SEED1_KINDS,
        subset_reference_jsonl=SUBSET_REFERENCE_30,
        expected_rows=30,
        exp_suffix="subset30",
        openrouter_max_concurrency=30,
        batch_size=30,
        checkpoint_every=30,
        request_timeout=1200.0,
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
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def ensure_repo_root() -> None:
    os.chdir(REPO_ROOT)
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def build_args(preset: ModelPreset, exp_id: str, *, resume: bool, profile: DatasetProfile) -> Any:
    ensure_repo_root()
    from src.exps_performance.main import Args

    common: dict[str, Any] = {
        "root": str(REPO_ROOT / "src" / "exps_performance"),
        "n": profile.n,
        "digits_list": list(profile.digits_list),
        "kinds": list(profile.kinds),
        "gsm_samples": profile.gsm_samples,
        "clrs_samples": profile.clrs_samples,
        "seed": 1,
        "backend": "openrouter",
        "model": preset.model,
        "tb_disable": True,
        "exp_id": exp_id,
        "openrouter_structured_outputs": True,
        "openrouter_structured_strict": True,
        "openrouter_response_healing": True,
        "openrouter_reasoning_enabled": True,
        "openrouter_reasoning_exclude": True,
        "openrouter_retry_attempts": 6,
        "openrouter_max_concurrency": profile.openrouter_max_concurrency,
        "stage_batch_retry_attempts": 3,
        "max_tokens": 100000,
        "batch_size": profile.batch_size,
        "checkpoint_every": profile.checkpoint_every,
        "request_timeout": profile.request_timeout,
        "exec_code": True,
        "controlled_sim": False,
        "resume": resume,
    }
    if profile.subset_reference_jsonl is not None:
        common["subset_reference_jsonl"] = str(profile.subset_reference_jsonl)
    if preset.reasoning_effort:
        common["openrouter_reasoning_effort"] = preset.reasoning_effort
    if preset.reasoning_max_tokens:
        common["openrouter_reasoning_max_tokens"] = preset.reasoning_max_tokens
    if preset.verbosity:
        common["openrouter_verbosity"] = preset.verbosity
    return Args(**common)


def expected_dataset_size(profile: DatasetProfile) -> int:
    if profile.subset_reference_jsonl is not None:
        return len(best_effort_jsonl(profile.subset_reference_jsonl))
    ensure_repo_root()
    from src.exps_performance.dataset import make_dataset

    data = make_dataset(
        profile.kinds,
        n=profile.n,
        digits_list=list(profile.digits_list),
        gsm_samples=profile.gsm_samples,
        clrs_samples=profile.clrs_samples,
    )
    return len(data)


def best_effort_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return rows


def tail_lines(path: Path, limit: int = 4) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cleaned = [ANSI_RE.sub("", line).rstrip() for line in lines if line.strip()]
    return cleaned[-limit:]


def summarize_run(run_dir: Path) -> dict[str, Any]:
    res_rows = best_effort_jsonl(run_dir / "res.jsonl")
    stat_rows = best_effort_jsonl(run_dir / "llm_stage_stats.jsonl")
    summary: dict[str, Any] = {
        "rows": len(res_rows),
        "sim_ok": sum(1 for row in res_rows if row.get("sim_err_msg") == "ok"),
        "sim_correct": sum(1 for row in res_rows if row.get("sim_correct") is True),
        "nl_rows": sum(1 for row in res_rows if bool(row.get("nl_question"))),
        "code_rows": sum(1 for row in res_rows if bool(row.get("code_question"))),
        "code_ok": sum(1 for row in res_rows if row.get("code_err_msg") == "ok,ok"),
        "last_stat": stat_rows[-1] if stat_rows else {},
    }
    return summary


def load_args_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def latest_resumable_exp_id(preset: ModelPreset, profile: DatasetProfile) -> Optional[str]:
    tb_root = SRC_RESULTS_ROOT / preset.outdir / "tb"
    if not tb_root.exists():
        return None
    dirs = [d for d in tb_root.iterdir() if d.is_dir()]
    if not dirs:
        return None

    def _int_arg(args: dict[str, Any], key: str) -> int:
        value = args.get(key, None)
        return -1 if value is None else int(value)

    def matches_profile(args: dict[str, Any]) -> bool:
        if _int_arg(args, "n") != profile.n:
            return False
        if _int_arg(args, "clrs_samples") != profile.clrs_samples:
            return False
        if _int_arg(args, "gsm_samples") != profile.gsm_samples:
            return False
        if list(args.get("digits_list", [])) != list(profile.digits_list):
            return False
        if list(args.get("kinds", [])) != list(profile.kinds):
            return False
        current_subset = args.get("subset_reference_jsonl")
        expected_subset = str(profile.subset_reference_jsonl) if profile.subset_reference_jsonl is not None else None
        return current_subset == expected_subset

    def sort_key(path: Path) -> tuple[int, int, int, int, int, float]:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            mtime = 0.0
        args = load_args_json(path / "args.json")
        profile_match = int(matches_profile(args))
        has_nonempty_res = int((path / "res.jsonl").exists() and (path / "res.jsonl").stat().st_size > 0)
        is_100k = int(int(args.get("max_tokens", 0) or 0) >= 100000)
        is_structured = int(bool(args.get("openrouter_structured_outputs", False)))
        has_reasoning = int(bool(args.get("openrouter_reasoning_enabled", False)))
        return (profile_match, has_nonempty_res, is_100k, is_structured, has_reasoning, mtime)

    dirs.sort(key=sort_key, reverse=True)
    chosen = None
    for path in dirs:
        args = load_args_json(path / "args.json")
        if matches_profile(args):
            chosen = path
            break
    return chosen.name if chosen else None


def format_state(spec: LaunchSpec) -> str:
    code = spec.process.poll()
    if code is None:
        return "running"
    if code == 0:
        return "done"
    return f"failed({code})"


def human_elapsed(started_at: float) -> str:
    elapsed = max(0, int(time.time() - started_at))
    mins, secs = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def render_dashboard(specs: list[LaunchSpec]) -> Group:
    summary_table = Table(title="Structured Seed1 Runner", expand=True)
    summary_table.add_column("Model", style="bold")
    summary_table.add_column("Mode")
    summary_table.add_column("State")
    summary_table.add_column("Elapsed", justify="right")
    summary_table.add_column("Rows", justify="right")
    summary_table.add_column("Sim Ok", justify="right")
    summary_table.add_column("Sim Correct", justify="right")
    summary_table.add_column("Sim Acc", justify="right")
    summary_table.add_column("NL Rows", justify="right")
    summary_table.add_column("Code Rows", justify="right")
    summary_table.add_column("Batch Try", justify="right")
    summary_table.add_column("Remaining", justify="right")
    summary_table.add_column("Empty", justify="right")

    log_panels: list[Panel] = []
    for spec in specs:
        summary = summarize_run(spec.run_dir)
        last = summary["last_stat"]
        sim_acc = "-"
        if summary["rows"]:
            sim_acc = f"{summary['sim_correct'] / summary['rows']:.1%}"
        summary_table.add_row(
            spec.preset.label,
            "resume" if spec.resumed else "fresh",
            format_state(spec),
            human_elapsed(spec.started_at),
            str(summary["rows"]),
            str(summary["sim_ok"]),
            str(summary["sim_correct"]),
            sim_acc,
            str(summary["nl_rows"]),
            str(summary["code_rows"]),
            str(last.get("batch_attempt", "-")),
            str(last.get("rows_remaining_after_attempt", "-")),
            str(last.get("empty_text_responses", "-")),
        )
        tail = tail_lines(spec.log_path, limit=5)
        tail_text = "\n".join(tail) if tail else "(no log output yet)"
        log_panels.append(
            Panel(
                tail_text,
                title=f"{spec.preset.label} log",
                subtitle=str(spec.log_path.relative_to(REPO_ROOT)),
                expand=True,
            )
        )

    info_table = Table.grid(expand=True)
    info_table.add_row(f"repo: {REPO_ROOT}")
    configured = specs[0].run_dir / "args.json" if specs else None
    config_bits = "openrouter_retry_attempts=6, stage_batch_retry_attempts=3, max_tokens=100000"
    if configured and configured.exists():
        args = load_args_json(configured)
        config_bits = (
            f"openrouter_retry_attempts={int(args.get('openrouter_retry_attempts', 0) or 0)}, "
            f"stage_batch_retry_attempts={int(args.get('stage_batch_retry_attempts', 0) or 0)}, "
            f"request_timeout={float(args.get('request_timeout', 0.0) or 0.0):.0f}s, "
            f"max_tokens={int(args.get('max_tokens', 0) or 0)}, "
            f"concurrency={int(args.get('openrouter_max_concurrency', 0) or 0)}"
        )
    info_table.add_row(f"config: {config_bits}")
    info_table.add_row("behavior: launcher works from any cwd, validates profile size before launch, resumes the latest matching checkpoint by default, and keeps child runs alive if the monitor exits")

    return Group(Panel(info_table, title="Launch Info", expand=True), summary_table, *log_panels)


def select_python() -> Path:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    return venv_python if venv_python.exists() else Path(sys.executable)


def launch_one(
    console: Console,
    preset: ModelPreset,
    stamp: str,
    *,
    resume_existing: bool,
    explicit_exp_id: Optional[str],
    profile: DatasetProfile,
) -> LaunchSpec:
    resumed = False
    exp_id = explicit_exp_id
    if exp_id is None and resume_existing:
        exp_id = latest_resumable_exp_id(preset, profile)
        resumed = exp_id is not None
    if exp_id is None:
        exp_id = (
            f"run_{stamp}_structured_{preset.alias}_seed1_{profile.exp_suffix}"
            f"_rt{int(profile.request_timeout)}_c{int(profile.openrouter_max_concurrency)}_or6_sr3"
        )
    run_dir = SRC_RESULTS_ROOT / preset.outdir / "tb" / exp_id
    log_stem = f"{preset.alias}_{exp_id}"
    log_suffix = f"resume_{stamp}" if resumed else f"fresh_{stamp}"
    log_path = RESULTS_ROOT / f"{log_stem}_{log_suffix}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    cmd = [str(select_python()), str(Path(__file__)), "--worker", preset.alias, "--profile", profile.name, "--exp-id", exp_id]
    if resumed:
        cmd.append("--resume")
    env = os.environ.copy()
    process = subprocess.Popen(
        cmd,
        cwd=str(Path.home()),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    action = "resuming" if resumed else "launched"
    console.print(f"[bold green]{action}[/bold green] {preset.label} pid={process.pid} exp_id={exp_id}")
    return LaunchSpec(
        preset=preset,
        exp_id=exp_id,
        run_dir=run_dir,
        log_path=log_path,
        process=process,
        started_at=time.time(),
        log_handle=log_handle,
        resumed=resumed,
    )


def run_worker(alias: str, exp_id: str, *, resume: bool, profile: DatasetProfile) -> int:
    load_dotenv(REPO_ROOT / ".env")
    ensure_repo_root()
    from src.exps_performance.main import run

    preset = MODEL_PRESETS[alias]
    args = build_args(preset, exp_id, resume=resume, profile=profile)
    print(
        json.dumps(
            {
                "event": "worker_start",
                "model": preset.model,
                "exp_id": exp_id,
                "profile": profile.name,
                "resume": resume,
                "cwd": os.getcwd(),
                "subset_reference_jsonl": str(profile.subset_reference_jsonl) if profile.subset_reference_jsonl else None,
                "n": profile.n,
                "clrs_samples": profile.clrs_samples,
                "openrouter_retry_attempts": 6,
                "stage_batch_retry_attempts": 3,
                "openrouter_max_concurrency": profile.openrouter_max_concurrency,
                "request_timeout": profile.request_timeout,
                "max_tokens": 100000,
            },
            indent=2,
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
                "exp_id": exp_id,
                "elapsed_seconds": round(time.time() - start, 2),
            }
        ),
        flush=True,
    )
    return 0


def run_parent(
    model: str,
    poll_seconds: float,
    *,
    resume_existing: bool,
    explicit_exp_id: Optional[str],
    profile: DatasetProfile,
) -> int:
    load_dotenv(REPO_ROOT / ".env")
    console = Console()
    if not os.environ.get("OPENROUTER_API_KEY"):
        console.print("[red]OPENROUTER_API_KEY is not set after loading repo .env[/red]")
        return 2
    actual_rows = expected_dataset_size(profile)
    if actual_rows != profile.expected_rows:
        console.print(
            f"[red]profile size mismatch[/red] {profile.name}: expected {profile.expected_rows}, got {actual_rows}. refusing to launch."
        )
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    selected = list(MODEL_PRESETS) if model == "both" else [model]
    if explicit_exp_id and model == "both":
        console.print("[red]--exp-id can only be used with --model gpt or --model opus[/red]")
        return 2
    specs = [
        launch_one(
            console,
            MODEL_PRESETS[key],
            stamp,
            resume_existing=resume_existing,
            explicit_exp_id=explicit_exp_id if len(selected) == 1 else None,
            profile=profile,
        )
        for key in selected
    ]

    try:
        with Live(render_dashboard(specs), console=console, refresh_per_second=2, screen=False) as live:
            while True:
                live.update(render_dashboard(specs))
                if all(spec.process.poll() is not None for spec in specs):
                    break
                time.sleep(poll_seconds)
            live.update(render_dashboard(specs))
    except KeyboardInterrupt:
        console.print("[yellow]monitor interrupted; detached child runs continue in background[/yellow]")
        for spec in specs:
            console.print(f"{spec.preset.label}: log={spec.log_path} run_dir={spec.run_dir}")
        return 130
    finally:
        for spec in specs:
            spec.log_handle.close()

    failed = [spec for spec in specs if spec.process.poll() not in (0, None)]
    if failed:
        console.print("[red]one or more runs failed[/red]")
        for spec in failed:
            console.print(f"{spec.preset.label}: exit={spec.process.poll()} log={spec.log_path}")
        return 1

    console.print("[bold green]all launched runs finished[/bold green]")
    for spec in specs:
        console.print(f"{spec.preset.label}: run_dir={spec.run_dir} log={spec.log_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch and monitor structured seed1 runs with rich logging.")
    parser.add_argument("--model", choices=["gpt", "opus", "both"], default="both")
    parser.add_argument("--profile", choices=sorted(DATASET_PROFILES), default="full")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--fresh", action="store_true", help="Start a fresh run instead of resuming the latest checkpointed run.")
    parser.add_argument("--exp-id", help="Resume or launch a specific exp_id. Only valid with a single model.")
    parser.add_argument("--worker", choices=["gpt", "opus"], help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = DATASET_PROFILES[args.profile]
    if args.worker:
        if not args.exp_id:
            raise SystemExit("--exp-id is required in worker mode")
        return run_worker(args.worker, args.exp_id, resume=bool(args.resume), profile=profile)
    return run_parent(args.model, args.poll_seconds, resume_existing=not args.fresh, explicit_exp_id=args.exp_id, profile=profile)


if __name__ == "__main__":
    raise SystemExit(main())
