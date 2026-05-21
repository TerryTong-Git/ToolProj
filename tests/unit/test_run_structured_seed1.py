import importlib.util
import os
import sys
from pathlib import Path


def load_runner_module():  # type: ignore[no-untyped-def]
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "results" / "run_structured_seed1.py"
    spec = importlib.util.spec_from_file_location("run_structured_seed1", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_subset350_runner_builds_reproduction_args(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    old_cwd = os.getcwd()
    runner = load_runner_module()
    try:
        args = runner.build_args(runner.MODEL_PRESETS["gpt"], "exp-test", resume=False)
    finally:
        os.chdir(old_cwd)

    assert args.backend == "openrouter"
    assert args.model == "openai/gpt-5.4"
    assert args.exp_id == "exp-test"
    assert args.subset_reference_jsonl.endswith("results/seed1_subset_reference_350.jsonl")
    assert args.openrouter_structured_outputs is True
    assert args.openrouter_reasoning_enabled is True
    assert args.openrouter_reasoning_effort == "xhigh"
    assert args.exec_code is True
    assert args.controlled_sim is False
    assert args.batch_size == 350
    assert args.checkpoint_every == 350
