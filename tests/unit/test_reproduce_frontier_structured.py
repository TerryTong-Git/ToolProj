import importlib
import os


def load_runner_module():  # type: ignore[no-untyped-def]
    return importlib.import_module("src.reasoning_benchmark.reproductions.frontier.run")


def test_frontier_structured_runner_builds_reproduction_args(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    old_cwd = os.getcwd()
    runner = load_runner_module()
    try:
        args = runner.build_frontier_args(runner.MODEL_PRESETS["gpt"], "exp-test", resume=False)
    finally:
        os.chdir(old_cwd)

    assert args.backend == "openrouter"
    assert args.model == "openai/gpt-5.4"
    assert args.exp_id == "exp-test"
    assert args.subset_reference_jsonl.endswith("src/reasoning_benchmark/reproductions/frontier/fixtures/seed1_reference.jsonl")
    assert args.openrouter_structured_outputs is True
    assert args.openrouter_reasoning_enabled is True
    assert args.openrouter_reasoning_effort == "xhigh"
    assert args.exec_code is True
    assert args.controlled_sim is False
    assert args.batch_size == 350
    assert args.checkpoint_every == 350
