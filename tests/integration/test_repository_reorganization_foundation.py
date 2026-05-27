"""Foundation checks for the staged repository reorganization."""

from __future__ import annotations

import importlib
import importlib.util


def test_reasoning_benchmark_canonical_interfaces() -> None:
    run_config = importlib.import_module("src.reasoning_benchmark.run_config")
    task_sets = importlib.import_module("src.reasoning_benchmark.task_sets")
    strategies = importlib.import_module("src.reasoning_benchmark.reasoning_strategies")
    executor = importlib.import_module("src.reasoning_benchmark.execution.python_executor")
    llm_clients = importlib.import_module("src.reasoning_benchmark.llm_clients")
    checkpoints = importlib.import_module("src.reasoning_benchmark.checkpoints")
    records = importlib.import_module("src.reasoning_benchmark.records")

    assert run_config.BenchmarkRunConfig.__name__ == "BenchmarkRunConfig"
    assert callable(task_sets.load_benchmark_tasks)
    assert strategies.NaturalLanguageReasoning.__name__ == "NaturalLanguageReasoning"
    assert strategies.CodeSimulationReasoning.__name__ == "CodeSimulationReasoning"
    assert strategies.CodeExecutionReasoning.__name__ == "CodeExecutionReasoning"
    assert strategies.ControlledSimulationReasoning.__name__ == "ControlledSimulationReasoning"
    assert executor.ProgramChatInterface.__name__ == "ProgramChatInterface"
    assert callable(llm_clients.build_llm_client)
    assert checkpoints.CheckpointManager.__name__ == "CheckpointManager"
    assert records.Record.__name__ == "Record"


def test_translation_discrimination_canonical_interfaces() -> None:
    canonical = importlib.import_module("src.translation_discrimination.source_label")

    assert canonical.Sample.__name__ == "Sample"
    assert canonical.Trial.__name__ == "Trial"
    assert canonical.Results.__name__ == "Results"
    assert callable(canonical.run_experiment)


def test_translation_additivity_canonical_interfaces() -> None:
    canonical_information = importlib.import_module("src.translation_additivity.information_additivity")
    canonical_native = importlib.import_module("src.translation_additivity.native_translation_additivity")

    assert canonical_information.Sample.__name__ == "Sample"
    assert canonical_information.ExperimentResults.__name__ == "ExperimentResults"
    assert callable(canonical_information.run_experiment)
    assert canonical_native.Sample.__name__ == "Sample"
    assert canonical_native.Trial.__name__ == "Trial"
    assert callable(canonical_native.build_arg_parser)


def test_legacy_experiment_packages_are_no_longer_import_surfaces() -> None:
    assert importlib.util.find_spec("src.exps_control_again") is None
    assert importlib.util.find_spec("src.exps_functional") is None
    assert importlib.util.find_spec("src.exps_performance.main") is None


def test_canonical_cli_modules_export_main() -> None:
    modules = [
        "src.reasoning_benchmark.cli",
        "src.reasoning_benchmark.reproductions.frontier.run",
        "src.translation_discrimination.cli",
        "src.translation_additivity.cli",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert callable(module.main)
