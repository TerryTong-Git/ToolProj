import sys
from pathlib import Path

from src.exps_mixed_model import benchmarker, data_loader, main, mixed_model


class DummyClient:
    def __init__(self, scores):
        self.scores = scores

    def evaluate(self, model, tasks):
        return self.scores


def _build_cleaned_dir(tmp_path: Path):
    # Minimal cleaned datasets
    (tmp_path / "livebench.csv").write_text("Model,Coding-Average\nm1,0.5\nm2,0.6\n")
    (tmp_path / "berkeley.csv").write_text("Model,Overall_Acc_Avg_2_3\nm1,0.4\nm2,0.5\n")
    (tmp_path / "aiden.csv").write_text("Model,Accuracy\nm1,0.55\nm2,0.65\n")
    (tmp_path / "mcpmark.csv").write_text("Model,Pass@1\nm1,0.45\nm2,0.55\n")


def test_full_pipeline_synthetic(tmp_path):
    models = ["m1", "m2"]
    scores = {"gsm8k": 0.6, "ifeval": 0.5, "toolbench_function_calling": 0.4, "humaneval": 0.3, "mbpp": 0.2}
    client = DummyClient(scores)
    bench_df = benchmarker.run_benchmarks(models=models, client=client, standardize=False)
    bench_df["task_performance"] = [0.5, 0.6]
    processed = data_loader.prepare_feature_frame(
        bench_df, required_columns=["reasoning", "nl_instruction", "tool_use", "coding", "task_performance"], missing_strategy="drop"
    )
    artifacts = mixed_model.fit_mixed_model(
        processed, outcome="task_performance", features=["reasoning", "nl_instruction", "tool_use", "coding"], group_col="model_id"
    )
    preds = mixed_model.predict(artifacts, processed)
    assert len(preds) == len(processed)


def test_cli_execution_simulated(tmp_path, monkeypatch):
    cleaned_root = tmp_path / "cleaned"
    cleaned_root.mkdir()
    _build_cleaned_dir(cleaned_root)
    task_perf = tmp_path / "task_perf.csv"
    task_perf.write_text("model_id,task_performance\nm1,0.5\nm2,0.6\n")

    out_dir = tmp_path / "out"
    argv = [
        "prog",
        "--models",
        "m1,m2",
        "--generate-benchmarks",
        "--simulate",
        "--task-performance-file",
        str(task_perf),
        "--cleaned-data-root",
        str(cleaned_root),
        "--output-dir",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main.main()
    assert (out_dir / "diagnostics.json").exists()


def test_benchmark_generation_mock(tmp_path):
    models = ["m1"]
    scores = {"gsm8k": 0.6, "ifeval": 0.5, "toolbench_function_calling": 0.4, "humaneval": 0.3, "mbpp": 0.2}
    client = DummyClient(scores)
    df = benchmarker.run_benchmarks(models=models, client=client, standardize=False)
    cache_path = tmp_path / "bench.csv"
    benchmarker.cache_results(df, cache_path)
    loaded = benchmarker.load_cached_results(cache_path)
    assert loaded.equals(df)
