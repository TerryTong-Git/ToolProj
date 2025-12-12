import json

from src.exps_mixed_model import benchmarker
from src.exps_mixed_model.data_loader import normalize_model_name


class DummyClient:
    def __init__(self, scores):
        self.scores = scores

    def evaluate(self, model, tasks):
        return self.scores


class DummySafetyClient:
    def evaluate_safety(self, model, suite):
        return {"reliability": 0.2}


def test_lm_eval_wrapper_smoke(monkeypatch, tmp_path):
    # Simulate parse_lm_eval_results path without running subprocess
    sample = {"results": {"gsm8k": {"acc": 0.5}, "humaneval": {"acc,exact": 0.3}}}
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(sample))
    assert benchmarker.parse_lm_eval_results(results_path) == {"gsm8k": 0.5, "humaneval": 0.3}


def test_run_benchmarks_standardization():
    scores = {"gsm8k": 0.8, "ifeval": 0.7, "toolbench_function_calling": 0.6, "humaneval": 0.5, "mbpp": 0.4}
    client = DummyClient(scores=scores)
    df = benchmarker.run_benchmarks(models=["ModelA", "ModelB"], client=client, standardize=True)
    numeric = df.drop(columns=["model_id"])
    assert numeric.mean().abs().sum() == 0  # standardized mean = 0


def test_aggregate_multi_source_tool_use():
    scores = {"toolbench_function_calling": 0.6, "humaneval": 0.5, "mbpp": 0.3, "gsm8k": 0.7, "ifeval": 0.4}
    client = DummyClient(scores=scores)
    df = benchmarker.run_benchmarks(models=["X"], client=client, standardize=False)
    row = df.iloc[0]
    assert row["tool_use"] == 0.6  # single task
    assert row["coding"] == (0.5 + 0.3) / 2


def test_no_nan_after_processing():
    scores = {"toolbench_function_calling": 0.6, "humaneval": 0.5, "mbpp": 0.3, "gsm8k": 0.7, "ifeval": 0.4}
    client = DummyClient(scores=scores)
    df = benchmarker.run_benchmarks(models=["M1"], client=client, standardize=False)
    assert not df.isna().any(axis=None)


def test_safety_optional():
    scores = {"toolbench_function_calling": 0.6, "humaneval": 0.5, "mbpp": 0.3, "gsm8k": 0.7, "ifeval": 0.4}
    client = DummyClient(scores=scores)
    df = benchmarker.run_benchmarks(
        models=["M1"],
        client=client,
        safety_client=DummySafetyClient(),
        safety_suite=("suite1",),
        standardize=False,
    )
    assert "safety_reliability" in df.columns
    assert df.iloc[0]["model_id"] == normalize_model_name("M1")
