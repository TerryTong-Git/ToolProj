import pandas as pd
import pytest

from src.exps_mixed_model import data_loader


def test_load_single_benchmark(tmp_path):
    csv_path = tmp_path / "bench.csv"
    pd.DataFrame({"Model": ["Foo"], "Score": [0.5]}).to_csv(csv_path, index=False)
    spec = data_loader.BenchmarkSpec(path=csv_path, score_name="score", model_column="Model", score_column="Score")
    df = data_loader.load_benchmark_file(spec)
    assert list(df.columns) == ["model_id", "score"]
    assert df.iloc[0]["model_id"] == "foo"


def test_merge_benchmarks():
    a = pd.DataFrame({"model_id": ["foo"], "a": [1.0]})
    b = pd.DataFrame({"model_id": ["foo", "bar"], "b": [2.0, 3.0]})
    merged = data_loader.merge_benchmarks([a, b])
    assert set(merged["model_id"]) == {"foo", "bar"}
    assert not merged[merged["model_id"] == "foo"].isna().any(axis=None)


def test_missing_value_handling_mean():
    df = pd.DataFrame({"model_id": ["a", "b"], "x": [1.0, None]})
    filled = data_loader.handle_missing(df, strategy="mean")
    assert filled.loc[filled["model_id"] == "b", "x"].iloc[0] == pytest.approx(1.0)


def test_model_name_normalization_on_merge():
    df1 = pd.DataFrame({"model_id": ["Foo Bar"], "x": [1.0]})
    df1["model_id"] = df1["model_id"].map(data_loader.normalize_model_name)
    df2 = pd.DataFrame({"model_id": ["foo  bar"], "y": [2.0]})
    df2["model_id"] = df2["model_id"].map(data_loader.normalize_model_name)
    merged = data_loader.merge_benchmarks([df1, df2])
    assert len(merged) == 1
    assert merged.iloc[0]["model_id"] == "foo bar"


def test_empty_dataset_error():
    df = pd.DataFrame(columns=["model_id", "reasoning", "nl_instruction", "tool_use", "coding", "task_performance"])
    with pytest.raises(ValueError):
        data_loader.prepare_feature_frame(df, required_columns=df.columns, missing_strategy="drop")
