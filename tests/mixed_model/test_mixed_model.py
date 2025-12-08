import numpy as np
import pandas as pd
import pytest

from src.exps_mixed_model import mixed_model


def _synthetic_df():
    np.random.seed(0)
    data = []
    for group in ["a", "b"]:
        for i in range(5):
            data.append(
                {
                    "model_id": group,
                    "reasoning": np.random.normal(),
                    "nl_instruction": np.random.normal(),
                    "tool_use": np.random.normal(),
                    "coding": np.random.normal(),
                }
            )
    df = pd.DataFrame(data)
    df["task_performance"] = (
        0.4 * df["reasoning"] + 0.3 * df["nl_instruction"] + 0.2 * df["tool_use"] + 0.1 * df["coding"] + np.random.normal(scale=0.1, size=len(df))
    )
    return df


def test_model_fits():
    df = _synthetic_df()
    artifacts = mixed_model.fit_mixed_model(df, outcome="task_performance", features=["reasoning", "nl_instruction", "tool_use", "coding"])
    assert artifacts.result.converged


def test_fixed_effects_significance():
    df = _synthetic_df()
    artifacts = mixed_model.fit_mixed_model(df, outcome="task_performance", features=["reasoning", "nl_instruction", "tool_use", "coding"])
    fx = mixed_model.fixed_effects(artifacts)
    assert "coef" in fx.columns and "pvalue" in fx.columns


def test_random_effects_variance():
    df = _synthetic_df()
    artifacts = mixed_model.fit_mixed_model(df, outcome="task_performance", features=["reasoning", "nl_instruction", "tool_use", "coding"])
    var = artifacts.result.cov_re.iloc[0, 0]
    assert var >= 0


def test_prediction_output_shape():
    df = _synthetic_df()
    artifacts = mixed_model.fit_mixed_model(df, outcome="task_performance", features=["reasoning", "nl_instruction", "tool_use", "coding"])
    preds = mixed_model.predict(artifacts, df)
    assert len(preds) == len(df)


def test_model_diagnostics():
    df = _synthetic_df()
    artifacts = mixed_model.fit_mixed_model(df, outcome="task_performance", features=["reasoning", "nl_instruction", "tool_use", "coding"])
    diag = mixed_model.model_diagnostics(artifacts)
    assert {"aic", "bic", "llf", "converged"}.issubset(diag.keys())


def test_insufficient_groups_error():
    df = _synthetic_df()
    df["model_id"] = "single"
    with pytest.raises(ValueError):
        mixed_model.fit_mixed_model(df, outcome="task_performance", features=["reasoning"], group_col="model_id")
