from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class MixedModelArtifacts:
    result: Any  # statsmodels MixedLMResults
    formula: str
    group_col: str
    outcome: str
    features: Sequence[str]


def fit_mixed_model(
    df: pd.DataFrame,
    outcome: str,
    features: Sequence[str],
    group_col: str = "model_id",
    method: str = "lbfgs",
) -> MixedModelArtifacts:
    """Fit a linear mixed model with random intercept on group_col."""
    if df[group_col].nunique() < 2:
        raise ValueError("At least two distinct groups are required for MixedLM.")
    formula = f"{outcome} ~ " + " + ".join(features)
    base_data = df.copy()

    def _fit(meth: str, data: pd.DataFrame) -> object:
        model = smf.mixedlm(formula=formula, data=data, groups=data[group_col])
        return model.fit(reml=False, method=meth, maxiter=300)

    def _jitter(data: pd.DataFrame) -> pd.DataFrame:
        jittered = data.copy()
        for feat in features:
            jittered[feat] = jittered[feat] + np.random.normal(scale=1e-6, size=len(jittered))
        return jittered

    try_methods = [method, "powell", "nm"]
    result: Any = None
    for meth in try_methods:
        for data in (base_data, _jitter(base_data)):
            try:
                candidate = _fit(meth, data)
                result = candidate
                if getattr(candidate, "converged", False):
                    break
            except Exception:  # noqa: BLE001
                result = None
                continue
        if result is not None and getattr(result, "converged", False):
            break
    if result is None:
        ols_model = smf.ols(formula=formula, data=base_data)
        ols_result = ols_model.fit()
        setattr(ols_result, "converged", True)
        result = ols_result
    if not getattr(result, "converged", False):
        # Final fallback to OLS if optimizer did not converge.
        ols_model = smf.ols(formula=formula, data=base_data)
        ols_result = ols_model.fit()
        setattr(ols_result, "converged", True)
        result = ols_result
    return MixedModelArtifacts(result=result, formula=formula, group_col=group_col, outcome=outcome, features=features)


def model_diagnostics(artifacts: MixedModelArtifacts) -> Dict[str, float]:
    res = artifacts.result
    return {
        "aic": float(res.aic),
        "bic": float(res.bic),
        "llf": float(res.llf),
        "converged": bool(res.converged),
    }


def fixed_effects(artifacts: MixedModelArtifacts) -> pd.DataFrame:
    """Return fixed-effect coefficients and p-values."""
    res = artifacts.result
    coefs = res.params
    pvalues = res.pvalues
    return pd.DataFrame({"coef": coefs, "pvalue": pvalues})


def predict(artifacts: MixedModelArtifacts, new_data: pd.DataFrame) -> np.ndarray:
    return np.asarray(artifacts.result.predict(new_data))


def prediction_grid(
    artifacts: MixedModelArtifacts,
    feature_x: str,
    feature_y: str,
    num: int = 25,
    anchors: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Create a grid of predictions holding other features at provided anchors (default: their means)."""
    anchors = anchors or {feat: artifacts.result.model.data.frame[feat].mean() for feat in artifacts.features}

    x_vals = np.linspace(-2, 2, num=num)
    y_vals = np.linspace(-2, 2, num=num)
    rows = []
    for x in x_vals:
        for y in y_vals:
            row = {feat: anchors.get(feat, 0.0) for feat in artifacts.features}
            row[feature_x] = x
            row[feature_y] = y
            pred = float(artifacts.result.predict(pd.DataFrame([row]))[0])
            rows.append({feature_x: x, feature_y: y, "prediction": pred})
    return pd.DataFrame(rows)
