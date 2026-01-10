from typing import List

import pandas as pd

from src.exps_performance.logger import Record


def accuracy(records: List[Record]) -> pd.DataFrame:
    to_df = []
    for r in records:
        to_df.append(r.model_dump())
    df = pd.DataFrame(to_df)

    bykind = df.groupby("kind")
    return bykind[["nl_correct", "code_correct", "sim_correct", "controlsim_correct"]].mean()


def parse_fails(records: List[Record]) -> pd.DataFrame:
    to_df = []
    for r in records:
        to_df.append(r.model_dump())
    df = pd.DataFrame(to_df)
    return df[["nl_correct", "code_correct", "sim_correct", "controlsim_correct"]]


def summary() -> None:
    pass


def accuracy_by_noise(records: List[Record], noise_type: str, sigma: float) -> pd.DataFrame:
    """
    Compute per-kind, per-digit accuracy for each arm with noise metadata attached.
    """
    df = pd.DataFrame([r.model_dump() for r in records])
    arm_map = {
        "nl_correct": "nl",
        "sim_correct": "sim",
        "controlsim_correct": "controlsim",
        "code_correct": "code",
    }
    rows = []
    for col, arm in arm_map.items():
        if col not in df:
            continue
        # Group by both kind and digit
        grouped = df.groupby(["kind", "digit"])[col].mean().reset_index()
        grouped = grouped.rename(columns={col: "accuracy"})
        grouped["arm"] = arm
        grouped["noise_type"] = noise_type
        grouped["sigma"] = sigma
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(columns=["kind", "digit", "arm", "accuracy", "noise_type", "sigma"])
    return pd.concat(rows, ignore_index=True)


# instruction following ability is a confounder...
