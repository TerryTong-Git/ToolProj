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


def parse_fails(records: List[Record]):
    to_df = []
    for r in records:
        to_df.append(r.model_dump())
    df = pd.DataFrame(to_df)
    df[["nl_correct", "code_correct", "sim_correct", "controlsim_correct"]]


def summary():
    pass


# instruction following ability is a confounder...
