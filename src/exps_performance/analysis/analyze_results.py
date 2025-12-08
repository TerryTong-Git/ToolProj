#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.stats.contingency_tables import mcnemar

csv.field_size_limit(sys.maxsize)


def acc(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def literal_eval(row: str) -> Any:
    import ast

    row_dict = ast.literal_eval(row)
    return row_dict


def create_big_df(csv_files: List[Path]) -> pd.DataFrame:
    pattern = r"seed_(\d)"
    big_df = []
    for csv_file in csv_files:
        seed = re.search(pattern, csv_file.name)
        if seed is not None:
            parsed_seed = seed.group(1)
            df = pd.read_csv(csv_file)
            df["seed"] = int(parsed_seed)
            big_df.append(df)
    return pd.concat(big_df, axis=0, ignore_index=True)


# add mcnemar t-test between nl and code exec and nl and code run to see difference


def mcnemar_stat(df: pd.DataFrame, col1: str, col2: str) -> Any:
    ct1 = pd.crosstab(df[col1], df[col2])
    table1 = sm.stats.Table2x2(ct1)  # when code is executable 1.9 times more likely to observe correct code sim
    return mcnemar(table1.table_orig)


def analyze_contingency_table(df: pd.DataFrame, col1: str, col2: str, name: str) -> None:
    ct1 = pd.crosstab(df[col1], df[col2])
    table1 = sm.stats.Table2x2(ct1)  # when code is executable 1.9 times more likely to observe correct code sim
    print(f"We are {table1.oddsratio} more likely to observe {col1} with {col2} than with not {col1}. We have p-value, {table1.oddsratio_pvalue()}")

    df_to_plot = df.copy()
    df_to_plot[col1] = df_to_plot[col1].apply(lambda x: col1 if x == 1 else f"not {col1}")
    df_to_plot[col2] = df_to_plot[col2].apply(lambda x: col2 if x == 1 else f"not {col2}")
    mosaic(df_to_plot, index=[col1, col2], title=f"{col1} v.s. {col2}")
    plt.savefig(f"{name}_mosaic_{col1}_{col2}")
    plt.close()


def plot_df(summary: Any, df: pd.DataFrame) -> None:
    import pdb

    import plotly.graph_objects as go

    pdb.set_trace()
    x = df["digits"]
    y = df["mean"]
    df.groupby()

    # confidence_interval =
    go.Figure(go.Scatter(x=x, y=y, mode="lines", name="Mean", line=dict(color="blue")))
    plt.show()


def get_csv_stats(df: pd.DataFrame, name: str) -> None:
    """
    Get the mean and the variance
    """

    df["answer_code_exec"] = df["answer_code_exec"].apply(literal_eval)
    df["correct_code_sim"] = df["correct_code"]
    df["executable_code"] = df["answer_code_exec"].apply(lambda x: int(x["ok"]))

    analyze_contingency_table(df, "executable_code", "correct_code_sim", name)
    analyze_contingency_table(df, "executable_code", "correct_code_exec", name)
    analyze_contingency_table(df, "correct_code_exec", "correct_code_sim", name)

    digits = df[["digits", "executable_code", "correct_code_sim", "correct_code_exec"]].groupby("digits")
    seed = df[["seed", "executable_code", "correct_code_sim", "correct_code_exec"]].groupby("seed")
    digit_summary = digits.describe()[
        [
            ("executable_code", "mean"),
            ("executable_code", "std"),
            ("correct_code_exec", "mean"),
            ("correct_code_exec", "std"),
            ("correct_code_sim", "mean"),
            ("correct_code_sim", "std"),
        ]
    ]
    another = df[["kind", "executable_code", "correct_code_sim", "correct_code_exec", "seed"]].groupby(["seed", "kind"]).describe().reset_index()
    import pdb

    pdb.set_trace()
    another[[("kind", ""), ("seed", ""), ("executable_code", "mean"), ("correct_code_exec", "mean"), ("correct_code_sim", "mean")]].boxplot(
        by=("kind", "")
    )
    plt.savefig("boxplot")
    plt.close()  # plot boxplots w/ standard deviations across the seeds.

    print(digit_summary.to_latex())
    # with open('digit_summary.txt', 'w+') as f:
    #     f.write(digit_summary.to_latex())

    seed_summary = seed.describe()[
        [
            ("executable_code", "mean"),
            ("executable_code", "std"),
            ("correct_code_exec", "mean"),
            ("correct_code_exec", "std"),
            ("correct_code_sim", "mean"),
            ("correct_code_sim", "std"),
        ]
    ]
    print(seed_summary.to_latex())

    # with open('seed_summary.txt', 'w+') as f:
    #     f.write(seed_summary.to_latex())

    from scipy.stats import friedmanchisquare

    stat, p_val = friedmanchisquare(df["executable_code"], df["correct_code_sim"], df["correct_code_exec"])
    print(f"friedman statistic {stat}, p-value {p_val} ")

    print("Correct_nl and correct code sim mcnemar statistic")
    print(mcnemar_stat(df, "correct_nl", "correct_code_sim"))

    print("Correct_nl and executable code mcnemar statistic")
    print(mcnemar_stat(df, "correct_nl", "executable_code"))

    print("Correct_nl and correct code exec mcnemar statistic")
    print(mcnemar_stat(df, "correct_nl", "correct_code_exec"))

    print(f"Overall Code Exec Correct mean: {df['correct_code_exec'].describe()[['mean', 'std']]}")
    print(f"Overall Code Sim: {df['correct_code_sim'].describe()[['mean', 'std']]}")
    print(f"Overall Exeuctable Code Sim: {df['executable_code'].describe()[['mean', 'std']]}")
    print(f"Overall NL: {df['correct_nl'].describe()[['mean', 'std']]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-folder", help="Path to csv folder (the row-level file)")
    ap.add_argument(
        "--exec-code",
        action="store_true",
        help="If set, also compute execution accuracy where available",
    )

    # regex parse the seed.
    args = ap.parse_args()
    name = args.csv_folder.split("_")[-1]
    csv_folder = Path(os.path.join(Path(__name__).parent, args.csv_folder))
    csv_files = [f for f in csv_folder.iterdir() if (f.is_file() and "results" in f.name)]
    df = create_big_df(csv_files)
    get_csv_stats(df, name)
    return


if __name__ == "__main__":
    main()
