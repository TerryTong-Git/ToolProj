#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import torch
from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.stats.contingency_tables import mcnemar

csv.field_size_limit(sys.maxsize)


def acc(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def literal_eval(row):
    import ast

    row_dict = ast.literal_eval(row)
    return row_dict


def create_big_df(csv_files: List[Path]):
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


def mcnemar_stat(df, col1: str, col2: str):
    ct1 = pd.crosstab(df[col1], df[col2])
    table1 = sm.stats.Table2x2(ct1)  # when code is executable 1.9 times more likely to observe correct code sim
    return mcnemar(table1.table_orig)


def analyze_contingency_table(df, col1: str, col2: str, name: str):
    ct1 = pd.crosstab(df[col1], df[col2])
    table1 = sm.stats.Table2x2(ct1)  # when code is executable 1.9 times more likely to observe correct code sim
    print(f"We are {table1.oddsratio} more likely to observe {col1} with {col2} than with not {col1}. We have p-value, {table1.oddsratio_pvalue()}")

    df_to_plot = df.copy()
    df_to_plot[col1] = df_to_plot[col1].apply(lambda x: col1 if x == 1 else f"not {col1}")
    df_to_plot[col2] = df_to_plot[col2].apply(lambda x: col2 if x == 1 else f"not {col2}")
    mosaic(df_to_plot, index=[col1, col2], title=f"{col1} v.s. {col2}")
    plt.savefig(f"{name}_mosaic_{col1}_{col2}")
    plt.close()


def get_csv_stats(df: pd.DataFrame, name: str):
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

    # one way anova of the means, two-way friedman test.

    # chi square test actuall fisher exact test, not independent, two-sided test.
    # 'odds ratio.  mcnemar p, two sample t-test, permutation test, unable to retrain so...

    # print(df.describe())

    # with open(csv_file, newline="") as f:
    #     reader = csv.DictReader(f)
    #     for r in reader:
    #         rows.append(r)

    # N = len(rows)
    # correct_nl = [int(r["correct_nl"]) for r in rows]
    # correct_code = [int(r["correct_code"]) for r in rows]
    # acc_nl = acc(correct_nl)
    # acc_code = acc(correct_code)

    # if args.exec_code:
    #     exec_vals = [int(r["correct_code_exec"]) for r in rows if r.get("answer_code_exec", "") != ""]
    #     acc_exec = acc(exec_vals)
    # else:
    #     acc_exec = float("nan")

    # b = sum(1 for r in rows if int(r["correct_code"]) == 1 and int(r["correct_nl"]) == 0)
    # c = sum(1 for r in rows if int(r["correct_code"]) == 0 and int(r["correct_nl"]) == 1)

    # print("=" * 60)
    # print(f"Summary for {args.csv_folder}")
    # print(f"Total N={len(df)}")
    # print(f"Accuracy NL-CoT (overall):   {df['correct_nl'].mean():.4f}")
    # print(f"Accuracy Code-CoT (overall): {df['correct_code'].mean():.4f}")

    # if args.exec_code:
    #     print(f"Execution (overall):        {df['correct_code_exec'].mean():.4f}")

    # print("=" * 60)
    # print()

    # # group by (kind, digits)
    # by_kd = defaultdict(list)
    # for r in rows:
    #     kind = r["kind"]
    #     digits = int(r["digits"])
    #     by_kd[(kind, digits)].append(r)

    # # pretty table
    # results = []
    # kinds = sorted({k for (k, _) in by_kd})
    # for kind in kinds:
    #     print(f"Kind={kind}")
    #     print(f"{'Digits':>8} {'N':>6} {'NL':>8} {'Code':>8} {'Exec':>8}")
    #     print("-" * 42)
    #     nested = []
    #     for d in sorted({d for (k, d) in by_kd if k == kind}):
    #         nested_results = []
    #         grp = by_kd[(kind, d)]
    #         N = len(grp)
    #         acc_nl = acc([int(x["correct_nl"]) for x in grp])
    #         acc_code = acc([int(x["correct_code"]) for x in grp])
    #         if args.exec_code:
    #             exec_vals = [int(x["correct_code_exec"]) for x in grp if x.get("answer_code_exec", "") != ""]
    #             acc_exec = acc(exec_vals)
    #         else:
    #             acc_exec = float("nan")
    #         exec_str = f"{acc_exec:.4f}" if not math.isnan(acc_exec) else "-"
    #         nested_results.append(acc_code)  # ordered
    #         nested_results.append(acc_exec)
    #         nested_results.append(acc_nl)
    #         print(f"{d:>8} {N:>6} {acc_nl:>8.4f} {acc_code:>8.4f} {exec_str:>8}")
    #         nested.append(nested_results)
    #     results.append(nested)
    #     print()
    # return torch.tensor(results)


def main():
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
    # group csv_files by seed
    res = []
    for file in csv_files:
        accs = get_csv_stats(args, file)  # ordered by seed anyway
    for file in csv_files:
        accs = get_csv_stats(args, file)  # ordered by seed anyway
        res.append(accs)
        return

    final = torch.stack(res, axis=0)  # 4D tensor Seed Kind Digit Res
    average_over_seeds = final.mean(dim=0)
    std = final.std(dim=0, unbiased=True)

    total_avg = final.mean(dim=(0, 1, 2))
    std_avg = final.std(dim=(0, 1, 2), unbiased=True)

    print(f"Accuracy NL-CoT (overall):   {total_avg[2]:.4f}     STD: {std_avg[2]:.4f}")

    total_avg = final.mean(dim=(0, 1, 2))
    std_avg = final.std(dim=(0, 1, 2), unbiased=True)

    print(f"Accuracy NL-CoT (overall):   {total_avg[2]:.4f}     STD: {std_avg[2]:.4f}")
    print(f"Accuracy Code-CoT (overall): {total_avg[0]:.4f}     STD: {std_avg[0]:.4f}")
    if args.exec_code:
        print(f"Execution (overall):        {total_avg[1]:.4f}     STD: {std_avg[1]:.4f}")
    rows = []
    with open(csv_files[0], newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    by_kd = defaultdict(list)
    for r in rows:
        dig = int(r["digits"])
        kind = r["kind"]
        by_kd[(dig, kind)].append(r)
    kinds = sorted({k for (_, k) in by_kd})
    digits = sorted({d for (d, _) in by_kd})
    for i, k in enumerate(kinds):
        dig = int(r["digits"])
        kind = r["kind"]
        by_kd[(dig, kind)].append(r)
    kinds = sorted({k for (_, k) in by_kd})
    digits = sorted({d for (d, _) in by_kd})
    for i, k in enumerate(kinds):
        print(f"Kind={k}")
        print(f"{'Digits':>8} {'NL':>8} {'NL_std':>8} {'Code':>8} {'Code_std':>8} {'Exec':>8} {'Exec_std':>8}")
        print("-" * 70)
        for j, d in enumerate(digits):
            print(
                f"{d:>8} {average_over_seeds[i,j,2]:>8.4f} {std[i,j,2]:>8.4f} {average_over_seeds[i,j,0]:>8.4f} \
                {std[i,j,0]:>8.4f} {average_over_seeds[i,j,1]:>8.4f} {std[i,j,1]:>8.4f}"
            )

    for i, k in enumerate(kinds):
        print("-" * 70)
        for j, d in enumerate(digits):
            print(
                f"{d:>8} {average_over_seeds[i,j,2]:>8.4f} {std[i,j,2]:>8.4f} {average_over_seeds[i,j,0]:>8.4f} \
                {std[i,j,0]:>8.4f} {average_over_seeds[i,j,1]:>8.4f} {std[i,j,1]:>8.4f}"
            )

    for i, k in enumerate(kinds):
        print(f"Kind={k}")
        print(f"{'NL':>8} {'NL_std':>8} {'Code':>8} {'Code_std':>8} {'Exec':>8} {'Exec_std':>8}")
        print("-" * 70)
        average_over_digits = final.mean(dim=(0, 2))
        avg_std = final.std(dim=(0, 2), unbiased=True)
        print(
            f"{average_over_digits[i,2]:>8.4f} {avg_std[i,2]:>8.4f} {average_over_digits[i,0]:>8.4f} \
            {avg_std[i,0]:>8.4f} {average_over_digits[i,1]:>8.4f} {avg_std[i,1]:>8.4f}"
        )

        print("-" * 70)
        average_over_digits = final.mean(dim=(0, 2))
        avg_std = final.std(dim=(0, 2), unbiased=True)
        print(
            f"{average_over_digits[i,2]:>8.4f} {avg_std[i,2]:>8.4f} {average_over_digits[i,0]:>8.4f} \
            {avg_std[i,0]:>8.4f} {average_over_digits[i,1]:>8.4f} {avg_std[i,1]:>8.4f}"
        )


if __name__ == "__main__":
    main()
