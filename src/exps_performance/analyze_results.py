#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import torch


def acc(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def get_csv_stats(args, csv_file):
    """
    Get the mean and the variance
    """
    rows = []
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    N = len(rows)
    correct_nl = [int(r["correct_nl"]) for r in rows]
    correct_code = [int(r["correct_code"]) for r in rows]
    acc_nl = acc(correct_nl)
    acc_code = acc(correct_code)

    if args.exec_code:
        exec_vals = [int(r["correct_code_exec"]) for r in rows if r.get("answer_code_exec", "") != ""]
        acc_exec = acc(exec_vals)
    else:
        acc_exec = float("nan")

    b = sum(1 for r in rows if int(r["correct_code"]) == 1 and int(r["correct_nl"]) == 0)
    c = sum(1 for r in rows if int(r["correct_code"]) == 0 and int(r["correct_nl"]) == 1)

    print("=" * 60)
    print(f"Summary for {args.csv_folder}")
    print(f"Total N={N}")
    print(f"Accuracy NL-CoT (overall):   {acc_nl:.4f}")
    print(f"Accuracy Code-CoT (overall): {acc_code:.4f}")
    if args.exec_code:
        print(f"Execution (overall):        {acc_exec:.4f}")
    print(f"Discordant pairs: b=code>nl={b}, c=nl>code={c}")
    print("=" * 60)
    print()

    # group by (kind, digits)
    by_kd = defaultdict(list)
    for r in rows:
        kind = r["kind"]
        digits = int(r["digits"])
        by_kd[(kind, digits)].append(r)

    # pretty table
    results = []
    kinds = sorted({k for (k, _) in by_kd})
    for kind in kinds:
        print(f"Kind={kind}")
        print(f"{'Digits':>8} {'N':>6} {'NL':>8} {'Code':>8} {'Exec':>8}")
        print("-" * 42)
        nested = []
        for d in sorted({d for (k, d) in by_kd if k == kind}):
            nested_results = []
            grp = by_kd[(kind, d)]
            N = len(grp)
            acc_nl = acc([int(x["correct_nl"]) for x in grp])
            acc_code = acc([int(x["correct_code"]) for x in grp])
            if args.exec_code:
                exec_vals = [int(x["correct_code_exec"]) for x in grp if x.get("answer_code_exec", "") != ""]
                acc_exec = acc(exec_vals)
            else:
                acc_exec = float("nan")
            exec_str = f"{acc_exec:.4f}" if not math.isnan(acc_exec) else "-"
            nested_results.append(acc_code)  # ordered
            nested_results.append(acc_exec)
            nested_results.append(acc_nl)
            print(f"{d:>8} {N:>6} {acc_nl:>8.4f} {acc_code:>8.4f} {exec_str:>8}")
            nested.append(nested_results)
        results.append(nested)
        print()
    return torch.tensor(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-folder", help="Path to csv folder (the row-level file)")
    ap.add_argument(
        "--exec-code",
        action="store_true",
        help="If set, also compute execution accuracy where available",
    )
    args = ap.parse_args()
    csv_folder = Path(os.path.join(Path(__name__).parent, args.csv_folder))
    csv_files = [f for f in csv_folder.iterdir() if (f.is_file() and "results" in f.name)]

    # group csv_files by seed
    res = []
    for file in csv_files:
        accs = get_csv_stats(args, file)  # ordered by seed anyway
        res.append(accs)
    final = torch.stack(res, axis=0)  # 4D tensor Seed Kind Digit Res
    average_over_seeds = final.mean(dim=0)
    std = final.std(dim=0, unbiased=True)

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
        print(f"Kind={k}")
        print(f"{'Digits':>8} {'NL':>8} {'NL_std':>8} {'Code':>8} {'Code_std':>8} {'Exec':>8} {'Exec_std':>8}")
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


if __name__ == "__main__":
    main()
