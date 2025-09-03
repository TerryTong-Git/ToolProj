#!/usr/bin/env python3
import csv, argparse
from collections import defaultdict
import math

def is_executed(v: str) -> bool:
    if v is None: return False
    v = str(v).strip().lower()
    return v != "" and v != "none"

def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to results CSV from cot_general.py (or cot_*.py)")
    args = ap.parse_args()

    # per-kind counters
    total = defaultdict(int)           # all rows of that kind
    n_exec = defaultdict(int)          # rows where subprocess produced an answer
    correct_exec = defaultdict(int)    # correct among executed rows
    correct_total = defaultdict(int)   # correct_exec also counted toward total accuracy

    with open(args.csv_path, newline="") as f:
        reader = csv.DictReader(f)
        needed = {"kind", "answer_code_exec", "correct_code_exec"}
        if not needed.issubset(reader.fieldnames):
            missing = needed - set(reader.fieldnames)
            raise SystemExit(f"CSV missing required columns: {missing}")

        for row in reader:
            k = (row.get("kind") or "").strip()
            if not k:
                # skip unlabeled kind
                continue
            total[k] += 1

            executed = is_executed(row.get("answer_code_exec"))
            corr = safe_int(row.get("correct_code_exec"), 0)
            if executed:
                n_exec[k] += 1
                correct_exec[k] += corr
            # For total subprocess arm accuracy, we count correct_code_exec even when not executed (it’ll be 0).
            correct_total[k] += corr

    # Print table
    header = f"{'kind':14s} {'N_total':>7s} {'N_exec':>7s} {'Exec_Acc_valid':>15s} {'Exec_Acc_total':>15s} {'Coverage':>9s}"
    print(header)
    print("-" * len(header))
    kinds = sorted(total.keys())
    for k in kinds:
        n_tot = total[k]
        n_exe = n_exec[k]
        acc_valid = (correct_exec[k] / n_exe) if n_exe > 0 else float("nan")
        acc_total = (correct_total[k] / n_tot) if n_tot > 0 else float("nan")
        coverage = (n_exe / n_tot) if n_tot > 0 else float("nan")
        def fmt(x): return f"{x:.4f}" if x == x and not math.isinf(x) else "n/a"  # handle NaN/inf
        print(f"{k:14s} {n_tot:7d} {n_exe:7d} {fmt(acc_valid):>15s} {fmt(acc_total):>15s} {fmt(coverage):>9s}")

if __name__ == "__main__":
    main()
