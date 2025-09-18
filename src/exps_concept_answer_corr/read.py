#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import math

def phi_from_counts(a,b,c,d):
    den = math.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    return 0.0 if den == 0 else (a*d - b*c)/den

def phi_binary(x, y):
    x = np.asarray(x, dtype=int).ravel()
    y = np.asarray(y, dtype=int).ravel()
    a = int(((x==1)&(y==1)).sum())  # TP
    b = int(((x==1)&(y==0)).sum())  # FP
    c = int(((x==0)&(y==1)).sum())  # FN
    d = int(((x==0)&(y==0)).sum())  # TN
    return phi_from_counts(a,b,c,d), (a,b,c,d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with concept, answer_correct, concept_top1_correct")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Overall φ
    phi_overall, (a,b,c,d) = phi_binary(df["concept_top1_correct"], df["answer_correct"])
    print("Overall hard–hard correlation:")
    print(f"  φ = {phi_overall:.4f}")
    print(f"  Confusion table: a(TP)={a}, b(FP)={b}, c(FN)={c}, d(TN)={d}")

    # Per-kind φ
    print("\nPer-concept φ:")
    for kind, g in df.groupby("concept"):
        if g["answer_correct"].nunique() < 2 or g["concept_top1_correct"].nunique() < 2:
            if g["concept_top1_correct"].nunique() < 2:
                print("concept same")
            if g["answer_correct"].nunique() < 2:
                print("answer same")
            print(f"  {kind:>14s}  N={len(g):4d}  φ=nan (degenerate labels)")
            continue
        phi_k, (ak,bk,ck,dk) = phi_binary(g["concept_top1_correct"], g["answer_correct"])
        print(f"  {kind:>14s}  N={len(g):4d}  φ={phi_k: .4f}  [a={ak}, b={bk}, c={ck}, d={dk}]")

if __name__ == "__main__":
    main()
