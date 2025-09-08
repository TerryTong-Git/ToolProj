#!/usr/bin/env python3
import io
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_DATA = """kind,digits,N,acc_nl,acc_code,acc_exec
add,2,222,0.981982,1.000000,1.000000
add,4,222,0.945946,0.990991,1.000000
add,8,222,0.603604,0.954955,1.000000
add,16,223,0.632287,0.614350,1.000000
add,32,224,0.035714,0.035714,0.991071
ilp_assign,2,222,0.265766,0.207207,0.000000
ilp_assign,4,222,0.000000,0.000000,0.004505
ilp_assign,8,222,0.022523,0.022523,0.139640
ilp_assign,16,222,0.004505,0.004505,0.031532
ilp_assign,32,222,0.000000,0.000000,0.009009
ilp_partition,2,222,0.387387,0.346847,0.554054
ilp_partition,4,222,0.153153,0.148649,0.454955
ilp_partition,8,222,0.000000,0.004505,0.126126
ilp_partition,16,223,0.000000,0.000000,0.067265
ilp_partition,32,222,0.000000,0.000000,0.049550
ilp_prod,2,222,0.153153,0.166667,0.000000
ilp_prod,4,222,0.009009,0.013514,0.000000
ilp_prod,8,222,0.009009,0.027027,0.000000
ilp_prod,16,222,0.004505,0.013514,0.000000
ilp_prod,32,223,0.000000,0.008969,0.004484
knap,2,222,0.148649,0.162162,0.094595
knap,4,223,0.004484,0.107623,0.183857
knap,8,222,0.000000,0.000000,0.130631
knap,16,222,0.000000,0.000000,0.031532
knap,32,222,0.000000,0.000000,0.054054
lcs,2,223,0.874439,0.668161,0.246637
lcs,4,222,0.463964,0.396396,0.945946
lcs,8,222,0.391892,0.400901,1.000000
lcs,16,222,0.022523,0.148649,1.000000
lcs,32,222,0.000000,0.148649,0.995495
mul,2,223,0.878924,0.968610,1.000000
mul,4,223,0.022422,0.008969,0.973094
mul,8,222,0.000000,0.000000,0.986486
mul,16,222,0.000000,0.000000,0.990991
mul,32,222,0.000000,0.000000,1.000000
rod,2,222,0.680180,0.382883,0.941441
rod,4,222,0.000000,0.027027,0.558559
rod,8,222,0.009009,0.009009,0.545045
rod,16,222,0.000000,0.000000,0.950450
rod,32,222,0.000000,0.000000,0.891892
sub,2,222,1.000000,1.000000,1.000000
sub,4,222,0.887387,0.995495,1.000000
sub,8,222,0.905405,0.927928,1.000000
sub,16,222,0.283784,0.463964,1.000000
sub,32,223,0.022422,0.026906,0.991031
"""

def read_df(path: str | None) -> pd.DataFrame:
    if path:
        return pd.read_csv(path)
    return pd.read_csv(io.StringIO(DEFAULT_DATA))

def slopegrid(df: pd.DataFrame, out_path="slopegrid_acc.png"):
    kinds = sorted(df["kind"].unique())
    digits = sorted(df["digits"].unique())
    nrows, ncols = len(kinds), len(digits)
    # Make a reasonably sized grid
    fig_w = max(10, ncols * 2.2)
    fig_h = max(8, nrows * 1.3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    x_pos = np.array([0, 1, 2])  # NL, Code, Exec
    labels = ["NL", "Code", "Exec"]
    for i, kind in enumerate(kinds):
        for j, d in enumerate(digits):
            ax = axes[i, j]
            row = df[(df.kind == kind) & (df.digits == d)]
            if row.empty:
                ax.axis("off")
                continue
            y = [float(row["acc_nl"]), float(row["acc_code"]), float(row["acc_exec"])]
            # thin gray baseline for the “shape”
            ax.plot(x_pos, y, color="#555555", lw=1.5, marker="o", ms=3)
            # highlight the Exec jump
            ax.plot(x_pos[1:], y[1:], color="#1f77b4", lw=2.0, marker="o", ms=4)

            ax.set_xticks(x_pos, labels if i == nrows - 1 else ["", "", ""])
            if j == 0:
                ax.set_ylabel(kind)
            ax.set_ylim(0.0, 1.05)
            ax.set_title(f"d={d}", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.25)
    fig.suptitle("Accuracy per (kind, digits) with NL → Code → Exec slope", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")

def delta_bars(df: pd.DataFrame, out_path="delta_bars.png"):
    # Weighted deltas across digits per kind
    df2 = df.copy()
    df2["d_code_nl"] = df2["acc_code"] - df2["acc_nl"]
    df2["d_exec_code"] = df2["acc_exec"] - df2["acc_code"]

    rows = []
    for kind, grp in df2.groupby("kind"):
        w = grp["N"].to_numpy()
        w = w / w.sum()
        d1 = float((w * grp["d_code_nl"].to_numpy()).sum())
        d2 = float((w * grp["d_exec_code"].to_numpy()).sum())
        rows.append({"kind": kind, "delta": "Code − NL", "value": d1})
        rows.append({"kind": kind, "delta": "Exec − Code", "value": d2})
    agg = pd.DataFrame(rows)

    kinds = sorted(agg["kind"].unique())
    x = np.arange(len(kinds))
    width = 0.38

    # two side-by-side bars per kind
    fig, ax = plt.subplots(figsize=(max(10, len(kinds)*1.2), 5))
    vals_d1 = [agg[(agg.kind==k)&(agg.delta=="Code − NL")]["value"].item() for k in kinds]
    vals_d2 = [agg[(agg.kind==k)&(agg.delta=="Exec − Code")]["value"].item() for k in kinds]

    b1 = ax.bar(x - width/2, vals_d1, width, label="Code − NL")
    b2 = ax.bar(x + width/2, vals_d2, width, label="Exec − Code")

    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, kinds, rotation=30, ha="right")
    ax.set_ylabel("Δ Accuracy (weighted by N)")
    ax.set_title("Method deltas per kind (weighted across digits)")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV path; otherwise uses embedded data")
    ap.add_argument("--out1", type=str, default="slopegrid_acc.png")
    ap.add_argument("--out2", type=str, default="delta_bars.png")
    args = ap.parse_args()

    df = read_df(args.csv)
    # Clean & types
    df["digits"] = df["digits"].astype(int)
    df["N"] = df["N"].astype(int)
    for c in ["acc_nl","acc_code","acc_exec"]:
        df[c] = df[c].astype(float)

    slopegrid(df, out_path=args.out1)
    delta_bars(df, out_path=args.out2)

if __name__ == "__main__":
    main()
