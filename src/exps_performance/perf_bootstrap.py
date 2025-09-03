# perf_bootstrap.py
import argparse, pandas as pd, numpy as np, json

def acc(x):
    return float(np.mean(np.asarray(x, dtype=float))) if len(x)>0 else float("nan")

def percentile_ci(samples, level=0.95):
    s = np.asarray(samples, dtype=float); s = s[np.isfinite(s)]
    if s.size==0: return float("nan"), float("nan")
    lo = (1-level)/2; hi = 1-lo
    return float(np.quantile(s, lo)), float(np.quantile(s, hi))

def bootstrap_accs(dfk, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    idxs = dfk["idx"].unique().tolist()
    exec_mask = dfk["answer_code_exec"].notna().values
    y_exec = dfk["correct_code_exec"].fillna(0).astype(int).values
    y_arm  = dfk["correct_code_exec"].fillna(0).astype(int).values  # non-exec already 0
    # masks per-row; we need grouped resampling by idx
    rows_by_idx = {i: dfk[dfk["idx"]==i] for i in idxs}

    # point estimates
    exec_rows = dfk[ dfk["answer_code_exec"].notna() ]
    acc_exec_pt = acc(exec_rows["correct_code_exec"])  # mean over valid execs
    acc_arm_pt  = acc(dfk["correct_code_exec"].fillna(0))  # mean over all

    # bootstrap
    boot_exec, boot_arm = [], []
    for _ in range(B):
        samp = [rows_by_idx[i] for i in rng.choice(idxs, size=len(idxs), replace=True)]
        dfb = pd.concat(samp, ignore_index=True)

        exec_rows_b = dfb[dfb["answer_code_exec"].notna()]
        acc_exec_b = acc(exec_rows_b["correct_code_exec"])
        acc_arm_b  = acc(dfb["correct_code_exec"].fillna(0))

        boot_exec.append(acc_exec_b)
        boot_arm.append(acc_arm_b)

    e_lo,e_hi = percentile_ci(boot_exec, 0.95)
    a_lo,a_hi = percentile_ci(boot_arm,  0.95)

    return dict(
        acc_exec=float(acc_exec_pt), acc_exec_std=float(np.nanstd(boot_exec)), acc_exec_CI95=[e_lo,e_hi],
        acc_arm=float(acc_arm_pt),   acc_arm_std=float(np.nanstd(boot_arm)),   acc_arm_CI95=[a_lo,a_hi]
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.results_csv)
    out = {"overall": {}, "by_kind": {}}

    # overall
    out["overall"] = bootstrap_accs(df, B=args.B)

    # per kind
    for k, dfk in df.groupby("kind"):
        out["by_kind"][k] = bootstrap_accs(dfk, B=args.B)

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out_json}")

if __name__ == "__main__":
    main()
