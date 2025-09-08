# mi_bootstrap.py
import json, math, argparse, numpy as np, pandas as pd
from typing import List, Dict

def _logsumexp(a, axis=None):
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64); s = v.sum()
    if not np.isfinite(s) or s <= 0: return np.full_like(v, 1.0/max(1,v.size), dtype=np.float64)
    return v / (s + eps)

def _pz_given_theta_logweights(df_b_th: pd.DataFrame, support: List[str], col: str, alpha=1e-3) -> np.ndarray:
    per_seq_logs: Dict[str, List[float]] = {}
    for r in df_b_th.itertuples(index=False):
        per_seq_logs.setdefault(r.seq_id, []).append(float(getattr(r, col)))
    logvec = np.full(len(support), -np.inf, dtype=np.float64)
    idx = {sid:i for i,sid in enumerate(support)}
    for sid, lst in per_seq_logs.items():
        if sid in idx: logvec[idx[sid]] = _logsumexp(np.array(lst, dtype=np.float64))
    m = np.max(logvec)
    exps = np.exp(logvec - m, where=np.isfinite(logvec))
    exps = np.where(np.isfinite(exps), exps, 0.0)
    exps += float(alpha)
    return _normalize(exps)

def _mi_for_bucket(df_b: pd.DataFrame, thetas_all: List[str], col: str, alpha=1e-3, prior="uniform"):
    support = sorted(df_b["seq_id"].unique().tolist())
    present = sorted(df_b["theta"].unique().tolist())
    valid = [t for t in thetas_all if t in present]
    if len(support)==0 or len(valid)<2:
        return float("nan"), float("nan"), float("nan")
    P = []; counts = []
    for th in valid:
        df_th = df_b[df_b["theta"]==th]
        P.append(_pz_given_theta_logweights(df_th, support, col, alpha=alpha))
        counts.append(len(df_th))
    P = np.stack(P, axis=0)
    if prior=="weighted":
        pth = _normalize(np.asarray(counts, dtype=np.float64))
    else:
        pth = np.full(len(valid), 1.0/len(valid), dtype=np.float64)
    pz = _normalize((pth[:,None]*P).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(P, pz[None,:], out=np.ones_like(P), where=(pz[None,:]>0))
        terms = pth[:,None]*P*np.log(np.clip(ratio, 1e-300, None), dtype=np.float64)
        mi_nats = float(terms.sum())
    H_theta = -float((pth*np.log(np.clip(pth,1e-300,None), dtype=np.float64)).sum())
    log2 = math.log(2.0)
    return mi_nats/log2, H_theta/log2, (H_theta-mi_nats)/log2

def percentile_ci(samples: np.ndarray, level=0.95):
    lo = (1-level)/2; hi = 1-lo
    return float(np.quantile(samples, lo)), float(np.quantile(samples, hi))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_rows", required=True, help="Parquet/CSV with per-sample scores")
    ap.add_argument("--score_col", choices=["sum_logp_cont","sum_logp_joint"], default="sum_logp_cont")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--theta_prior", choices=["uniform","weighted"], default="uniform")
    ap.add_argument("--B", type=int, default=1000, help="bootstrap draws")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.scored_rows) if args.scored_rows.endswith(".parquet") else pd.read_csv(args.scored_rows)

    out = []
    for (model, rep, bucket), df_mrb in df.groupby(["model","rep","bucket"]):
        thetas_all = sorted(df_mrb["theta"].unique().tolist())
        # point estimate
        mi_pt, H_pt, Hc_pt = _mi_for_bucket(df_mrb, thetas_all, args.score_col, alpha=args.alpha, prior=args.theta_prior)

        # cluster IDs for bootstrap: resample problem indices within (theta,bucket)
        # fall back to per-row if no 'idx' column
        has_idx = "idx" in df_mrb.columns
        boot_vals = []
        rng = np.random.default_rng(0)
        for _ in range(args.B):
            parts = []
            for th, df_th in df_mrb.groupby("theta"):
                if has_idx:
                    keys = df_th["idx"].unique().tolist()
                    resampled = rng.choice(keys, size=len(keys), replace=True)
                    df_th_b = pd.concat([df_th[df_th["idx"]==k] for k in resampled], ignore_index=True)
                else:
                    resampled = rng.choice(len(df_th), size=len(df_th), replace=True)
                    df_th_b = df_th.iloc[resampled].reset_index(drop=True)
                parts.append(df_th_b)
            df_b = pd.concat(parts, ignore_index=True)
            mi_b, _, _ = _mi_for_bucket(df_b, thetas_all, args.score_col, alpha=args.alpha, prior=args.theta_prior)
            boot_vals.append(mi_b)
        boot_vals = np.array(boot_vals, dtype=float)
        lo, hi = percentile_ci(boot_vals[np.isfinite(boot_vals)], level=0.95) if np.isfinite(boot_vals).any() else (float("nan"), float("nan"))
        out.append({
            "model": model, "rep": rep, "bucket": bucket,
            "score_col": args.score_col,
            "MI_bits": float(mi_pt), "MI_boot_mean": float(np.nanmean(boot_vals)),
            "MI_boot_std": float(np.nanstd(boot_vals)),
            "MI_CI95": [lo, hi],
            "B": int(args.B)
        })

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out_json}")

if __name__ == "__main__":
    main()
