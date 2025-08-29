#!/usr/bin/env python3
import os, argparse, math, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   # pretty heatmaps

from cox.store import Store

# ---------------- utils ----------------
EPS = 1e-40

def _safe_log(x):
    return np.log(np.clip(x, EPS, 1.0))

def build_discrete_pz_given_theta(df_ab_theta, support, use_weights=True, alpha=1e-3):
    """
    Return length-|support| vector p(z|theta,x,r) over the provided support.
    * Stable weighting: log-sum-exp aggregation of sum_logp_joint per seq_id,
      then subtract max before exponentiation (prevents underflow).
    * If use_weights=False, falls back to simple counts.
    * Dirichlet smoothing by alpha.
    """
    idx = {sid: i for i, sid in enumerate(support)}
    Z = len(support)

    if use_weights:
        # log-weights init at -inf (log 0)
        logw = np.full(Z, -np.inf, dtype=np.float64)
        for r in df_ab_theta.itertuples():
            i = idx[r.seq_id]
            # sum over duplicates in log-space
            logw[i] = np.logaddexp(logw[i], float(r.sum_logp_joint))
        if not np.isfinite(logw).any():
            # no usable mass -> uniform
            vec = np.full(Z, 1.0 / Z, dtype=np.float64)
            return vec
        m = np.max(logw)
        w = np.exp(logw - m)  # safe exponentiation
    else:
        w = np.zeros(Z, dtype=np.float64)
        for r in df_ab_theta.itertuples():
            w[idx[r.seq_id]] += 1.0

    # Dirichlet smoothing & normalize
    vec = w + float(alpha)
    s = vec.sum()
    if s <= 0:
        vec[:] = 1.0 / Z
    else:
        vec /= s
    return vec

def kl_divergence(p, q):
    """KL(p||q) with clipping for numerical safety."""
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return float(np.sum(p * (_safe_log(p) - _safe_log(q))))

def symmetric_kl(p, q):
    return 0.5 * (kl_divergence(p, q) + kl_divergence(q, p))

def js_divergence(p, q):
    m = 0.5*(p+q)
    return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

def build_theta_dists_for_x(df_x, thetas, use_weights=True, alpha=1e-3, min_support=2):
    """
    Build dict(theta -> p(z|theta,x)) over a *pooled* support across thetas for this x.
    Skips cases with too-small support.
    """
    support = sorted(df_x["seq_id"].unique().tolist())
    if len(support) < int(min_support):
        return {}, support  # not enough variation to compare

    out = {}
    for th in thetas:
        dth = df_x[df_x["theta"] == th]
        if len(dth) == 0:
            continue
        out[th] = build_discrete_pz_given_theta(dth, support, use_weights=use_weights, alpha=alpha)
    return out, support

def pairwise_matrix(theta_to_vec, metric_fn):
    ths = sorted(theta_to_vec.keys())
    n = len(ths)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        pi = theta_to_vec[ths[i]]
        for j in range(n):
            if i == j:
                M[i, j] = 0.0
            else:
                pj = theta_to_vec[ths[j]]
                # ensure same support length
                if pi.shape != pj.shape:
                    raise ValueError("Vectors have mismatched shapes; support must be aligned.")
                M[i, j] = metric_fn(pi, pj)
    return ths, M

def heatmap(M, labels, title, out_png):
    plt.figure(figsize=(5 + 0.3*len(labels), 4 + 0.3*len(labels)))
    ax = sns.heatmap(M, xticklabels=labels, yticklabels=labels, annot=True, fmt=".3f", square=True, cmap="mako")
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, type=str, help="runner.py output dir (Cox store root)")
    ap.add_argument("--exp_id", default=None, type=str)
    ap.add_argument("--alpha", default=1e-3, type=float)
    ap.add_argument("--use_weighted", action="store_true", help="weight sequences by exp(logp) (log-sum-exp stabilized)")
    ap.add_argument("--metric", choices=["kl", "skl", "jsd"], default="skl",
                    help="pairwise divergence to visualize")
    ap.add_argument("--per_input", action="store_true",
                    help="emit a heatmap per input (a,b) as well as aggregate")
    ap.add_argument("--aggregate", action="store_true",
                    help="emit an aggregate heatmap over inputs (avg of pairwise mats)")
    ap.add_argument("--save_json", action="store_true",
                    help="dump numeric divergence matrices to JSON too")
    ap.add_argument("--min_support", type=int, default=2,
                    help="minimum # of distinct seq_ids required for an input to be considered")
    ap.add_argument("--aggregate_require_all", action="store_true",
                    help="only aggregate inputs where *all* thetas are present")
    args = ap.parse_args()

    # load store
    store = Store(args.out_dir, exp_id=args.exp_id)
    if "samples" not in store.tables:
        raise RuntimeError("No 'samples' table found. Run runner.py first (preferably with --paired_inputs).")
    df = store["samples"].df.copy()

    # sanity: list models/reps/thetas
    models = sorted(df["model_name"].unique().tolist())
    reps   = sorted(df["rep"].unique().tolist())
    thetas_all = sorted(df["theta"].unique().tolist())
    print(f"[viz] models={models} reps={reps} thetas={thetas_all}")

    metric_fn = {"kl": kl_divergence, "skl": symmetric_kl, "jsd": js_divergence}[args.metric]
    out_root = os.path.join(args.out_dir, "viz_kl")
    os.makedirs(out_root, exist_ok=True)

    # loop per (model, rep)
    for model_name in models:
        for rep in reps:
            df_mr = df[(df["model_name"] == model_name) & (df["rep"] == rep)]
            if len(df_mr) == 0:
                continue
            thetas = sorted(df_mr["theta"].unique().tolist())
            print(f"[viz] {model_name} | {rep} | thetas={thetas}")

            # ---- per-input heatmaps ----
            if args.per_input:
                for (aa, bb), df_x in df_mr.groupby(["a", "b"]):
                    theta_vecs, support = build_theta_dists_for_x(
                        df_x, thetas, use_weights=args.use_weighted, alpha=args.alpha, min_support=args.min_support
                    )
                    if len(support) < args.min_support:
                        continue
                    if len(theta_vecs) < 2:
                        continue
                    labels, M = pairwise_matrix(theta_vecs, metric_fn)
                    title = f"{model_name} | {rep} | (a={aa}, b={bb}) | {args.metric.upper()}"
                    out_png = os.path.join(
                        out_root, f"per_input__{model_name.replace('/','_')}__{rep}__a{aa}_b{bb}__{args.metric}.png"
                    )
                    heatmap(M, labels, title, out_png)
                    print(f"[viz] wrote {out_png}")
                    if args.save_json:
                        with open(out_png.replace(".png", ".json"), "w") as f:
                            json.dump({"labels": labels, "matrix": M.tolist()}, f, indent=2)

            # ---- aggregate across inputs ----
            if args.aggregate:
                mats = []
                labels_ref = None
                kept = 0
                for (aa, bb), df_x in df_mr.groupby(["a", "b"]):
                    theta_vecs, support = build_theta_dists_for_x(
                        df_x, thetas, use_weights=args.use_weighted, alpha=args.alpha, min_support=args.min_support
                    )
                    if len(support) < args.min_support or len(theta_vecs) < 2:
                        continue
                    # Optionally require *all* thetas to be present for consistent aggregation
                    if args.aggregate_require_all and set(theta_vecs.keys()) != set(thetas):
                        continue

                    labels, M = pairwise_matrix(theta_vecs, metric_fn)

                    # align label order across inputs
                    if labels_ref is None:
                        labels_ref = labels
                    if labels != labels_ref:
                        # Only aggregate if the label sets match exactly; else skip to keep shapes consistent.
                        if set(labels) != set(labels_ref):
                            # First time we see a *different* label set, reset to this superset once.
                            if set(labels_ref) < set(labels):
                                labels_ref = labels
                                mats = []  # restart aggregation with new reference
                            else:
                                # skip this input (incomplete labels) to avoid NaN padding headaches
                                continue
                        # Reindex rows/cols into labels_ref order
                        mp = {lab: i for i, lab in enumerate(labels)}
                        idx = [mp[lab] for lab in labels_ref]
                        M = M[np.ix_(idx, idx)]

                    mats.append(M)
                    kept += 1

                if len(mats):
                    M_avg = np.mean(np.stack(mats, axis=0), axis=0)
                    title = f"{model_name} | {rep} | aggregate over {kept} inputs | {args.metric.upper()}"
                    out_png = os.path.join(
                        out_root, f"aggregate__{model_name.replace('/','_')}__{rep}__{args.metric}.png"
                    )
                    heatmap(M_avg, labels_ref, title, out_png)
                    print(f"[viz] wrote {out_png}")
                    if args.save_json:
                        with open(out_png.replace(".png", ".json"), "w") as f:
                            json.dump({"labels": labels_ref, "matrix": M_avg.tolist()}, f, indent=2)
                else:
                    print(f"[viz] (aggregate) nothing to aggregate for {model_name} | {rep}")

    print("[viz] done.")

if __name__ == "__main__":
    main()
