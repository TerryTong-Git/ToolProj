# mi_experiment.py
import os, argparse, math, json
import numpy as np
import pandas as pd

from cox.store import Store
from torch.utils.tensorboard import SummaryWriter

# ---------- numerics ----------
def _logsumexp(a, axis=None):
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

def _safe_log(x):
    x = np.clip(x, 1e-300, 1.0)
    return np.log(x, dtype=np.float64) if hasattr(np, "log") else math.log(x)

def _to_bits(nats):
    return float(nats) / math.log(2.0)

def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    s = v.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full_like(v, 1.0 / max(1, v.size), dtype=np.float64)
    v = v / (s + eps)
    return v

# ---------- p(z|theta,x) builders ----------
def build_pz_given_theta_counts(df_ab_theta, support, alpha=1e-3):
    """Count-based histogram over a fixed support with Dirichlet smoothing."""
    index = {sid: i for i, sid in enumerate(support)}
    vec = np.full(len(support), float(alpha), dtype=np.float64)
    for r in df_ab_theta.itertuples(index=False):
        i = index[getattr(r, "seq_id")]
        vec[i] += 1.0
    return _normalize(vec)

def build_pz_given_theta_logweights(df_ab_theta, support, alpha=1e-3):
    """
    Weight by exp(sum_logp_joint) but do it stably:
    - accumulate per-seq log-scores with logsumexp
    - softmax across the union support (plus Dirichlet smoothing)
    """
    # collect log-scores per seq_id
    per_seq_logs = {}
    for r in df_ab_theta.itertuples(index=False):
        sid = getattr(r, "seq_id")
        logp = float(getattr(r, "sum_logp_joint"))
        if sid not in per_seq_logs:
            per_seq_logs[sid] = [logp]
        else:
            per_seq_logs[sid].append(logp)

    # turn into a vector on the same support
    logvec = np.full(len(support), -np.inf, dtype=np.float64)
    idx = {sid: i for i, sid in enumerate(support)}
    for sid, lst in per_seq_logs.items():
        logvec[idx[sid]] = _logsumexp(np.array(lst, dtype=np.float64))

    # convert to probabilities with smoothing
    # softmax(logvec) ~ exp(logvec - max)
    m = np.max(logvec)
    exps = np.exp(logvec - m, dtype=np.float64, where=np.isfinite(logvec))
    exps = np.where(np.isfinite(exps), exps, 0.0)
    exps += float(alpha)  # Dirichlet prior
    return _normalize(exps)

# ---------- MI per input ----------
def mi_for_one_x(
    df_x: pd.DataFrame,
    thetas_all,
    alpha=1e-3,
    use_weighted=True,
    prior_mode="uniform",  # "uniform" or "weighted"
):
    """
    df_x: dataframe for a fixed (a,b), a single representation and model.
    thetas_all: iterable of theta labels expected in the run
    Returns (mi_nats, H_theta, H_theta_given_Z, support, valid_thetas, debug)
    """
    # Union support across thetas & seeds for this x
    support = sorted(df_x["seq_id"].unique().tolist())
    if len(support) == 0:
        return 0.0, 0.0, 0.0, support, [], {"reason": "empty support"}

    # Keep only thetas actually present for this x
    present = sorted(df_x["theta"].unique().tolist())
    valid_thetas = [th for th in thetas_all if th in present]
    if len(valid_thetas) < 2:
        return 0.0, 0.0, 0.0, support, valid_thetas, {"reason": "fewer than 2 thetas present"}

    # p(z|theta,x)
    builders = {
        True:  build_pz_given_theta_logweights,
        False: build_pz_given_theta_counts,
    }
    build = builders[bool(use_weighted)]
    P = []
    theta_weights = []  # for weighted prior if requested
    for th in valid_thetas:
        df_th = df_x[df_x["theta"] == th]
        P.append(build(df_th, support, alpha=alpha))
        theta_weights.append(len(df_th))
    P = np.stack(P, axis=0)  # [T, Z] float64

    # prior over thetas present
    if prior_mode == "weighted":
        pth = _normalize(np.array(theta_weights, dtype=np.float64))
    else:
        pth = np.full(len(valid_thetas), 1.0 / len(valid_thetas), dtype=np.float64)

    # mixture: p(z|x) = sum_theta p(theta) p(z|theta,x)
    pz = _normalize((pth[:, None] * P).sum(axis=0))

    # MI: sum_{theta,z} p(theta) p(z|theta) log p(z|theta)/p(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(P, pz[None, :], out=np.ones_like(P), where=(pz[None, :] > 0))
        terms = pth[:, None] * P * np.log(np.clip(ratio, 1e-300, None), dtype=np.float64)
        mi_nats = float(terms.sum())

    # Entropies (nats)
    H_theta = -float((pth * np.log(np.clip(pth, 1e-300, None), dtype=np.float64)).sum())
    H_theta_given_Z = H_theta - mi_nats

    # Clip tiny negative due to FP
    if mi_nats < 0 and mi_nats > -1e-9:
        mi_nats = 0.0

    dbg = {
        "support_size": len(support),
        "p_theta": dict(zip(valid_thetas, pth.tolist())),
    }
    return mi_nats, H_theta, H_theta_given_Z, support, valid_thetas, dbg

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, type=str, help="Path used by runner.py (has Cox store).")
    ap.add_argument("--exp_id", default=None, type=str, help="Cox exp_id if you used one.")
    ap.add_argument("--alpha", default=1e-3, type=float, help="Dirichlet smoothing for p(z|theta,x).")
    ap.add_argument("--use_weighted", action="store_true",
                    help="Use exp(logp) weights; otherwise count-based histograms.")
    ap.add_argument("--theta_prior", choices=["uniform","weighted"], default="uniform",
                    help="Prior over thetas present for this input.")
    ap.add_argument("--log_tb", action="store_true", help="Also log MI to TensorBoard under out_dir/tb_mi.")
    args = ap.parse_args()

    store = Store(args.out_dir, exp_id=args.exp_id)
    if "samples" not in store.tables:
        raise RuntimeError("No 'samples' table found. Run runner.py first (preferably with --paired_inputs).")

    df = store["samples"].df.copy()
    # Basic sanity columns
    needed = {"model_name","rep","theta","a","b","seed","seq_id","sum_logp_joint"}
    if not needed.issubset(set(df.columns)):
        missing = needed - set(df.columns)
        raise RuntimeError(f"'samples' table missing columns: {missing}")

    # Create MI table if needed
    if "mi_stats" not in store.tables:
        store.add_table("mi_stats", {
            "model_name": str, "rep": str,
            "a": int, "b": int,
            "support_size": int,
            "num_thetas_present": int,
            "mi_bits": float,
            "H_theta_bits": float,
            "H_theta_given_Z_bits": float
        })

    tb = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb_mi")) if args.log_tb else None

    # Compute MI per (model, rep) and per input (a,b)
    summaries = []
    for (model_name, rep), df_mr in df.groupby(["model_name", "rep"]):
        thetas_all = sorted(df_mr["theta"].unique().tolist())

        mi_bits_list = []
        for (aa, bb), df_x in df_mr.groupby(["a","b"]):
            mi_nats, H_theta_n, H_cond_n, support, valid_thetas, dbg = mi_for_one_x(
                df_x, thetas_all, alpha=args.alpha, use_weighted=args.use_weighted,
                prior_mode=args.theta_prior
            )
            mi_bits = _to_bits(mi_nats)
            H_bits = _to_bits(H_theta_n)
            Hc_bits = _to_bits(H_cond_n)

            store["mi_stats"].append_row({
                "model_name": model_name,
                "rep": rep,
                "a": int(aa), "b": int(bb),
                "support_size": int(len(support)),
                "num_thetas_present": int(len(valid_thetas)),
                "mi_bits": float(mi_bits),
                "H_theta_bits": float(H_bits),
                "H_theta_given_Z_bits": float(Hc_bits),
            })

            if tb is not None:
                tag = f"{model_name}/{rep}/a{aa}_b{bb}"
                tb.add_scalar(f"{tag}/MI_bits", mi_bits)
                tb.add_scalar(f"{tag}/H_theta_bits", H_bits)
                tb.add_scalar(f"{tag}/H_theta_given_Z_bits", Hc_bits)

            mi_bits_list.append(mi_bits)

        avg_mi = float(np.mean(mi_bits_list)) if mi_bits_list else float("nan")
        summaries.append({
            "model_name": model_name,
            "rep": rep,
            "avg_mi_bits": avg_mi,
            "num_inputs": len(mi_bits_list),
            "thetas_present": thetas_all
        })
        if tb is not None:
            tb.add_scalar(f"{model_name}/{rep}/MI_bits/avg_over_inputs", avg_mi)

        print(f"[mi] {model_name:>30s} | {rep:>4s} | avg I(Z;Theta|X) = {avg_mi:.4f} bits over {len(mi_bits_list)} inputs")

    # Write JSON summary
    out_json = os.path.join(args.out_dir, "mi_summary.json")
    with open(out_json, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"[mi] wrote {out_json}")

    if tb is not None:
        tb.flush()
        tb.close()

if __name__ == "__main__":
    main()
