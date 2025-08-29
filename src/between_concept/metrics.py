from typing import Dict, List
import numpy as np
from scipy.special import logsumexp

def kl_divergence(p_log: np.ndarray, q_log: np.ndarray, eps: float = 1e-12) -> float:
    """
    KL(p||q) for two log-prob vectors over the same support.
    """
    p = np.exp(p_log - logsumexp(p_log))
    q = np.exp(q_log - logsumexp(q_log))
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))

def summarize_pairwise_kl(logprob_vectors: List[np.ndarray]) -> Dict[str, float]:
    """
    Given a list of log-prob vectors (same length, same support), compute pairwise KL stats.
    """
    if len(logprob_vectors) < 2:
        return dict(num_pairs=0, max_kl=np.nan, avg_kl=np.nan, var_kl=np.nan)
    vals = []
    for i in range(len(logprob_vectors)):
        for j in range(len(logprob_vectors)):
            if i != j:
                vals.append(kl_divergence(logprob_vectors[i], logprob_vectors[j]))
    vals = np.array(vals, dtype=float)
    return dict(num_pairs=int(vals.size), max_kl=float(vals.max()),
                avg_kl=float(vals.mean()), var_kl=float(vals.var()))

def dist_over_sequences_from_scores(seq_id_to_logp: Dict[str, float], support: List[str]) -> np.ndarray:
    """
    Convert a dict of {sequence_id: logprob} to a logprob vector over a fixed 'support' list.
    Missing sequences get logprob = -inf (handled via small epsilon in KL).
    """
    vec = np.full(len(support), -np.inf, dtype=float)
    for idx, sid in enumerate(support):
        lp = seq_id_to_logp.get(sid, None)
        if lp is not None:
            vec[idx] = lp
    return vec

# --------- NEW: concept-pair helpers ---------

def symmetric_kl(p_log: np.ndarray, q_log: np.ndarray, eps: float = 1e-12) -> float:
    """0.5*(KL(p||q)+KL(q||p)) on log-prob vectors."""
    return 0.5 * (kl_divergence(p_log, q_log, eps) + kl_divergence(q_log, p_log, eps))

def js_divergence(p_log: np.ndarray, q_log: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence on log-prob vectors."""
    p = np.exp(p_log - logsumexp(p_log))
    q = np.exp(q_log - logsumexp(q_log))
    m = 0.5 * (p + q)
    def _kl(a, b):
        return float(np.sum(a * (np.log(a + eps) - np.log(b + eps))))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def logmeanexp_pool(logps: List[float]) -> float:
    """
    Pool multiple log-probs for the same sequence (e.g., across seeds)
    by log-mean-exp: log( (1/S) * sum_s exp(logp_s) ).
    """
    if not logps:
        return -np.inf
    return float(logsumexp(np.array(logps, dtype=float)) - np.log(len(logps)))

def build_pooled_dist(rows, support: List[str]) -> np.ndarray:
    """
    rows: iterable of records with .seq_id and .sum_logp_joint
    Pools duplicates by log-mean-exp, then projects onto support as a log-prob vector.
    """
    bucket: Dict[str, List[float]] = {}
    for r in rows:
        bucket.setdefault(r.seq_id, []).append(float(r.sum_logp_joint))
    pooled = {sid: logmeanexp_pool(lps) for sid, lps in bucket.items()}
    return dist_over_sequences_from_scores(pooled, support)
