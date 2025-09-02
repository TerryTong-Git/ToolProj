# cot_jointprob_exp/metrics.py
from typing import Dict, List, Tuple
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
