from __future__ import annotations

from typing import Any, Dict, List, Optional


def _try_import_pulp() -> Optional[Any]:
    try:
        import pulp  # type: ignore

        return pulp
    except Exception:
        return None


def lcs_len(s: str, t: str) -> int:
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if s[i] == t[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def knap_01_max_value(W: List[int], V: List[int], C: int) -> int:
    n = len(W)
    dp = [0] * (C + 1)
    for i in range(n):
        w, val = W[i], V[i]
        for c in range(C, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + val)
    return dp[C]


def rod_cut_max(P: List[int]) -> int:
    n = len(P)
    dp = [0] * (n + 1)
    for L in range(1, n + 1):
        best = P[L - 1]  # one piece of length L
        for k in range(1, L):
            best = max(best, dp[k] + dp[L - k])
        dp[L] = best
    return dp[n]


# ---- ILP / combinatorial helpers ----


def assignment_min_cost(C: List[List[int]]) -> int:
    n_rows = len(C)
    n_cols = len(C[0]) if C else 0
    if n_rows == 0 or n_cols == 0:
        return 0
    pulp = _try_import_pulp()
    if pulp is not None:
        prob = pulp.LpProblem("assign", pulp.LpMinimize)
        x = [[pulp.LpVariable(f"x_{i}_{j}", 0, 1, cat="Binary") for j in range(n_cols)] for i in range(n_rows)]
        prob += pulp.lpSum(C[i][j] * x[i][j] for i in range(n_rows) for j in range(n_cols))
        for i in range(n_rows):
            prob += pulp.lpSum(x[i][j] for j in range(n_cols)) == 1
        for j in range(n_cols):
            prob += pulp.lpSum(x[i][j] for i in range(n_rows)) <= 1
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        val = int(round(pulp.value(prob.objective)))
        return val
    # brute-force permutations (n small, e.g., <=5)
    import itertools

    best = float("inf")
    for perm in itertools.permutations(range(n_cols), min(n_rows, n_cols)):
        cost = sum(C[i][perm[i]] for i in range(min(n_rows, n_cols)))
        best = min(best, cost)
    return int(best)


def prodplan_max_profit(d: Dict[str, Any]) -> int:
    profit: List[int] = d["profit"]
    consumption: List[List[int]] = d["consumption"]  # R x P
    capacity: List[int] = d["capacity"]  # R
    upper: List[int] = d["upper_bound"]  # P
    R = len(consumption)
    P = len(profit)
    pulp = _try_import_pulp()
    if pulp is not None:
        prob = pulp.LpProblem("prodplan", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=int(upper[j]), cat="Integer") for j in range(P)]
        prob += pulp.lpSum(profit[j] * x[j] for j in range(P))
        for i in range(R):
            prob += pulp.lpSum(consumption[i][j] * x[j] for j in range(P)) <= capacity[i]
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        val = int(round(pulp.value(prob.objective)))
        return val
    # bounded brute force (P<=4, bounds small)
    best = 0

    def dfs(j: int, cur_prof: int, use: List[int]) -> None:
        nonlocal best
        if j == P:
            best = max(best, cur_prof)
            return
        for q in range(0, upper[j] + 1):
            ok = True
            for i in range(R):
                if use[i] + consumption[i][j] * q > capacity[i]:
                    ok = False
                    break
            if not ok:
                break
            for i in range(R):
                use[i] += consumption[i][j] * q
            dfs(j + 1, cur_prof + profit[j] * q, use)
            for i in range(R):
                use[i] -= consumption[i][j] * q

    dfs(0, 0, [0] * R)
    return int(best)


def partition_min_diff(weights: List[int]) -> int:
    total = sum(weights)
    target = total // 2
    possible = 1  # bitset; bit k means sum k achievable
    for w in weights:
        possible = possible | (possible << w)
    # scan for achievable sum closest to target
    best = None
    for s in range(target, -1, -1):
        if (possible >> s) & 1:
            best = s
            break
    if best is None:
        return total
    return int(total - 2 * best)
