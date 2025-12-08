# from src.exps_performance.algorithms import lcs_len,knap_01_max_value,rod_cut_max,assignment_min_cost,prodplan_max_profit,partition_min_diff
# import random
# from src.exps_performance.utils import rand_string
# # how big can dimensions get before computer blows up?

# def knap_01_max_value(W: List[int], V: List[int], C: int) -> int:
# def rod_cut_max(P: List[int]) -> int:
# def assignment_min_cost(C: List[List[int]]) -> int:
# def prodplan_max_profit(d: Dict[str, Any]) -> int:
# def partition_min_diff(weights: List[int]) -> int:


# def test_lcs(benchmark, d):
#     benchmark.group = d
#     rng = random.Random(1)
#     n = max(2, int(d))  # max 2 digit lcs
#     s = rand_string(rng, alpha="abcd", n=n)
#     t = rand_string(rng, alpha="abce", n=n)
#     result = benchmark(lcs_len,  s, t)

# def test_knapsack(benchmark, )
