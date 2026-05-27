# Route Accuracy Tables

Source data: checked-in main-run JSONL files matched by `src/exps_performance/results/*/tb/run_*/res.jsonl`.

Filters applied:
- includes the 44 tasks in the rebuttal complexity mapping
- excludes `bsp`, `msp`, `gcp_d`, and `tsp_d`, which are present in raw runs but not in the rebuttal table
- includes the six full-coverage models used in the main aggregate
- excludes `unused/` result directories and subset/debug runs

Conventions:
- `Route 1`, `Route 2`, and `Route 3` correspond to `nl_correct`, `sim_correct`, and `code_correct`
- accuracies and deltas are percentages / percentage-point differences
- `Instances` counts unique `(kind, digit, index_in_kind)` problems
- McNemar p-values are exact two-sided tests on paired outcomes

## Results Grouped By Asymptotic Complexity Class

| Complexity Class | Tasks | Instances | Route 1 | Route 2 | Route 3 | Route 2 - Route 1 | McNemar p (Route 2 vs Route 1) | Route 3 - Route 2 | McNemar p (Route 3 vs Route 2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O(1) | 4 | 227 | 49.8 | 49.4 | 85.0 | -0.4 | 0.52 | 35.6 | <1e-300 |
| O(log n) | 1 | 35 | 22.4 | 28.6 | 32.9 | 6.2 | 2.7e-04 | 4.3 | 0.0197 |
| O(n) | 4 | 135 | 4.8 | 5.0 | 7.5 | 0.2 | 0.635 | 2.5 | 2.2e-07 |
| O(n log n) | 5 | 68 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 1 |
| O(n^2) | 17 | 357 | 12.4 | 12.0 | 49.0 | -0.4 | 0.157 | 37.0 | <1e-300 |
| O(n^2 log n) | 1 | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 1 |
| O(n^3) | 4 | 37 | 0.0 | 0.0 | 0.0 | 0.0 | 1 | 0.0 | 1 |
| NP-hard | 8 | 480 | 23.8 | 21.7 | 59.2 | -2.1 | 5.0e-09 | 37.5 | <1e-300 |

## Results Grouped By Model

Only full-coverage models are shown; each row has `1340` problems and `4020` paired evaluations.

| Model | Type | Instances | Route 1 | Route 2 | Route 3 | Route 2 - Route 1 | McNemar p (Route 2 vs Route 1) | Route 3 - Route 2 | McNemar p (Route 3 vs Route 2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anthropic/claude-haiku-4.5 | closed | 1340 | 31.4 | 27.7 | 48.2 | -3.7 | 4.4e-13 | 20.5 | 6.6e-123 |
| google/gemini-2.0-flash-001 | closed | 1340 | 22.3 | 21.1 | 54.4 | -1.2 | 0.0159 | 33.3 | <1e-300 |
| google/gemini-2.5-flash | closed | 1340 | 23.4 | 23.5 | 54.3 | 0.0 | 0.961 | 30.8 | 1.5e-277 |
| openai/gpt-4o-mini | closed | 1340 | 18.2 | 16.8 | 49.6 | -1.4 | 0.00312 | 32.8 | 2.7e-280 |
| mistralai/codestral-2508 | open | 1340 | 17.6 | 18.9 | 52.3 | 1.2 | 0.00113 | 33.5 | 1.3e-284 |
| mistralai/mixtral-8x22b-instruct | open | 1340 | 15.2 | 15.7 | 42.8 | 0.5 | 0.143 | 27.1 | 3.0e-221 |

Task mapping:
- `O(1)`: `add`, `sub`, `mul`, `segments_intersect`
- `O(log n)`: `binary_search`
- `O(n)`: `minimum`, `quickselect`, `find_maximum_subarray_kadane`, `kmp_matcher`
- `O(n log n)`: `activity_selector`, `task_scheduling`, `heapsort`, `quicksort`, `graham_scan`
- `O(n^2)`: `lcs`, `rod`, `knap`, `bubble_sort`, `insertion_sort`, `lcs_length`, `naive_string_matcher`, `jarvis_march`, `dfs`, `bfs`, `topological_sort`, `strongly_connected_components`, `articulation_points`, `bridges`, `mst_prim`, `dijkstra`, `dag_shortest_paths`
- `O(n^2 log n)`: `mst_kruskal`
- `O(n^3)`: `matrix_chain_order`, `optimal_bst`, `floyd_warshall`, `bellman_ford`
- `NP-hard`: `ilp_assign`, `ilp_prod`, `ilp_partition`, `edp`, `gcp`, `ksp`, `spp`, `tsp`

Companion CSVs:
- `src/exps_performance/results/analysis/accuracy_by_asymptotic_class.csv`
- `src/exps_performance/results/analysis/accuracy_by_model.csv`
