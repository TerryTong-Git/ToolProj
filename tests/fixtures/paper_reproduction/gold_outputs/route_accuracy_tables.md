# Route Accuracy Tables

Source data: generated deterministic 5% route-accuracy shards.

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
| O(1) | 4 | 20 | 40.0 | 60.0 | 80.0 | 20.0 | 1.2e-07 | 20.0 | 1.2e-07 |
| O(log n) | 1 | 5 | 40.0 | 60.0 | 80.0 | 20.0 | 0.0313 | 20.0 | 0.0313 |
| O(n) | 4 | 20 | 40.0 | 60.0 | 80.0 | 20.0 | 1.2e-07 | 20.0 | 1.2e-07 |
| O(n log n) | 5 | 25 | 40.0 | 56.0 | 80.0 | 16.0 | 1.2e-07 | 24.0 | 2.9e-11 |
| O(n^2) | 17 | 79 | 32.9 | 43.0 | 70.9 | 10.1 | 7.1e-15 | 27.8 | 3.7e-40 |
| O(n^2 log n) | 1 | 4 | 25.0 | 50.0 | 75.0 | 25.0 | 0.0313 | 25.0 | 0.0313 |
| O(n^3) | 4 | 16 | 25.0 | 50.0 | 75.0 | 25.0 | 1.2e-07 | 25.0 | 1.2e-07 |
| NP-hard | 8 | 32 | 25.0 | 50.0 | 75.0 | 25.0 | 7.1e-15 | 25.0 | 7.1e-15 |

## Results Grouped By Model

Only full-coverage models are shown; each row has `1340` problems and `4020` paired evaluations.

| Model | Type | Instances | Route 1 | Route 2 | Route 3 | Route 2 - Route 1 | McNemar p (Route 2 vs Route 1) | Route 3 - Route 2 | McNemar p (Route 3 vs Route 2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anthropic/claude-haiku-4.5 | closed | 201 | 33.3 | 50.2 | 75.1 | 16.9 | 1.2e-10 | 24.9 | 1.8e-15 |
| google/gemini-2.0-flash-001 | closed | 201 | 33.3 | 50.2 | 75.1 | 16.9 | 1.2e-10 | 24.9 | 1.8e-15 |
| google/gemini-2.5-flash | closed | 201 | 33.3 | 50.2 | 75.1 | 16.9 | 1.2e-10 | 24.9 | 1.8e-15 |
| openai/gpt-4o-mini | closed | 201 | 33.3 | 50.2 | 75.1 | 16.9 | 1.2e-10 | 24.9 | 1.8e-15 |
| mistralai/codestral-2508 | open | 201 | 33.3 | 50.2 | 75.1 | 16.9 | 1.2e-10 | 24.9 | 1.8e-15 |
| mistralai/mixtral-8x22b-instruct | open | 201 | 33.3 | 50.2 | 75.1 | 16.9 | 1.2e-10 | 24.9 | 1.8e-15 |

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
- `accuracy_by_asymptotic_class.csv`
- `accuracy_by_model.csv`
