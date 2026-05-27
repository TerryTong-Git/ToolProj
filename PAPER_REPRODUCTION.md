# Paper Reproduction

This file records the commands needed to regenerate the paper tables, figures,
and PDF from ToolProj artifacts. The exact reproducibility gate first generates
deterministic experiment shards that cover at least 5% of each paper result
source, verifies them against a gold manifest, and then runs the table analyzers
and normalized figure-source extractors on those generated artifacts.

The paper source of truth is the sibling directory:

```bash
../Bayesian_Tool_Use_source_20260521
```

The older sibling `../Bayesian_Tool_Use` is not used for result matching.

## One Command

```bash
uv run python scripts/reproduce_paper.py --paper-dir ../Bayesian_Tool_Use_source_20260521 all
```

This regenerates the compact table outputs, regenerates the scripted figures, copies generated figures into the paper source, and builds `example_paper.pdf`.

To require generated experiment artifacts before tables are rebuilt:

```bash
uv run python scripts/reproduce_paper.py --paper-dir ../Bayesian_Tool_Use_source_20260521 all --shard 5pct --require-generated-results
```

That command writes deterministic experiment shards under
`results/paper_reproduction/experiments/`, verifies
`results/paper_reproduction/manifest_observed.json` against
`tests/fixtures/paper_reproduction/gold_manifest.json`, and passes the generated
artifact paths into the table analyzers. The verification step also regenerates
the compact analyzer outputs plus normalized figure-source CSVs and
byte-compares them with
`tests/fixtures/paper_reproduction/gold_outputs/`.

The older shell entry point still works:

```bash
PAPER_DIR=../Bayesian_Tool_Use_source_20260521 bash scripts/reproduce_paper_results.sh all
```

The Python entry point also accepts `--output-dir` for regenerated table outputs.
The old `PAPER_DIR`, `REPRO_OUT_DIR`, and `RUN_RECOVERY_NOTEBOOK` environment
variables remain supported as defaults.

Use `--dry-run` to inspect the commands for any target without running them:

```bash
uv run python scripts/reproduce_paper.py --dry-run tables
```

Use `--list experiments`, `--list verify-5pct`, `--list tables`,
`--list validation`, `--list figures`, or `--list paper` to print only one
command group.

## Tables

```bash
uv run python scripts/reproduce_paper.py --output-dir results/paper_reproduction tables
```

To rebuild tables from freshly generated 5% experiment artifacts:

```bash
uv run python scripts/reproduce_paper.py experiments --shard 5pct --output-dir results/paper_reproduction
uv run python scripts/reproduce_paper.py verify-5pct --output-dir results/paper_reproduction
uv run python scripts/reproduce_paper.py tables --input-dir results/paper_reproduction --output-dir results/paper_reproduction
```

The table source is `../Bayesian_Tool_Use_source_20260521/Appendix/part_1.tex`.

The driver writes regenerated tables to `results/paper_reproduction/` so it does not overwrite the checked-in rebuttal summaries.

| Paper result | Command | Output |
|---|---|---|
| Complexity class table | `uv run python -m src.reasoning_benchmark.scripts.analyze_route_accuracy_tables` | `results/paper_reproduction/route_accuracy_tables.md` |
| Model route table | `uv run python -m src.reasoning_benchmark.scripts.analyze_route_accuracy_tables` | `results/paper_reproduction/route_accuracy_tables.md` |
| Prompt shot ablation | `uv run python -m src.translation_additivity.reports.shot_ablation` | `results/paper_reproduction/functional_shot_ablation_summary.md` |
| RLM results | `uv run python -m src.reasoning_benchmark.scripts.analyze_rlm_subset_results` | `results/paper_reproduction/rlm_results.md` |
| Coding model table | `uv run python -m src.reasoning_benchmark.scripts.analyze_coding_model_table` | `results/paper_reproduction/coding_model_table.md` |
| Code failure distribution | `uv run python -m src.reasoning_benchmark.scripts.analyze_code_failure_distribution` | `results/paper_reproduction/code_failure_distribution.csv` |
| Frontier no-patching table | `uv run python -m src.reasoning_benchmark.scripts.analyze_frontier_nopatch_table` | `results/paper_reproduction/frontier_nopatch_table.md` |
| Structured sim/code overlap | `uv run python -m src.reasoning_benchmark.scripts.analyze_sim_code_overlap` | `results/paper_reproduction/sim_code_overlap.md` |

Without `--input-dir`, the commands read saved final result artifacts. With
`--input-dir`, the analyzers read the generated experiment shards and produce
exact outputs for the 5% reproducibility gate.

For the full 0-5 shot sweep, first run `src/translation_additivity/scripts/run_shot_ablation.sh`. Without those raw sweep files, the shot-ablation analysis only regenerates the legacy 10-shot row.

## Five Percent Validation

```bash
uv run python scripts/reproduce_paper.py experiments --shard 5pct --output-dir results/paper_reproduction
uv run python scripts/reproduce_paper.py verify-5pct --output-dir results/paper_reproduction
uv run python scripts/reproduce_paper.py validation
```

`experiments --shard 5pct` writes deterministic generated result artifacts for
every paper table and figure source in the manifest. `verify-5pct` hashes those
artifacts, checks that each shard covers at least 5% of the corresponding final
source, reruns the analyzers and figure-source extractors against the generated
artifacts, and byte-compares the regenerated markdown or CSV outputs with the
checked-in gold outputs.
`validation` keeps the smaller analyzer regression tests that assert exact
markdown or CSV values for targeted shards.

## Figures

```bash
uv run python scripts/reproduce_paper.py figures
```

The figure references are in `../Bayesian_Tool_Use_source_20260521/sections/*.tex`.

| Paper figure | Command | Generated file |
|---|---|---|
| Main route accuracy and delta | `uv run python -m src.reasoning_benchmark.analysis.reports` | `figures/combined_accuracy_delta.png` |
| Per-task accuracy by difficulty | `uv run python -m src.reasoning_benchmark.analysis.reports` | `figures/main_combined.png` |
| Judge discrimination barplot | `uv run python -m src.translation_discrimination.reports.judge_discrimination` | `src/translation_discrimination/results/judge_discrimination_barplot.png` |
| Native vs translated scatter | `uv run python -m src.translation_discrimination.reports.native_vs_translated_scatter` | `src/translation_discrimination/results/native_vs_translated_scatter.png` |
| Translation additivity | `uv run python -m src.translation_additivity.reports.translation_additivity` | `src/translation_additivity/results/translation_additivity.png` |
| Recovery vs digits | `uv run python scripts/reproduce_paper.py --run-recovery-notebook figures` | `src/exps_performance/figures/recovery_vs_digits_overall.png` |

The recovery figure is notebook-backed. The default figure command does not run the notebook unless `--run-recovery-notebook` is set.
When `figures` or `all --require-generated-results` is run with generated 5%
artifacts, the driver also writes normalized figure-source CSVs under
`results/paper_reproduction/figure_sources/`. The exact `verify-5pct` gate
byte-compares those CSVs against checked-in gold fixtures; pixel equality is not
the required exactness boundary.

## Paper Build

```bash
uv run python scripts/reproduce_paper.py --paper-dir ../Bayesian_Tool_Use_source_20260521 paper
```

This runs:

```bash
cd ../Bayesian_Tool_Use_source_20260521
env LC_ALL=en_US.UTF-8 LC_CTYPE=en_US.UTF-8 LANG=en_US.UTF-8 latexmk -pdf -interaction=nonstopmode example_paper.tex
```

The paper build expects a local TeX installation with `latexmk`.

## Validation Commands

```bash
bash -n scripts/reproduce_paper_results.sh
uv run python scripts/reproduce_paper.py --list
bash scripts/reproduce_paper_results.sh --list
uv run python scripts/reproduce_paper.py experiments --dry-run --shard 5pct --output-dir /tmp/toolproj-paper-gate
uv run python scripts/reproduce_paper.py experiments --shard 5pct --output-dir /tmp/toolproj-paper-gate
uv run python scripts/reproduce_paper.py verify-5pct --output-dir /tmp/toolproj-paper-gate
uv run python scripts/reproduce_paper.py figures --input-dir /tmp/toolproj-paper-gate --output-dir /tmp/toolproj-paper-figures --dry-run
uv run python scripts/reproduce_paper.py --dry-run tables
uv run python scripts/reproduce_paper.py tables --input-dir /tmp/toolproj-paper-gate --output-dir /tmp/toolproj-paper-tables
uv run python scripts/reproduce_paper.py --dry-run --paper-dir ../Bayesian_Tool_Use_source_20260521 paper
uv run pytest tests/integration/test_analyze_sim_code_overlap_e2e.py -q
uv run pytest tests/integration/test_route_accuracy_tables_e2e.py -q
uv run pytest tests/integration/test_translation_shot_ablation_e2e.py -q
uv run pytest tests/integration/test_rlm_subset_results_e2e.py -q
uv run pytest tests/integration/test_coding_model_table_e2e.py -q
uv run pytest tests/integration/test_code_failure_distribution_e2e.py -q
uv run pytest tests/integration/test_frontier_nopatch_table_e2e.py -q
```

These tests check that the compact reproduction scripts still produce the expected result summaries.
