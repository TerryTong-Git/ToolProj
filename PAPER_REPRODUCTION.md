# Paper Reproduction

This file records the commands needed to regenerate the paper tables, figures, and PDF from checked-in ToolProj artifacts.

The paper source of truth is the sibling directory:

```bash
../Bayesian_Tool_Use_source_20260521
```

The older sibling `../Bayesian_Tool_Use` is not used for result matching.

## One Command

```bash
PAPER_DIR=../Bayesian_Tool_Use_source_20260521 uv run python scripts/reproduce_paper.py all
```

This regenerates the compact table outputs, regenerates the scripted figures, copies generated figures into the paper source, and builds `example_paper.pdf`.

The older shell entry point still works:

```bash
PAPER_DIR=../Bayesian_Tool_Use_source_20260521 bash scripts/reproduce_paper_results.sh all
```

Use `--dry-run` to inspect the commands for any target without running them:

```bash
uv run python scripts/reproduce_paper.py --dry-run tables
```

## Tables

```bash
uv run python scripts/reproduce_paper.py tables
```

The table source is `../Bayesian_Tool_Use_source_20260521/Appendix/part_1.tex`.

The driver writes regenerated tables to `results/paper_reproduction/` so it does not overwrite the checked-in rebuttal summaries.

| Paper result | Command | Output |
|---|---|---|
| Complexity class table | `uv run python src/exps_performance/scripts/analyze_route_accuracy_tables.py` | `results/paper_reproduction/route_accuracy_tables.md` |
| Model route table | `uv run python src/exps_performance/scripts/analyze_route_accuracy_tables.py` | `results/paper_reproduction/route_accuracy_tables.md` |
| Prompt shot ablation | `uv run python src/exps_functional/scripts/analyze_translation_shot_ablation.py` | `results/paper_reproduction/functional_shot_ablation_summary.md` |
| RLM results | `uv run python src/exps_performance/scripts/analyze_rlm_subset_results.py` | `results/paper_reproduction/rlm_results.md` |
| Coding model table | `uv run python src/exps_performance/scripts/analyze_coding_model_table.py` | `results/paper_reproduction/coding_model_table.md` |
| Code failure distribution | `uv run python src/exps_performance/scripts/analyze_code_failure_distribution.py` | `results/paper_reproduction/code_failure_distribution.csv` |
| Frontier no-patching table | `uv run python src/exps_performance/scripts/analyze_frontier_nopatch_table.py` | `results/paper_reproduction/frontier_nopatch_table.md` |

Validation is simple. The commands read saved result artifacts. The generated tables should match the appendix values used in the paper source.

For the full 0-5 shot sweep, first run `src/exps_functional/scripts/run_translation_additivity_shot_ablation.sh`. Without those raw sweep files, the shot-ablation analysis only regenerates the legacy 10-shot row.

## Figures

```bash
uv run python scripts/reproduce_paper.py figures
```

The figure references are in `../Bayesian_Tool_Use_source_20260521/sections/*.tex`.

| Paper figure | Command | Generated file |
|---|---|---|
| Main route accuracy and delta | `uv run python src/exps_performance/analysis.py` | `figures/combined_accuracy_delta.png` |
| Per-task accuracy by difficulty | `uv run python src/exps_performance/analysis.py` | `figures/main_combined.png` |
| Judge discrimination barplot | `uv run python src/exps_control_again/scripts/plot_judge_discrimination.py` | `src/exps_control_again/results/judge_discrimination_barplot.png` |
| Native vs translated scatter | `uv run python src/exps_control_again/scripts/native_vs_translated_scatter.py` | `src/exps_control_again/results/native_vs_translated_scatter.png` |
| Translation additivity | `uv run python src/exps_functional/scripts/plot_translation_additivity.py` | `src/exps_functional/results/translation_additivity.png` |
| Recovery vs digits | `RUN_RECOVERY_NOTEBOOK=1 uv run jupyter nbconvert --to notebook --execute src/exps_performance/notebooks/recovery_vs_digits.ipynb --output /tmp/recovery_vs_digits.executed.ipynb` | `src/exps_performance/figures/recovery_vs_digits_overall.png` |

The recovery figure is notebook-backed. The default figure command does not run the notebook unless `RUN_RECOVERY_NOTEBOOK=1` is set.

## Paper Build

```bash
PAPER_DIR=../Bayesian_Tool_Use_source_20260521 uv run python scripts/reproduce_paper.py paper
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
uv run python scripts/reproduce_paper.py --dry-run tables
uv run pytest tests/integration/test_route_accuracy_tables_e2e.py -q
uv run pytest tests/integration/test_translation_shot_ablation_e2e.py -q
uv run pytest tests/integration/test_rlm_subset_results_e2e.py -q
uv run pytest tests/integration/test_coding_model_table_e2e.py -q
uv run pytest tests/integration/test_code_failure_distribution_e2e.py -q
uv run pytest tests/integration/test_frontier_nopatch_table_e2e.py -q
```

These tests check that the compact reproduction scripts still produce the expected result summaries.
