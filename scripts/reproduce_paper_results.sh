#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="${PAPER_DIR:-${ROOT_DIR}/../Bayesian_Tool_Use_source_20260521}"
REPRO_OUT_DIR="${REPRO_OUT_DIR:-results/paper_reproduction}"

cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce_paper_results.sh --list
  bash scripts/reproduce_paper_results.sh tables
  bash scripts/reproduce_paper_results.sh validation
  bash scripts/reproduce_paper_results.sh figures
  bash scripts/reproduce_paper_results.sh paper
  bash scripts/reproduce_paper_results.sh all

Environment:
  PAPER_DIR                  Paper source directory. Defaults to ../Bayesian_Tool_Use_source_20260521.
  REPRO_OUT_DIR              Output directory for regenerated tables. Defaults to results/paper_reproduction.
  RUN_RECOVERY_NOTEBOOK=1    Execute the recovery-vs-digits notebook during figure reproduction.
EOF
}

list_commands() {
  cat <<'EOF'
Tables:
  uv run python src/exps_performance/scripts/analyze_route_accuracy_tables.py
  uv run python src/exps_functional/scripts/analyze_translation_shot_ablation.py
  uv run python src/exps_performance/scripts/analyze_rlm_subset_results.py
  uv run python src/exps_performance/scripts/analyze_coding_model_table.py
  uv run python src/exps_performance/scripts/analyze_code_failure_distribution.py
  uv run python src/exps_performance/scripts/analyze_frontier_nopatch_table.py

Five percent validation:
  uv run pytest tests/integration/test_analyze_sim_code_overlap_e2e.py -q
  uv run pytest tests/integration/test_route_accuracy_tables_e2e.py -q
  uv run pytest tests/integration/test_translation_shot_ablation_e2e.py -q
  uv run pytest tests/integration/test_rlm_subset_results_e2e.py -q
  uv run pytest tests/integration/test_coding_model_table_e2e.py -q
  uv run pytest tests/integration/test_code_failure_distribution_e2e.py -q
  uv run pytest tests/integration/test_frontier_nopatch_table_e2e.py -q

Figures:
  uv run python src/exps_performance/analysis.py
  uv run python src/exps_control_again/scripts/plot_judge_discrimination.py
  uv run python src/exps_control_again/scripts/native_vs_translated_scatter.py
  uv run python src/exps_functional/scripts/plot_translation_additivity.py
  RUN_RECOVERY_NOTEBOOK=1 uv run jupyter nbconvert --to notebook --execute src/exps_performance/notebooks/recovery_vs_digits.ipynb --output /tmp/recovery_vs_digits.executed.ipynb

Paper:
  cd "$PAPER_DIR" && latexmk -pdf -interaction=nonstopmode example_paper.tex
EOF
}

run() {
  echo "+ $*"
  "$@"
}

run_tables() {
  mkdir -p "$REPRO_OUT_DIR"
  run uv run python src/exps_performance/scripts/analyze_route_accuracy_tables.py \
    --report-path "$REPRO_OUT_DIR/route_accuracy_tables.md" \
    --complexity-csv "$REPRO_OUT_DIR/accuracy_by_asymptotic_class.csv" \
    --model-csv "$REPRO_OUT_DIR/accuracy_by_model.csv"
  run uv run python src/exps_functional/scripts/analyze_translation_shot_ablation.py \
    --report-path "$REPRO_OUT_DIR/functional_shot_ablation_summary.md" \
    --csv-path "$REPRO_OUT_DIR/functional_shot_ablation_summary.csv"
  run uv run python src/exps_performance/scripts/analyze_rlm_subset_results.py \
    --report-path "$REPRO_OUT_DIR/rlm_results.md"
  run uv run python src/exps_performance/scripts/analyze_coding_model_table.py \
    --output-md "$REPRO_OUT_DIR/coding_model_table.md"
  run uv run python src/exps_performance/scripts/analyze_code_failure_distribution.py \
    --output "$REPRO_OUT_DIR/code_failure_distribution.csv"
  run uv run python src/exps_performance/scripts/analyze_frontier_nopatch_table.py \
    --output-md "$REPRO_OUT_DIR/frontier_nopatch_table.md"
}

run_validation() {
  run uv run pytest tests/integration/test_analyze_sim_code_overlap_e2e.py -q
  run uv run pytest tests/integration/test_route_accuracy_tables_e2e.py -q
  run uv run pytest tests/integration/test_translation_shot_ablation_e2e.py -q
  run uv run pytest tests/integration/test_rlm_subset_results_e2e.py -q
  run uv run pytest tests/integration/test_coding_model_table_e2e.py -q
  run uv run pytest tests/integration/test_code_failure_distribution_e2e.py -q
  run uv run pytest tests/integration/test_frontier_nopatch_table_e2e.py -q
}

copy_if_present() {
  local source_path="$1"
  local target_name="$2"

  if [[ ! -f "$source_path" ]]; then
    echo "skip copy: missing $source_path"
    return
  fi
  if [[ ! -d "$PAPER_DIR/images" ]]; then
    echo "skip copy: missing $PAPER_DIR/images"
    return
  fi

  cp -f "$source_path" "$PAPER_DIR/images/$target_name"
  echo "copied $source_path -> $PAPER_DIR/images/$target_name"
}

copy_figures_to_paper() {
  copy_if_present figures/combined_accuracy_delta.png combined_accuracy_delta.png
  copy_if_present figures/combined_accuracy_delta.pdf combined_accuracy_delta.pdf
  copy_if_present figures/main_combined.png main_combined.png
  copy_if_present figures/main_combined.pdf main_combined.pdf
  copy_if_present src/exps_control_again/results/judge_discrimination_barplot.png judge_discrimination_barplot.png
  copy_if_present src/exps_control_again/results/judge_discrimination_barplot.pdf judge_discrimination_barplot.pdf
  copy_if_present src/exps_control_again/results/native_vs_translated_scatter.png native_vs_translated_scatter.png
  copy_if_present src/exps_control_again/results/native_vs_translated_scatter.pdf native_vs_translated_scatter.pdf
  copy_if_present src/exps_functional/results/translation_additivity.png translation_additivity.png
  copy_if_present src/exps_functional/results/translation_additivity.pdf translation_additivity.pdf
  copy_if_present src/exps_performance/figures/recovery_vs_digits_overall.png recovery_vs_digits_overall.png
}

run_figures() {
  mkdir -p figures src/exps_performance/figures
  run uv run python src/exps_performance/analysis.py
  run uv run python src/exps_control_again/scripts/plot_judge_discrimination.py
  run uv run python src/exps_control_again/scripts/native_vs_translated_scatter.py
  run uv run python src/exps_functional/scripts/plot_translation_additivity.py

  if [[ "${RUN_RECOVERY_NOTEBOOK:-0}" == "1" ]]; then
    run uv run jupyter nbconvert \
      --to notebook \
      --execute src/exps_performance/notebooks/recovery_vs_digits.ipynb \
      --output /tmp/recovery_vs_digits.executed.ipynb
  else
    echo "skip recovery notebook: set RUN_RECOVERY_NOTEBOOK=1 to regenerate recovery_vs_digits_overall.png"
  fi

  copy_figures_to_paper
}

run_paper() {
  if [[ ! -d "$PAPER_DIR" ]]; then
    echo "missing paper source directory: $PAPER_DIR" >&2
    exit 1
  fi
  if ! command -v latexmk >/dev/null 2>&1; then
    echo "missing latexmk; install a TeX distribution before building the paper" >&2
    exit 1
  fi

  (cd "$PAPER_DIR" && run env LC_ALL=en_US.UTF-8 LC_CTYPE=en_US.UTF-8 LANG=en_US.UTF-8 latexmk -pdf -interaction=nonstopmode example_paper.tex)
}

target="${1:-all}"
case "$target" in
  --help|-h)
    usage
    ;;
  --list|list)
    list_commands
    ;;
  tables)
    run_tables
    ;;
  validation)
    run_validation
    ;;
  figures)
    run_figures
    ;;
  paper)
    run_paper
    ;;
  all)
    run_tables
    run_figures
    run_paper
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
