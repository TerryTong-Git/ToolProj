#!/bin/bash
# Synchronized Logistic Regression Pipeline
# Runs logistic regression on all exps_performance results and generates plots
#
# Usage: bash src/exps_logistic/sync_and_generate.sh [--dry-run]

set -e

# Configuration
RESULTS_DIR="${RESULTS_DIR:-src/exps_performance/results}"
LOGISTIC_OUTPUT_DIR="src/exps_logistic/results"
PLOTS_DIR="src/exps_logistic/notebooks"
TODAY=$(date +%Y%m%d)
DRY_RUN=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
    esac
done

echo "========================================"
echo "Synchronized Logistic Regression Pipeline"
echo "========================================"
echo "Results directory: $RESULTS_DIR"
echo "Output directory: $LOGISTIC_OUTPUT_DIR"
echo "Date: $TODAY"
echo "Dry run: $DRY_RUN"
echo ""

# Ensure output directory exists
mkdir -p "$LOGISTIC_OUTPUT_DIR"

# Discover all model-seed directories
MODEL_DIRS=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "*_seed*" | sort)
TOTAL_DIRS=$(echo "$MODEL_DIRS" | wc -l | tr -d ' ')

echo "Found $TOTAL_DIRS model-seed combinations:"
echo "$MODEL_DIRS" | xargs -I{} basename {}
echo ""

# Track progress
COMPLETED=0
FAILED=0
FAILED_LIST=""

# Process each model-seed combination
for dir in $MODEL_DIRS; do
    dirname=$(basename "$dir")

    # Parse model name and seed from directory name (format: {model}_seed{N})
    # Handle model names with underscores by finding last _seed occurrence
    SEED=$(echo "$dirname" | grep -oE 'seed[0-9]+$' | sed 's/seed//')
    MODEL=$(echo "$dirname" | sed "s/_seed${SEED}$//")

    echo "----------------------------------------"
    echo "Processing: $MODEL (seed $SEED)"
    echo "----------------------------------------"

    # Check if res.jsonl files exist
    RES_FILES=$(find "$dir" -name "res.jsonl" 2>/dev/null | head -1)
    if [ -z "$RES_FILES" ]; then
        echo "  WARNING: No res.jsonl found in $dir, skipping..."
        continue
    fi

    for REP in code nl; do
        echo "  Running $REP representation..."

        CMD="uv run --no-sync python -m src.exps_logistic.main \
            --results-dir \"$RESULTS_DIR\" \
            --models \"$MODEL\" \
            --seed $SEED \
            --rep $REP \
            --device cpu \
            --hf-batch 1 \
            --label gamma \
            --no-cv \
            --bits \
            --feats hf-cls \
            --embed-model google-bert/bert-base-uncased \
            --max_iter 20 \
            --C 0.5 \
            --kinds-preset extended"

        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY RUN] Would execute:"
            echo "  $CMD"
        else
            if eval $CMD; then
                echo "  ✓ $REP completed"
                ((COMPLETED++))
            else
                echo "  ✗ $REP failed"
                ((FAILED++))
                FAILED_LIST="$FAILED_LIST\n  - $MODEL seed$SEED $REP"
            fi
        fi
    done
done

echo ""
echo "========================================"
echo "Logistic Regression Complete"
echo "========================================"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"

if [ -n "$FAILED_LIST" ]; then
    echo "Failed experiments:$FAILED_LIST"
fi

# Generate plots
echo ""
echo "========================================"
echo "Generating Plots"
echo "========================================"

# Set TARGET_RUN_DATES to today's date pattern
export TARGET_RUN_DATES="${TODAY}_"
echo "TARGET_RUN_DATES set to: $TARGET_RUN_DATES"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would run: uv run python src/exps_logistic/notebooks/generate_plots.py"
else
    echo "Running plot generation..."
    uv run --no-sync python src/exps_logistic/notebooks/generate_plots.py

    echo ""
    echo "Generated plots:"
    ls -la "$PLOTS_DIR"/*.png 2>/dev/null | tail -10
fi

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "New logistic results:"
ls "$LOGISTIC_OUTPUT_DIR"/*${TODAY}*.json 2>/dev/null | wc -l | xargs -I{} echo "  {} files created today"
echo ""
echo "Plots location: $PLOTS_DIR/"
