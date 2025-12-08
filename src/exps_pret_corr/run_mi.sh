#!/bin/bash
# Run the pretraining data mutual information pipeline
#
# Usage:
#   ./run_mi.sh                    # Run with defaults
#   ./run_mi.sh --small            # Quick test run with small samples
#   ./run_mi.sh --full             # Full run with more samples
#
# Run tests from project root:
#   pytest tests/pret_corr -v

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default parameters
MAX_SAMPLES=10000
SUBSAMPLE_RATE=0.01
PROBES_PER_KIND=16
HF_MODEL="google/gemma-2-9b-it"
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Parse arguments
if [[ "$1" == "--small" ]]; then
    echo "Running in SMALL mode (quick test)..."
    MAX_SAMPLES=500
    SUBSAMPLE_RATE=0.1
    PROBES_PER_KIND=8
elif [[ "$1" == "--full" ]]; then
    echo "Running in FULL mode..."
    MAX_SAMPLES=50000
    SUBSAMPLE_RATE=0.02
    PROBES_PER_KIND=32
else
    echo "Running with DEFAULT parameters..."
fi

# Create output directory
mkdir -p outputs

echo "=============================================="
echo "Pretraining Data Mutual Information Pipeline"
echo "=============================================="
echo "Max samples per dataset: $MAX_SAMPLES"
echo "Subsample rate: $SUBSAMPLE_RATE"
echo "Probes per kind: $PROBES_PER_KIND"
echo "HF Model: $HF_MODEL"
echo "Embedding Model: $EMBED_MODEL"
echo "=============================================="

# Run the pipeline (as a module to handle relative imports)
cd ..  # Go to src directory
python -m exps_pret_corr.pretrain_mi \
    --max-samples-per-dataset "$MAX_SAMPLES" \
    --subsample-rate "$SUBSAMPLE_RATE" \
    --probes-per-kind "$PROBES_PER_KIND" \
    --hf-model "$HF_MODEL" \
    --embed-model "$EMBED_MODEL" \
    --out-labels "outputs/labeled_pretrain.csv" \
    --out-results "outputs/mi_results.json" \
    --seed 42

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "  Labels: outputs/labeled_pretrain.csv"
echo "  Results: outputs/mi_results.json"
echo "=============================================="

