# Plan: Scale Up Code vs NL Experiments on Promising Models

**Created:** 2026-01-10
**Priority:** 1
**Estimated Time:** 4-6 hours (parallelized API calls)

## Research Question

Do the four models identified in screening (codestral-2501, mistral-large-2411, gemini-2.0-flash-001, mixtral-8x22b-instruct) maintain their strong Code > NL accuracy advantage when evaluated on a statistically significant sample size across fine-grained, CLRS, and NP-hard problem types?

## Hypothesis

**H1 (Primary):** The four promising models will show statistically significant Code > NL accuracy differences (p < 0.05) across all problem categories, maintaining the 40%+ advantage observed in screening.

**H2 (Secondary):** The Code > NL effect will be consistent across problem difficulty (digit sizes) and problem types (fine-grained arithmetic, CLRS algorithms, NP-hard optimization).

**H3 (Exploratory):** Code execution accuracy will exceed code simulation accuracy, demonstrating the value of actual execution over simulation.

## Supporting Evidence

From the screening experiment (Jan 10, 2026):

| Model | Code Accuracy | NL Accuracy | Difference | Screening n |
|-------|---------------|-------------|------------|-------------|
| codestral-2508 | 94.4% | 50.0% | +44.4% | 18 |
| mistral-large-2411 | 88.9% | 44.4% | +44.4% | 18 |
| gemini-2.0-flash-001 | 94.4% | 50.0% | +44.4% | 18 |
| mixtral-8x22b-instruct | 88.9% | 44.4% | +44.4% | 18 |

These models showed the strongest Code > NL performance among 20 screened models. However, n=18 is insufficient for publication-quality claims. This experiment scales to n > 500 samples per model.

## Variables

- **Independent Variables:**
  - Model identity (4 levels: codestral-2501, mistral-large-2411, gemini-2.0-flash-001, mixtral-8x22b-instruct)
  - Problem type (20+ levels: fine-grained, CLRS30, NP-hard)
  - Problem difficulty (digit sizes: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
  - Representation arm (4 levels: NL, Code Sim, Controlled Code Sim, Code Exec)

- **Dependent Variables:**
  - Per-arm accuracy (binary correct/incorrect per sample)
  - Mean accuracy per model/arm/kind combination
  - Code - NL accuracy difference (effect size)

- **Controls:**
  - Fixed random seed (seed=0,1,2) for reproducibility
  - Same prompt templates across models
  - Same problem instances across models
  - Temp at 0

## Methodology

### Phase 1: Data Generation (~3-4 hours)

1. **Configure sample sizes:**
   - CLRS30: 500 samples (clrs_samples=500)
   - Fine-grained kinds (9 types): 30 samples per digit x 10 digits x 9 kinds = 2,700 samples
   - NP-hard kinds (9 types): 30 samples per type = 270 samples
   - Total: ~3,470 samples per model per seed
   - With 3 seeds x 4 models = ~41,640 total samples

2. **Run main.py for each model:**
   - Backend: openrouter (async API calls)
   - Batch size: 128 (maximize concurrent requests)
   - Checkpoint every: 128 (efficient checkpointing)
   - Workers: 4 (code execution parallelism)

3. **Parallelization strategy:**
   - Models are IO-bound (API calls), not CPU-bound
   - Run all 4 models in parallel using background processes
   - Each model writes to separate results directory

### Phase 2: Data Validation (~30 min)

1. **Verify sample counts:**
   - Check each res.jsonl has expected number of records
   - Verify coverage of all kinds and digit ranges
   - Check for missing or incomplete records

2. **Sanity checks:**
   - Accuracy values in [0, 1] range
   - No NaN or null values in correctness columns
   - Balanced sample counts across arms

### Phase 3: Statistical Analysis (~1 hour)

1. **Compute accuracy metrics:**
   - Mean accuracy per (model, arm, kind) combination
   - 95% confidence intervals via bootstrap (1000 resamples)

2. **Statistical tests:**
   - Paired Wilcoxon signed-rank test: Code vs NL per model
   - Two-way ANOVA: Model x Arm interaction
   - Effect size: Cohen's d for Code - NL difference

3. **Significance thresholds:**
   - Primary: alpha = 0.05 (Bonferroni corrected for 4 models)
   - Exploratory: alpha = 0.10

### Phase 4: Visualization (~30 min)

1. **Primary figure (line.png style):**
   - X-axis: Arm (NL, Sim, ControlSim, Code)
   - Y-axis: Accuracy
   - Lines: One per model, colored by provider
   - Include "All models" aggregate line

2. **Secondary figures:**
   - Accuracy by problem kind (faceted)
   - Accuracy by digit (showing difficulty scaling)
   - Code - NL difference heatmap (model x kind)

## Implementation

### Environment Setup

```bash
cd /Users/terrytong/Documents/CCG/ToolProj
source .env
export UV_CACHE_DIR="${UV_CACHE_DIR:-./src/models/}"
export HF_HOME="${HF_HOME:-./src/models/}"
```

### Main Experiment Script

Save as `src/exps_performance/scripts/run_scaled_experiment.sh`:

```bash
#!/bin/bash
# Scale-up experiment for 4 promising Code > NL models
# Optimized for maximum parallelism via large batch sizes

set -e

SCRIPT_DIR="$(cd -- "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Source environment variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${REPO_ROOT}/src/models/}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/src/models/}"

# Experiment parameters - scaled up for statistical power
N_SAMPLES=30          # Per kind per digit for fine-grained
CLRS_SAMPLES=500      # Total CLRS30 samples
SEEDS=(0 1 2)         # Multiple seeds for reproducibility
DIGITS="2 4 6 8 10 12 14 16 18 20"
BATCH_SIZE=128        # Large batch for IO parallelism
CHECKPOINT_EVERY=128
EXEC_WORKERS=4
MAX_TOKENS=4096
TEMPERATURE=0         # Deterministic sampling

# Fine-grained kinds (9 types)
FG_KINDS="add sub mul lcs knap rod ilp_assign ilp_prod ilp_partition"

# NP-hard kinds (9 types)
NPHARD_KINDS="spp tsp tsp_d msp ksp gcp gcp_d bsp edp"

# All kinds including CLRS30
ALL_KINDS="${FG_KINDS} clrs30 ${NPHARD_KINDS}"

# 4 promising models from screening
MODELS=(
    "mistralai/codestral-2508"
    "mistralai/mistral-large-2411"
    "google/gemini-2.0-flash-001"
    "mistralai/mixtral-8x22b-instruct"
)

RESULTS_ROOT="src/exps_performance/results/scaled_code_nl"
mkdir -p "${RESULTS_ROOT}"

echo "=============================================="
echo "Scaled Code vs NL Experiment"
echo "=============================================="
echo "Parameters:"
echo "  N_SAMPLES (per kind/digit): ${N_SAMPLES}"
echo "  CLRS_SAMPLES: ${CLRS_SAMPLES}"
echo "  SEEDS: ${SEEDS[*]}"
echo "  DIGITS: ${DIGITS}"
echo "  BATCH_SIZE: ${BATCH_SIZE}"
echo "  EXEC_WORKERS: ${EXEC_WORKERS}"
echo "  TEMPERATURE: ${TEMPERATURE}"
echo "  Total models: ${#MODELS[@]}"
echo "=============================================="

# Log file for tracking
LOG_FILE="${RESULTS_ROOT}/experiment_log.txt"
echo "Experiment started: $(date)" > "${LOG_FILE}"

# Function to run a single model with a specific seed
run_model() {
    local MODEL=$1
    local SEED=$2
    local LOG_PREFIX=$(echo "${MODEL}" | tr '/' '_')_seed${SEED}

    echo "[$(date)] Starting: ${MODEL} seed=${SEED}" | tee -a "${LOG_FILE}"

    if uv run --no-sync python src/exps_performance/main.py \
        --root "${RESULTS_ROOT}" \
        --backend openrouter \
        --model "${MODEL}" \
        --n "${N_SAMPLES}" \
        --digits ${DIGITS} \
        --kinds ${ALL_KINDS} \
        --clrs_samples "${CLRS_SAMPLES}" \
        --exec_code \
        --batch_size "${BATCH_SIZE}" \
        --checkpoint_every "${CHECKPOINT_EVERY}" \
        --seed "${SEED}" \
        --controlled_sim \
        --resume \
        --exec_workers "${EXEC_WORKERS}" \
        --max_tokens "${MAX_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        2>&1 | tee "${RESULTS_ROOT}/${LOG_PREFIX}.log"; then
        echo "[$(date)] SUCCESS: ${MODEL} seed=${SEED}" | tee -a "${LOG_FILE}"
        return 0
    else
        echo "[$(date)] FAILED: ${MODEL} seed=${SEED}" | tee -a "${LOG_FILE}"
        return 1
    fi
}

# Run all models x seeds in parallel
echo ""
echo "Starting parallel model runs (${#MODELS[@]} models x ${#SEEDS[@]} seeds = $((${#MODELS[@]} * ${#SEEDS[@]})) jobs)..."
PIDS=()
JOBS=()

for SEED in "${SEEDS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        run_model "${MODEL}" "${SEED}" &
        PIDS+=($!)
        JOBS+=("${MODEL}|seed${SEED}")
        echo "  Launched ${MODEL} seed=${SEED} (PID: ${PIDS[-1]})"
    done
done

echo ""
echo "Waiting for all ${#PIDS[@]} jobs to complete..."
echo "PIDs: ${PIDS[*]}"

# Wait for all processes and track results
SUCCESSFUL=()
FAILED=()

for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    JOB=${JOBS[$i]}
    if wait $PID; then
        SUCCESSFUL+=("${JOB}")
    else
        FAILED+=("${JOB}")
    fi
done

echo ""
echo "=============================================="
echo "Experiment Complete"
echo "=============================================="
echo "Successful (${#SUCCESSFUL[@]}):"
for m in "${SUCCESSFUL[@]}"; do
    echo "  - ${m}"
done
echo ""
echo "Failed (${#FAILED[@]}):"
for m in "${FAILED[@]}"; do
    echo "  - ${m}"
done
echo ""
echo "Results saved to: ${RESULTS_ROOT}"
echo "Completed: $(date)" >> "${LOG_FILE}"
```

### Single Model Command (for debugging)

```bash
uv run --no-sync python src/exps_performance/main.py \
    --root src/exps_performance/results/scaled_code_nl \
    --backend openrouter \
    --model "mistralai/codestral-2508" \
    --n 30 \
    --digits 2 4 6 8 10 12 14 16 18 20 \
    --kinds add sub mul lcs knap rod ilp_assign ilp_prod ilp_partition clrs30 spp tsp tsp_d msp ksp gcp gcp_d bsp edp \
    --clrs_samples 500 \
    --exec_code \
    --batch_size 128 \
    --checkpoint_every 128 \
    --seed 0 \
    --controlled_sim \
    --resume \
    --exec_workers 4 \
    --max_tokens 4096 \
    --temperature 0
```

### Analysis Script

After data collection, run analysis:

```bash
# Generate figures using existing analysis.py (modify filter for new models)
uv run --no-sync python -c "
from pathlib import Path
import pandas as pd
from src.exps_performance.logger import create_big_df
from src.exps_performance.analysis import plot_v_graph, plot_p_vals, plot_main_fig

# Load results
results_root = Path('src/exps_performance/results/scaled_code_nl')
jsonl_files = sorted(results_root.rglob('*.jsonl'))
df = create_big_df(jsonl_files)

# Filter to our 4 models
target_models = [
    'mistralai/codestral-2508',
    'mistralai/mistral-large-2411',
    'google/gemini-2.0-flash-001',
    'mistralai/mixtral-8x22b-instruct'
]
df = df[df['model'].isin(target_models)]

print(f'Total samples: {len(df)}')
print(f'Samples by model: {df.groupby(\"model\").size()}')

# Generate figures
plot_v_graph(df)
plot_p_vals(df)
plot_main_fig(df)
"
```

## Evaluation Criteria

### Success Criteria (all must be met for H1 confirmation)

- [ ] **Sample size:** >= 3,000 samples per model per seed collected
- [ ] **Statistical significance:** Wilcoxon p < 0.0125 (Bonferroni-corrected alpha) for Code > NL in at least 3/4 models
- [ ] **Effect size:** Cohen's d >= 0.5 (medium effect) for Code - NL difference
- [ ] **Consistency:** Code > NL advantage present across >= 80% of problem kinds

### Failure Criteria (any triggers H1 rejection)

- [ ] Wilcoxon p > 0.05 for Code > NL in >= 2 models
- [ ] Effect size Cohen's d < 0.2 (small effect)
- [ ] Code < NL accuracy for any model on aggregate

### Ambiguous Zone (requires follow-up)

- [ ] Mixed results: 2 models significant, 2 not
- [ ] Large variance across problem kinds
- [ ] Significant model x kind interaction effects

## Tractability Assessment

| Factor | Assessment | Notes |
|--------|------------|-------|
| Experiment loop time | 4-6 hours | API calls dominate; highly parallelizable |
| Data availability | Ready | Problem generators exist; CLRS30 dataset loaded |
| Tooling | Exists | main.py, analysis.py fully functional |
| Swamp risk | Low | Straightforward scale-up of existing pipeline |

## Risk Mitigation

1. **API rate limits:**
   - Monitor OpenRouter rate limits
   - Reduce batch_size if 429 errors occur
   - Use `--resume` for automatic checkpointing

2. **Model availability:**
   - Verify model IDs on OpenRouter before starting
   - Have backup model list (e.g., codestral-2501 vs codestral-2508)

3. **Cost management:**
   - Estimate: ~$150-300 total across all models (3 seeds x 4 models)
   - Monitor spend via OpenRouter dashboard

## Todoist Tasks

To be created after plan approval:
- [ ] Run scaled experiment script
- [ ] Validate data completeness
- [ ] Perform statistical analysis
- [ ] Generate publication figures
- [ ] Document results in Google Docs

## Open Questions

1. Should we include additional models from the original 6 (e.g., claude-haiku-4.5, gpt-4o-mini) for comparison baseline?

2. What is the correct model ID for Codestral? The screening used `codestral-2508` but the context mentions `codestral-2501`. Need to verify on OpenRouter.

3. Should we also run the logistic MI analysis (exps_logistic) on these results for a complementary information-theoretic perspective?

## Expected Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Setup & validation | 15 min | T+0 | T+15m |
| Data generation | 4-6 hours | T+15m | T+6h |
| Data validation | 30 min | T+6h | T+6.5h |
| Statistical analysis | 1 hour | T+6.5h | T+7.5h |
| Visualization | 30 min | T+7.5h | T+8h |
| Documentation | 30 min | T+8h | T+8.5h |

**Total estimated time: 8.5 hours** (can be reduced if models run faster than expected)

---

## Execution Checklist

Before starting:
- [ ] Verify OpenRouter API key is valid
- [ ] Verify model IDs exist on OpenRouter
- [ ] Confirm sufficient API credits (~$100 budget)
- [ ] Create results directory
- [ ] Source .env file

During execution:
- [ ] Monitor log files for errors
- [ ] Check checkpoint files growing
- [ ] Watch for rate limit errors

After completion:
- [ ] Verify all 4 models have results
- [ ] Count total samples per model
- [ ] Run analysis script
- [ ] Review generated figures
- [ ] Document in Google Docs
