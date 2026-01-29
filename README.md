# Code as a Cognitive Lever: Measuring Information in LLM Reasoning Strategies

Research repository for studying how LLMs encode and process information across different reasoning strategies (natural language, code generation, code simulation). We evaluate 48 algorithmic problem types across 9 models to quantify the relationship between reasoning modality and task performance.

## Overview

This repository implements three categories of experiments:

1. **Performance Experiments** (`src/exps_performance/`) — Evaluates LLM accuracy on algorithmic tasks using three reasoning arms: natural language (NL), code execution, and code simulation. Covers arithmetic, dynamic programming, graph algorithms, and NP-hard problems across varying difficulty levels (2–20 digits).

2. **Mutual Information Estimation** (`src/exps_logistic/`) — Estimates mutual information between model reasoning traces and problem parameters using logistic regression, measuring how much task-relevant information each reasoning strategy captures.

3. **Control & Functional Experiments** (`src/exps_control_again/`, `src/exps_functional/`) — Tests indistinguishability of translated vs. native NL reasoning (source discrimination), embedding-based separability analysis, and translation additivity properties.

## Installation

### Prerequisites

- Python 3.10
- CUDA 12.4+ (for GPU acceleration, optional)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ToolProj

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Environment Variables

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key  # For accessing various LLMs
```

## Quick Start

### Run Performance Experiments

```bash
# Run experiments for a specific model
uv run python -m src.exps_performance.main \
  --model "google/gemini-2.0-flash-001" \
  --seed 0 \
  --output-dir results/

# Or use the production script
bash src/exps_performance/scripts/prod_all.sh
```

### Run MI Estimation

```bash
# Run logistic regression MI estimation
uv run python -m src.exps_logistic.main \
  --results-dir src/exps_performance/results \
  --rep nl \
  --label gamma

# Or use the production script
bash src/exps_logistic/prod_logistic.sh
```

### Run Source Discrimination

```bash
# Run translation discrimination experiment
uv run python src/exps_control_again/run_source_discrimination.py --n_samples 200
```

### Run Tests

```bash
uv run pytest tests/
```

## Project Structure

```
ToolProj/
├── src/
│   ├── exps_performance/             # LLM performance benchmark
│   │   ├── main.py                   # Main experiment runner
│   │   ├── arms.py                   # Reasoning strategies (NL, code, sim)
│   │   ├── analysis.py               # Statistical analysis
│   │   ├── logger.py                 # JSONL logging & checkpointing
│   │   ├── problems/                 # Problem generators
│   │   │   ├── finegrained.py        #   Arithmetic (add/sub/mul), DP, ILP
│   │   │   ├── clrs.py              #   CLRS algorithm problems (48 types)
│   │   │   └── nphardeval.py         #   NP-hard problems (TSP, GCP, etc.)
│   │   ├── clrs/                     # CLRS algorithm implementations
│   │   ├── Data_V2/                  # NP-hard benchmark datasets
│   │   ├── results/                  # Experiment results (25 model-seed runs)
│   │   ├── figures/                  # Generated plots
│   │   ├── notebooks/               # Analysis notebooks
│   │   └── scripts/                  # Production & analysis scripts
│   │
│   ├── exps_logistic/                # MI estimation via logistic regression
│   │   ├── main.py                   # Logistic regression runner
│   │   ├── data_utils.py             # Data loading and preprocessing
│   │   ├── featurizer.py             # Text feature extraction
│   │   └── classifier.py             # Logistic regression classifier
│   │
│   ├── exps_control_again/           # Source discrimination experiments
│   │   ├── run_source_discrimination.py  # Main experiment
│   │   ├── prompts/                  # Judge & translator prompts
│   │   │   ├── source_classifier.md  # Discrimination judge prompt
│   │   │   └── translator_native_10shot.md  # Code-to-NL translator prompt
│   │   ├── scripts/                  # Embedding & classifier analyses
│   │   └── results/                  # Discrimination results & plots
│   │
│   └── exps_functional/              # Functional property experiments
│       ├── run_translation_additivity.py  # Translation additivity test
│       ├── scripts/                  # Plot generation
│       └── results/                  # Trial data & figures
│
├── tests/
│   ├── unit/                         # 18 unit test files
│   ├── integration/                  # 3 integration test files
│   └── logistic/                     # 6 logistic regression tests
│
├── figures/                          # Root-level publication figures
├── Bayesian_Tool_Use/                # LaTeX paper source
├── pyproject.toml                    # Project configuration
└── .env                              # API keys (not tracked)
```

## Experiment Results

### Performance Benchmark

The `src/exps_performance/results/` directory contains validated experiment results:

- **25 model-seed runs** across 9 models with 3 seeds each
- **48 algorithmic problem types** spanning:
  - Fine-grained arithmetic (add, sub, mul) with digit scaling 2–20
  - Dynamic programming (LCS, knapsack, rod cutting)
  - Integer linear programming (assignment, production, partition)
  - CLRS algorithms (sorting, searching, graph algorithms, string matching)
  - NP-hard problems (TSP, graph coloring, bin packing, etc.)
- **Models tested**: Claude Haiku 4.5, Claude Opus 4, Gemini 2.0/2.5 Flash, GPT-4o-mini, Codestral 2508, Mixtral 8x22B, Qwen 2.5 Coder 32B, Ministral 14B
- **Three reasoning arms per sample**: NL reasoning, code execution, code simulation

### Source Discrimination

Tests whether code-to-NL translations are distinguishable from native NL reasoning:
- Judge model classifies traces as "native" or "translated"
- Positive controls verify judge discriminative power
- Results with confidence intervals and per-task breakdown

### Translation Additivity

Tests functional properties of the translation mapping across multiple translator models.

## Data Validation

All result files pass strict validation tests:
- Consistent sample counts per file
- Required fields present across all reasoning arms
- No unexpected blank values
- Valid problem kind labels

```bash
# Run validation tests
uv run pytest tests/unit/test_data_validation_post_hoc.py -v
```

## Development

```bash
# Lint and format
uv run ruff check src/ && uv run ruff format src/

# Type checking
uv run mypy src/

# Run full test suite (including slow tests)
uv run pytest tests/ -m ""
```

## License

MIT License
