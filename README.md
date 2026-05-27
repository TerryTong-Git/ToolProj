# Code as a Cognitive Lever: Measuring Information in LLM Reasoning Strategies

Research repository for studying how LLMs encode and process information across different reasoning strategies (natural language, code generation, code simulation). We evaluate 48 algorithmic problem types across 9 models to quantify the relationship between reasoning modality and task performance.

## Documentation

The `docs/` folder is the current source of truth for how the repository works and what was done in the project:

- `docs/README.md` for the map
- `docs/architecture.md` for the run pipeline and system structure
- `docs/experiment-tracks.md` for the active experiment families
- `docs/results-and-analysis.md` for result layout and validation
- `docs/development.md` for setup and command references
- `docs/project-history.md` for the timeline of major work

Some older sections below predate repo cleanup; prefer `docs/` when they disagree.

## Overview

The active repository is centered on three experiment families:

1. **Performance Benchmark** (`src/exps_performance/`) — the main runner that generates benchmark data across NL, simulated-code, executed-code, and control-simulation arms.
2. **Source Discrimination / Control Work** (`src/exps_control_again/`) — tests whether code-translated NL traces can be distinguished from native NL traces.
3. **Functional Experiments** (`src/exps_functional/`) — tests additivity and translation-preservation claims using the benchmark outputs as input data.

The benchmark outputs in `src/exps_performance/results/` act as the shared dataset for the later experiment families.

## Installation

### Prerequisites

- Python 3.11
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
# Small benchmark slice
uv run python src/exps_performance/main.py \
  --root src/exps_performance/ \
  --backend openrouter \
  --model "openai/gpt-4o-mini" \
  --seed 0 \
  --n 3 \
  --digits 2 4 8 \
  --kinds add sub mul lcs rod knap ilp_assign ilp_partition ilp_prod \
  --exec_code \
  --controlled_sim

# Or use the production script
bash src/exps_performance/scripts/prod_all.sh
```

### Run Source Discrimination

```bash
# Run translation discrimination experiment
uv run python src/exps_control_again/run_source_discrimination.py --n_samples 200
```

### Run Functional Experiments

```bash
uv run python src/exps_functional/run_additivity.py --n_samples 200
uv run python src/exps_functional/run_translation_additivity.py --n_samples 200
```

### Run Tests

```bash
uv run pytest tests/
```

## Project Structure

```
ToolProj/
├── src/
│   ├── exps_performance/             # Main benchmark runner and shared dataset
│   ├── exps_control_again/           # Source discrimination + embedding analysis
│   └── exps_functional/              # Additivity and translation-preservation experiments
│
├── docs/                             # Current project documentation
├── tests/
│   ├── unit/                         # Unit and artifact-validation tests
│   ├── integration/                  # End-to-end benchmark checks
│   └── logistic/                     # Legacy-named analysis/parsing tests
│
├── figures/                          # Root-level publication figures
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
