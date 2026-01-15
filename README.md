# ToolProj: LLM Reasoning Experiments

Research repository for studying LLM reasoning capabilities through structured problem-solving tasks.

## Overview

This repository contains experiments analyzing how LLMs perform on algorithmic and mathematical problems using different prompting strategies (natural language, code generation, simulation).

### Key Experiments

| Experiment | Description | Location |
|------------|-------------|----------|
| **Performance Experiments** | Evaluates LLM accuracy across problem types (arithmetic, dynamic programming, graph algorithms, NP-hard) with different reasoning strategies | `src/exps_performance/` |
| **MI Estimation** | Estimates mutual information between model reasoning traces and problem parameters using logistic regression | `src/exps_logistic/` |

## Installation

### Prerequisites

- Python 3.10
- CUDA 12.4+ (for GPU acceleration)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/TerryTong-Git/ToolProj.git
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
bash src/exps_performance/scripts/prod.sh
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

### Run Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test suite
uv run pytest tests/unit/test_data_validation_post_hoc.py -v
```

## Project Structure

```
ToolProj/
├── src/
│   ├── exps_performance/       # LLM performance experiments
│   │   ├── main.py             # Main experiment runner
│   │   ├── arms.py             # Different prompting strategies (NL, code, sim)
│   │   ├── problems/           # Problem generators (arithmetic, DP, graphs)
│   │   ├── results/            # Experiment results (validated, tracked)
│   │   └── figures/            # Generated plots
│   │
│   └── exps_logistic/          # MI estimation experiments
│       ├── main.py             # Logistic regression runner
│       ├── data_utils.py       # Data loading and preprocessing
│       ├── featurizer.py       # Text feature extraction
│       └── classifier.py       # Logistic regression classifier
│
├── tests/
│   └── unit/
│       └── test_data_validation_post_hoc.py  # Data integrity tests
│
├── pyproject.toml              # Project configuration
└── .env                        # API keys (not tracked)
```

## Experiment Results

The `src/exps_performance/results/` directory contains validated experiment results:

- **15 model-seed combinations** with exactly 1580 samples each
- **Models tested**: Gemini 2.0/2.5 Flash, GPT-4o-mini, Codestral, Mixtral, Qwen, Ministral
- **Problem types**: Arithmetic (add/sub/mul), Dynamic Programming (LCS, knapsack, rod cutting), ILP, CLRS algorithms, NP-hard problems

### Data Validation

All result files pass strict validation tests:
- Consistent sample counts (1580 per file)
- Required fields present
- No unexpected blank values
- Valid problem kinds

## Development

### Linting and Formatting

```bash
uv run ruff check src/
uv run ruff format src/
```

### Type Checking

```bash
uv run mypy src/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{toolproj2025,
  title={ToolProj: LLM Reasoning Experiments},
  author={Terry Tong},
  year={2025},
  url={https://github.com/TerryTong-Git/ToolProj}
}
```

## License

MIT License
