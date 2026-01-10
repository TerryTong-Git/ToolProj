# Experiment: Screen New Models for Code vs NL Trend

## Hypothesis

Models that perform well on code execution will show higher mutual information (MI) between code Chain-of-Thought (CoT) representations and problem parameters compared to natural language (NL) representations.

## Variables

### Independent Variables
- **Model**: 20 different LLM models across various providers (Anthropic, DeepSeek, Meta, Qwen, Mistral, Google, OpenAI)
- **Representation Type**: Code vs Natural Language CoT

### Dependent Variables
- **MI Lower Bound**: Mutual information lower bound estimated via logistic regression
- **Task Accuracy**: Performance on algorithmic problems

## Success Criteria

1. Successfully screen at least 15/20 models
2. For each successful model, measure code vs NL MI difference
3. Identify models with statistically significant code > NL MI trend
4. Generate reproducible results (random seed = 0)

## Methodology

### Phase 1: Smoke Tests
- Verify API connectivity for at least one model
- Verify result parsing works correctly
- Verify MI analysis pipeline works

### Phase 2: Unit Tests
- Test data loading from screening results
- Test embedding extraction
- Test logistic regression classifier
- Test MI computation

### Phase 3: Integration Tests
- Run full screening on 2-3 cheap models
- Run full MI analysis pipeline
- Verify end-to-end reproducibility

### Phase 4: Full Screening
- Screen all 20 models with n=5 samples
- Collect all results
- Run MI analysis on each model

## Models to Screen

### Anthropic (expected strong code > NL trend)
- anthropic/claude-sonnet-4
- anthropic/claude-opus-4

### DeepSeek (strong reasoning models)
- deepseek/deepseek-chat-v3-0324
- deepseek/deepseek-r1
- deepseek/deepseek-r1-distill-llama-70b
- deepseek/deepseek-r1-distill-qwen-14b

### Meta Llama (open source baseline)
- meta-llama/llama-3.3-70b-instruct
- meta-llama/llama-4-maverick
- meta-llama/llama-4-scout

### Qwen (strong code models)
- qwen/qwen3-235b-a22b
- qwen/qwen3-32b
- qwen/qwq-32b

### Mistral (varied sizes)
- mistralai/mistral-large-2411
- mistralai/mistral-medium-3.1
- mistralai/codestral-2508
- mistralai/mixtral-8x22b-instruct

### Google (expected strong trend)
- google/gemini-2.5-pro
- google/gemini-2.0-flash-001
- google/gemini-3-pro-preview

### OpenAI
- openai/gpt-4o

## Configuration

```bash
N_SAMPLES=5           # Quick screening with small samples
SEED=0                # Reproducibility
DIGITS="2 4 6"        # Problem sizes
KINDS="add sub mul lcs knap rod"  # Problem types
BATCH_SIZE=16
```

## Results Directory

```
src/exps_performance/results_screening/
  {model}_seed{seed}/
    tb/
      run_{timestamp}/
        res.jsonl
```

## Documentation

- **Google Doc**: https://docs.google.com/document/d/1hNxAFNAqD0rn4mFbw_rM4LQRsrCQ8IZoOonFGV4LoM8/edit
- **Branch**: exp/screen-new-models

## Commands

### Run Screening
```bash
bash src/exps_performance/scripts/screen_new_models.sh
```

### Run MI Analysis
```bash
uv run --no-sync python -m src.exps_logistic.main \
  --results-dir src/exps_performance/results_screening \
  --label gamma --feats tfidf --bits
```
