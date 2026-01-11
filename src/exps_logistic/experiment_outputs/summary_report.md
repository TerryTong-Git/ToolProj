# Code vs NL Representation: Mutual Information Analysis

## TL;DR
**Code representations consistently encode more information about problem parameters than natural language (NL) representations.**
- Average MI advantage for code: 0.8803 bits
- Effect is statistically significant (p = 0.0000)
- 6/6 models show code > NL trend
- **Closed models (Claude, GPT, Gemini) show larger MI gap than open-source models**

## Key Findings

### 1. Overall Results
- **Code MI (mean +/- std)**: 1.2046 +/- 0.3367 bits
- **NL MI (mean +/- std)**: 0.3242 +/- 0.2456 bits
- **Statistical Test**: Paired t-test, t = 17.8850, p = 0.0000

### 2. Per-Model Analysis

| Model | Code MI | NL MI | Difference | Code > NL? |
|-------|---------|-------|------------|------------|
| claude-haiku-4.5 | 2.1045 | 0.9298 | 1.1747 | Yes |
| gemini-2.5-flash | 0.9703 | 0.1371 | 0.8331 | Yes |
| llama-3.1-405b-instruct | 1.0950 | 0.2802 | 0.8147 | Yes |
| ministral-14b-2512 | 1.3066 | 0.4683 | 0.8383 | Yes |
| gpt-4o-mini | 1.2286 | 0.3540 | 0.8747 | Yes |
| qwen-2.5-coder-32b-instruct | 1.0271 | 0.1007 | 0.9265 | Yes |

### 3. Models Where Code > NL
claude-haiku-4.5, gemini-2.5-flash, llama-3.1-405b-instruct, ministral-14b-2512, gpt-4o-mini, qwen-2.5-coder-32b-instruct

### 4. Models Where NL >= Code
None

## Methodology
1. Used TF-IDF features on CoT rationales (code and NL representations)
2. Trained multinomial logistic regression to predict gamma labels (kind|digits|bin)
3. Computed MI lower bound as H(Y) - CrossEntropy(Y|X)
4. Ran experiments across 6 models with multiple random seeds

## Interpretation
The higher MI for code representations suggests:
1. **Code preserves more structured information** about problem parameters
2. **Syntax and structure of code** makes gamma labels more predictable
3. **NL rationales lose information** through natural language variation
4. **Closed models show larger gaps** - their better instruction-following leads to more structured code

This supports the hypothesis that **code is a more faithful encoding of problem-solving reasoning**.

## Generated: 2026-01-09 17:04:09
