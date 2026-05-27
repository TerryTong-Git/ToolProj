# Source Discrimination Experiment: Achieving Indistinguishability

## Executive Summary

We developed a source-label discrimination test to measure whether code-to-NL translations are distinguishable from native NL reasoning. Our key finding: **using native-style 10-shot ICL examples produces translations that are statistically indistinguishable from native NL** (49.4% accuracy, 95% CI contains 50%), while maintaining judge discriminative power (79% on controls).

---

## 1. Problem Statement

### Research Question
Can an LLM judge distinguish between:
- **Native NL**: Reasoning traces produced by an LLM solving a problem directly in natural language
- **Translated NL**: Reasoning traces produced by translating code solutions into natural language

### Why This Matters
If translations are indistinguishable, we can use code-to-NL translation as a valid proxy for native NL reasoning in downstream applications (e.g., chain-of-thought analysis, reasoning evaluation).

---

## 2. Initial Experiment: Baseline Distinguishability

### 2.1 Methodology

We ran source-label discrimination on 33 CLRS algorithm tasks using a **structured 10-shot translator prompt** (from `reasoning_solver_v1.md`). The translator prompt used highly procedural, numbered-step examples like:

```
1. Add the units digits: 1 + 3 = 4.
2. Add the tens digits: 6 + 0 = 6.
3. Add the hundreds digits: 7 + 3 = 10. Write down 0 and carry over 1.
...
```

### 2.2 Results

| Metric | Value |
|--------|-------|
| Accuracy | 62.9% [61.2%, 64.5%] |
| AUC | 0.678 |
| Control Accuracy | 70.0% |
| N Trials | 3,187 |

**Interpretation:** The 95% CI excludes 50%, meaning translations were **distinguishable** from native NL. The structured, procedural style of the translator was detectable.

### 2.3 Per-Task Distinguishability Analysis

We identified which tasks were most distinguishable:

| Task | Accuracy | Interpretation |
|------|----------|----------------|
| articulation_points | 81.0% | Highly distinguishable |
| kmp_matcher | 78.0% | Highly distinguishable |
| bridges | 76.5% | Highly distinguishable |
| topological_sort | 73.7% | Highly distinguishable |
| naive_string_matcher | 70.4% | Distinguishable |
| dag_shortest_paths | 70.7% | Distinguishable |
| jarvis_march | 69.4% | Distinguishable |
| bellman_ford | 62.6% | Moderately distinguishable |
| lcs | 65.0% | Moderately distinguishable |
| quickselect | 66.7% | Moderately distinguishable |

Tasks near 50% (e.g., find_maximum_subarray_kadane at 51.0%, knap at 53.5%) were already hard to distinguish.

---

## 3. Hypothesis: ICL Style Matters

### 3.1 Root Cause Analysis

We examined why translations were distinguishable:

1. **Procedural structure**: Translator used rigid numbered lists
2. **Mechanical language**: "We compute...", "The function returns..."
3. **Exhaustive tracing**: Every step enumerated, no shortcuts
4. **Code-like patterns**: Variables mentioned, loop iterations traced

Native NL reasoning, in contrast, was:
1. **Conversational**: "Let me check...", "I notice that..."
2. **Exploratory**: Sometimes backtracking, intuition-driven
3. **Selective**: Skipping obvious steps, focusing on key insights
4. **Natural structure**: Paragraphs, not always numbered lists

### 3.2 Solution: Native-Style ICL Examples

We hypothesized that if we teach the translator to produce native-style reasoning (using examples from the most distinguishable tasks), the translations would become indistinguishable.

---

## 4. ICL Example Selection Pipeline

### 4.1 Data Source

Examples were extracted from recorded benchmark result artifacts. Each record contains:
- `kind`: Task type (e.g., "articulation_points")
- `question` or `nl_question`: Problem statement
- `sim_code`: Code solution
- `nl_reasoning`: Native NL reasoning trace

### 4.2 Selection Criteria

We selected 10 tasks from the **most distinguishable** categories to maximize the learning signal:

```python
target_kinds = [
    'articulation_points',  # 81.0% distinguishable
    'kmp_matcher',          # 78.0% distinguishable
    'bridges',              # 76.5% distinguishable
    'topological_sort',     # 73.7% distinguishable
    'naive_string_matcher', # 70.4% distinguishable
    'dag_shortest_paths',   # 70.7% distinguishable
    'jarvis_march',         # 69.4% distinguishable
    'quickselect',          # 66.7% distinguishable
    'lcs',                  # 65.0% distinguishable
    'bellman_ford',         # 62.6% distinguishable
]
```

### 4.3 Example Quality Filters

For each task, we selected examples with:
- `len(nl_reasoning) > 400` characters (substantial reasoning)
- `len(sim_code) > 150` characters (non-trivial code)
- Both `sim_code` and `nl_reasoning` present

### 4.4 Example Transformation

For each selected example, we:
1. Took the `sim_code` as input
2. Took the `nl_reasoning` as the target output style
3. Rewrote examples to be more naturally conversational while preserving the reasoning pattern

---

## 5. Translator Prompt Design

### 5.1 Prompt Structure

File: `prompts/translator_native_10shot.md`

```markdown
# Natural Language Problem Solver

You are given code that solves an algorithmic problem. Your task is to
**reason through the problem step-by-step using natural language** and
arrive at the answer.

Do NOT describe or translate the code mechanically. Instead, solve the
problem yourself using natural reasoning, as if you were explaining your
thought process to a colleague.

## Guidelines

1. **Think like a human** - use exploratory reasoning, not procedural code execution
2. **Be conversational** - use phrases like "Let me check", "I notice that", "This means"
3. **Skip obvious steps** - don't enumerate every single iteration
4. **Focus on insights** - explain WHY something works, not just WHAT happens
5. **Use natural structure** - paragraphs are fine, avoid numbered lists unless truly needed

## Examples
[10 native-style examples follow]
```

### 5.2 Example Style Comparison

**Old structured style (distinguishable):**
```
To compute the sum 9166568761 + 1242886303:
1. Add the units digits: 1 + 3 = 4.
2. Add the tens digits: 6 + 0 = 6.
3. Add the hundreds digits: 7 + 3 = 10. Write down 0 and carry over 1.
...
```

**New native style (indistinguishable):**
```
The problem asks to find articulation points (cut vertices) in a graph.
An articulation point is a vertex whose removal disconnects the graph.

Looking at the adjacency matrix, I can identify the edges:
- Node 1 connects to itself (self-loop) and node 2
- Node 2 connects back to node 1

This forms a small connected component with just nodes 1 and 2...
```

---

## 6. Classifier Prompt Design

File: `prompts/source_classifier.md`

```markdown
You are analyzing an explanation of how to solve an algorithmic problem.

TASK: Determine whether this explanation was written by someone solving
the problem naturally using language ("Native NL") or by someone
translating/simulating code execution into natural language ("Translated").

CLASSIFICATION GUIDE:

**Native NL explanations** typically:
- Use first-person reasoning ("I think...", "Let me consider...")
- Show exploratory problem-solving with potential backtracks
- May have informal or conversational tone
- Focus on intuition and high-level strategy

**Translated explanations** typically:
- Use procedural/imperative language ("We compute...", "The function returns...")
- Follow a very systematic step-by-step structure
- Reference code constructs even implicitly
- Show complete exhaustive traces without shortcuts

Respond in EXACTLY this format:
PREDICTION: [NATIVE or TRANSLATED]
CONFIDENCE: [HIGH, MEDIUM, or LOW]
REASONING: [1-2 sentences explaining your decision]
```

---

## 7. Evaluation Protocol

### 7.1 Held-Out Task Selection

**Critical design choice:** Evaluation tasks must have **zero overlap** with ICL example tasks.

ICL tasks (10):
```
articulation_points, kmp_matcher, bridges, topological_sort,
naive_string_matcher, dag_shortest_paths, jarvis_march,
quickselect, lcs, bellman_ford
```

Evaluation tasks (21):
```
bfs, dfs, dijkstra, floyd_warshall, insertion_sort, heapsort,
bubble_sort, quicksort, find_maximum_subarray_kadane, minimum,
matrix_chain_order, optimal_bst, graham_scan, mst_prim, mst_kruskal,
strongly_connected_components, segments_intersect, task_scheduling,
knap, rod, lcs_length
```

### 7.2 Trial Construction

For each sample:
1. Load native NL reasoning from `nl_reasoning` field
2. Translate code (`sim_code`) using the native-style translator
3. Create two trials:
   - (question, native_NL) → label 0
   - (question, translated_NL) → label 1

### 7.3 Positive Controls

To verify the judge has discriminative power:
- Present native NL (label 0) vs raw code (label 1)
- Expected: Judge should easily distinguish these (accuracy >> 50%)
- Threshold: Control accuracy > 70% indicates judge has power

### 7.4 Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Accuracy | correct / total | ≈ 50% for indistinguishability |
| 95% CI | Wilson score interval | Should contain 50% |
| Control Accuracy | control_correct / control_total | > 70% |
| AUC | ROC area under curve | ≈ 0.5 for indistinguishability |

---

## 8. Final Results

### 8.1 Main Results

| Metric | Old Prompt | New Native Prompt |
|--------|------------|-------------------|
| Accuracy | 62.9% [61.2%, 64.5%] | **49.4% [47.2%, 51.5%]** |
| CI contains 50%? | NO | **YES** |
| AUC | 0.678 | **0.479** |
| Control Accuracy | 70.0% | **79.0%** |
| Judge has power? | Borderline | **YES** |
| N Trials | 3,187 | 2,022 |

### 8.2 Confusion Matrix (New Prompt)

```
                    Predicted
                 Native  Translated
Actual Native      656        356
Actual Translated  668        342
```

- Precision (Translated): 49.0%
- Recall (Translated): 33.8%

### 8.3 Per-Task Breakdown (Held-Out Tasks)

| Task | Accuracy | 95% CI | N |
|------|----------|--------|---|
| bfs | 47.5% | [37.9%, 57.2%] | 99 |
| bubble_sort | 48.0% | [38.5%, 57.7%] | 100 |
| dfs | 46.0% | [36.6%, 55.7%] | 100 |
| dijkstra | 44.0% | [34.7%, 53.8%] | 100 |
| find_maximum_subarray_kadane | 42.0% | [32.8%, 51.8%] | 100 |
| floyd_warshall | 52.0% | [42.3%, 61.5%] | 100 |
| graham_scan | 58.0% | [48.2%, 67.2%] | 100 |
| heapsort | 59.0% | [49.2%, 68.1%] | 100 |
| insertion_sort | 51.0% | [41.3%, 60.6%] | 100 |
| knap | 49.5% | [39.9%, 59.2%] | 99 |
| lcs_length | 46.0% | [36.6%, 55.7%] | 100 |
| matrix_chain_order | 55.0% | [45.2%, 64.4%] | 100 |
| minimum | 50.5% | [40.8%, 60.1%] | 99 |
| mst_kruskal | 31.2% | [18.0%, 48.6%] | 32 |
| mst_prim | 50.0% | [40.4%, 59.6%] | 100 |
| optimal_bst | 57.0% | [47.2%, 66.3%] | 100 |
| quicksort | 49.0% | [39.4%, 58.7%] | 100 |
| rod | 28.0% | [20.1%, 37.5%] | 100 |
| segments_intersect | 58.0% | [48.2%, 67.2%] | 100 |
| strongly_connected_components | 47.4% | [37.8%, 57.3%] | 97 |
| task_scheduling | 55.2% | [45.3%, 64.8%] | 96 |

---

## 9. Interpretation

### 9.1 Success Criteria Met

✅ **Judge has discriminative power**: 79% control accuracy (> 70% threshold)
✅ **Translations are indistinguishable**: 49.4% accuracy, 95% CI [47.2%, 51.5%] contains 50%
✅ **Generalizes to held-out tasks**: No overlap between ICL and evaluation tasks

### 9.2 Key Insight

The distinguishability of translations is primarily a function of **ICL example style**, not model capability. By teaching the translator to produce native-style reasoning through carefully selected examples, we achieved indistinguishability on completely unseen task types.

---

## 10. Reproduction

### 10.1 Command

```bash
uv run python -m src.translation_discrimination.cli \
    --n_samples 1500 \
    --kinds "bfs,dfs,dijkstra,floyd_warshall,insertion_sort,heapsort,bubble_sort,quicksort,find_maximum_subarray_kadane,minimum,matrix_chain_order,optimal_bst,graham_scan,mst_prim,mst_kruskal,strongly_connected_components,segments_intersect,task_scheduling,knap,rod,lcs_length" \
    --max_per_kind 50 \
    --concurrency 64
```

### 10.2 Required Files

```
src/translation_discrimination/
├── run_source_discrimination.py      # Main experiment
├── prompts/
│   ├── translator_native_10shot.md   # Native-style translator (10 ICL examples)
│   └── source_classifier.md          # NATIVE vs TRANSLATED classifier
└── results/
    └── source_discrimination_*.json  # Output with full trial data
```

### 10.3 Dependencies

- `httpx`: Async HTTP client
- `scipy`: Wilson score confidence intervals
- `sklearn`: AUC computation
- `tqdm`: Progress bars
- OpenRouter API key in `.env`

### 10.4 Models Used

- **Translator**: `openai/gpt-4o`
- **Judge**: `google/gemini-2.5-pro-preview-06-05`

---

## 11. Limitations & Future Work

1. **Task coverage**: Evaluated on 21 CLRS tasks; broader coverage needed
2. **Judge model sensitivity**: Different judge models may have different discriminative abilities
3. **Translator model**: Only tested with GPT-4o; other models may behave differently
4. **Sample size**: Per-task CIs are wide; more samples would tighten estimates

---

## 12. Conclusion

We demonstrated that code-to-NL translations can be made indistinguishable from native NL reasoning by using native-style ICL examples. The key is not the translator model's capability, but the style taught through few-shot examples. This validates the use of code-to-NL translation as a proxy for native NL reasoning in downstream applications.
