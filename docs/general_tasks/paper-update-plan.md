# Plan: Update example_paper.tex to Match Current Experiments

**Created:** 2026-01-10
**Status:** Draft
**Target File:** `Bayesian_Tool_Use/example_paper.tex`

---

## Overview

Update the ICML paper "Neuro-symbolic language models for scaling algorithmic reasoning" to reflect the current experimental code in `src/exps_performance/` and `src/exps_logistic/`. Exclude pretraining experiments.

### Key Decisions (from Brainstorming)

| Decision | Choice |
|----------|--------|
| Primary Goal | Sync with current experiments + polish for submission |
| Models | Use only the 6 models with current results |
| Problem Scope | Cover all 44 problem kinds, keep presentation concise |
| Pretraining | Exclude entirely |

---

## Paper Writing Guide Reference

Follow `.claude/agents/How to Write a paper.docx` strictly. Key structure:

### Abstract Structure
Same as Introduction but replace "paragraph" with "sentence":
1. What is the problem?
2. How have people attempted to solve it?
3. What are the limitations?
4. What is your approach?
5. What are the benefits/results?

### Introduction Structure (5+ paragraphs)
1. **Problem/Task** - What are you addressing? Why is it interesting?
2. **Prior Work** - How have people attempted to solve it?
3. **Limitations** - What are the gaps? Why should people care?
4. **Your Approach** - What you do that overcomes limitations
5. **Benefits + Results** - How you verify benefits, key contributions (max 3)

### Experimental Evaluation
- Begin with: "The experiments are designed to answer the following research questions:"
- List 2-4 concise points
- Three subsections: Setting, Key Results, Analysis
- Tables and figures should be self-explanatory with captions

---

## Section-by-Section Update Plan

### 1. Abstract (Lines 152-168)

**Current State:** Claims 78% vs 30% vs 21% accuracy on Deepseek/Gemma

**Updates Required:**
- [ ] Update accuracy numbers from new 6-model results
- [ ] Update model names (Claude, GPT-4o-mini, Llama-405b, Gemini, Qwen, Ministral)
- [ ] Update statistical test results from new data
- [ ] Verify MI lower bound claim (currently 6%) matches new logistic regression results

### 2. Introduction (Lines 171-226)

**Current State:** Describes three-arm framework, cites old model results

**Updates Required:**
- [ ] Update empirical results paragraph with new numbers
- [ ] Update model list in findings
- [ ] Keep theoretical claims unchanged (Bayesian inference framework still applies)
- [ ] Verify contributions list still accurate

### 3. Preliminaries and Setup (Lines 237-272)

**Current State:** Defines task, modeling, random variables

**Updates Required:**
- [ ] No changes expected (theoretical framework unchanged)
- [ ] Verify notation consistency with experiments

### 4. Experiment: Three Arms Comparison (Lines 274-344)

**Current State:** Deepseek-Coder and Gemma-2 on arithmetic, DP, ILP

**Updates Required:**

#### 4.1 Tasks (Lines 288-294)
- [ ] Expand task list to include all 44 problem kinds
- [ ] Organize into 3 categories: Fine-grained (9), CLRS (30), NP-hard (5)
- [ ] Keep descriptions concise

#### 4.2 Models (Lines 297-298)
- [ ] Replace with 6 current models:
  - `anthropic/claude-haiku-4.5`
  - `google/gemini-2.5-flash`
  - `meta-llama/llama-3.1-405b-instruct`
  - `mistralai/ministral-14b-2512`
  - `openai/gpt-4o-mini`
  - `qwen/qwen-2.5-coder-32b-instruct`

#### 4.3 Figures
- [ ] **fig:main** - Update bar chart with new model results
- [ ] **fig:line** - Update accuracy vs difficulty plot
- [ ] **fig:pval** - Update p-value heatmap from new statistical tests

#### 4.4 Statistical Analysis (Lines 324-335)
- [ ] Recompute Friedman test statistics
- [ ] Recompute McNemar pairwise comparisons
- [ ] Update all p-values

### 5. Bayesian Explanation (Lines 345-434)

**Current State:** Theoretical proofs for Arm 1 < Arm 2 < Arm 3

**Updates Required:**
- [ ] No changes to proofs (theory is model-agnostic)
- [ ] Verify Lemma/Theorem statements still hold

### 6. Experimental Validation of Theory (Lines 435-533)

**Current State:** MI estimation with logistic regression, recovery analysis

**Updates Required:**

#### 6.1 MI Estimation (Lines 439-466)
- [ ] Update cross-entropy values from `exps_logistic/results/`
- [ ] Update K (number of classes) - currently 761, may change with 44 kinds
- [ ] Recalculate Bayes error improvement bound
- [ ] Update **fig:vlb** with new cross-entropy comparison

#### 6.2 Recovery Analysis (Lines 470-525)
- [ ] Update recovery rate statistics
- [ ] Update **fig:recovery_final**
- [ ] Update **fig:recovery_powerlaw**
- [ ] Update **fig:kde** distribution plot

### 7. Discussion (Lines 534-543)

**Updates Required:**
- [ ] Update model list in limitations
- [ ] Update task list description
- [ ] Keep broader impact unchanged

### 8. Conclusion (Lines 544-549)

**Updates Required:**
- [ ] Update accuracy numbers (78% vs 21%)
- [ ] Update task count (44 problem kinds across 3 categories)

---

## New Figures Required

Generate from `src/exps_performance/analysis.py` and `src/exps_logistic/notebooks/generate_plots.py`:

| Figure | Source | Description |
|--------|--------|-------------|
| fig1.png | Manual | Three-arms diagram (keep current) |
| main.png | analysis.py | Bar chart: 3 arms × 6 models |
| line.png | analysis.py | Accuracy vs hardness parameter |
| pval.png | analysis.py | McNemar p-value heatmap |
| vlb.png | generate_plots.py | Cross-entropy comparison |
| mi_v_acc.png | generate_plots.py | MI vs accuracy correlation |
| recovery_final.png | analysis.py | Recovery rate by task |
| recovery_powerlaw.png | analysis.py | Recovery vs difficulty |
| kde.png | analysis.py | Accuracy distribution KDE |

---

## Data Sources

### Performance Results
```
src/exps_performance/results/{model}_seed{seed}/tb/run_*/res.jsonl
```

### Logistic Regression Results
```
src/exps_logistic/results/
```

### Problem Kinds Reference
```python
# From src/exps_logistic/config.py
FG_KINDS = {"add", "sub", "mul", "lcs", "knap", "rod", "ilp_assign", "ilp_prod", "ilp_partition"}
CLRS_KINDS = {"activity_selector", "articulation_points", "bellman_ford", ...}  # 30 kinds
NPHARD_KINDS = {"edp", "gcp", "ksp", "spp", "tsp"}
```

---

## Implementation Workflow

### Phase 1: Data Collection
1. Run `src/exps_performance/analysis.py` to generate aggregate statistics
2. Run `src/exps_logistic/notebooks/generate_plots.py` to generate MI plots
3. Extract key numbers for paper

### Phase 2: Figure Generation
1. Generate all required figures
2. Copy to `Bayesian_Tool_Use/` directory
3. Verify figure quality and formatting

### Phase 3: Text Updates
1. Update Abstract with new numbers
2. Update Introduction results paragraph
3. Update Experiment section (tasks, models, figures, statistics)
4. Update Experimental Validation section (MI values, recovery rates)
5. Update Discussion and Conclusion

### Phase 4: Review
1. Use paper-reviewer agent for feedback
2. Follow academic-writing-cs skill for clarity
3. Verify all claims match experimental data

---

## Skills to Use

| Skill | Purpose |
|-------|---------|
| `paper-reviewer` | Review draft for quality and coherence |
| `academic-writing-cs` | Ensure academic writing standards |
| `latex-compilation` | Debug any LaTeX errors |
| `research-code-reviewer` | Verify code-methodology alignment |

---

## Checklist Before Submission

- [ ] All figures generated from current data
- [ ] All accuracy numbers verified against results
- [ ] Statistical tests recomputed
- [ ] MI bounds recalculated
- [ ] Model names updated throughout
- [ ] Task list expanded to 44 kinds
- [ ] Bibliography up to date
- [ ] LaTeX compiles without errors
- [ ] Paper-reviewer agent feedback addressed

---

## Notes

- **Exclude pretraining experiments** - Not in current src/ folder
- **Keep theoretical framework unchanged** - Bayesian inference + MI analysis still valid
- **Maintain paper length** - Expand task coverage concisely, don't inflate
