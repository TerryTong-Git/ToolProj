# ToolProj Repository Reorganization Plan

## Summary

This plan records the staged reorganization for ToolProj. The active refactor
branch is the first domain-rename PR against `origin/master`: it moves the
reasoning benchmark, translation discrimination, and translation additivity
implementations into canonical packages while retaining temporary
`src/exps_*` compatibility wrappers during migration, then removes those
wrappers after canonical docs and tests are in place. It also adds the
deterministic paper-reproduction 5% generated-results manifest gate. It does
not retarget merged PRs or rewrite published branch history.

`src/goal.md` was inspected on 2026-05-26 and is currently an untracked empty
file, so this plan follows the active GitHub repository organization guide and
the live ToolProj state.

The main correction from the first version of this plan is naming depth:
`exps_performance`, `exps_control_again`, and `exps_functional` are not good
long-term package names. The refactor should use domain names, split overloaded
interfaces, and remove temporary compatibility wrappers once docs and tests
move to canonical imports.

Canonical package targets:

- `src/exps_performance` to `src/reasoning_benchmark`
- `src/exps_control_again` to `src/translation_discrimination`
- `src/exps_functional` to `src/translation_additivity`
- Remove the unused logistic/MI experiment surface from this paper-focused
  refactor.

## Current Evidence

Repository:

- Path: `/Users/terrytong/Documents/Research Projects/ICML 2026 - Algorithmic Reasoning/ToolProj`
- Origin: `https://github.com/TerryTong-Git/ToolProj`
- Current local branch: `feat/openrouter-structured-reasoning`
- Upstream for the current local branch is gone after #45 merged.
- Current fetched `origin/master`: `ddeeee9` (`Make figure dry run environment-free`)
- Refactor worktree branch: `chore/rebuttal-review-foundation`, based on
  `origin/master`.
- `src/reorganize.md` started as an empty placeholder and now points to this
  authoritative plan.
- The working tree is dirty and includes local uncommitted work. Do not reset,
  rebase, force-push, delete untracked artifacts, or sweep unrelated files into
  any cleanup branch without explicit approval.

Observed naming debt:

- `exps_performance` currently owns benchmark execution, task loading, LLM
  clients, code execution, checkpointing, result records, analysis, figures,
  notebooks, scripts, and checked-in results. That is too many responsibilities
  for one flat package.
- `exps_control_again` encodes iteration history, not domain meaning. The code
  is about source-label discrimination, translator separability, embedding
  separability, prompts, and reports.
- `exps_functional` is vague. The code is specifically about information
  additivity and native-vs-translated additivity.
- The old logistic/MI experiment package is not present in current `src/` and
  is not used by the paper. Remove its stale tests/docs references instead of
  repairing or renaming it in this refactor.

Merged PR topology, checked on 2026-05-27:

| PR | Branch | Merge commit | Landed responsibility |
| --- | --- | --- | --- |
| #45 | `feat/openrouter-structured-reasoning` | `5d0cd3b` | Structured frontier runtime and reproduction |
| #46 | `rebuttal/code-failure-distribution` | `3b977ab` | Code-failure table |
| #47 | `rebuttal/functional-shot-ablation` | `38c55b0` | Functional shot-ablation table |
| #48 | `rebuttal/route-accuracy-tables` | `b209b84` | Route-accuracy tables |
| #49 | `rebuttal/rlm-results` | `e8e2ae6` | RLM runtime plus RLM rebuttal result |
| #50 | `rebuttal/coding-model-table` | `6feacd9` | Coding-model table |
| #51 | `rebuttal/frontier-nopatch-table` | `8a4fe14` | Frontier no-patching table |
| #52 | `rebuttal/paper-reproduction-readme` | `0797d8d` | Paper reproduction guide and driver |
| #53 | `rebuttal/paper-reproduction-refactor` | `766e79c` | Python reproduction CLI |
| #54 | `rebuttal/paper-reproduction-interface-cleanup` | `da62095` | Explicit reproduction path flags |
| #55 | `rebuttal/paper-reproduction-targeted-list` | `5e69a81` | Targeted command listing |
| #56 | `rebuttal/paper-reproduction-recovery-flag-message` | `5bbd33a` | Recovery notebook flag hint |
| #57 | `rebuttal/paper-reproduction-figure-dry-run` | `ddeeee9` | Environment-free figure dry run |

Live status check on 2026-05-27:

- `gh pr view 45..57` reports `MERGED`.
- `gh pr list --state open` returns no open PRs.
- `rebuttal/paper-reproduction-stack-base` is no longer an active review
  blocker because #52 merged to `master` after #45-#51 landed.
- `origin/master` contains `PAPER_REPRODUCTION.md` and
  `scripts/reproduce_paper.py`, but it does not contain
  `tests/fixtures/paper_reproduction/gold_manifest.json`. This refactor branch
  adds that manifest and the generated-results verification path.

## Target Shape

Target package layout:

```text
ToolProj/
  README.md
  pyproject.toml
  docs/
    repo_reorganization_plan.md
    architecture.md
    adr/
  scripts/
    reproduce_paper.py
  src/
    reasoning_benchmark/
      cli.py
      runner.py
      run_config.py
      reasoning_strategies.py
      task_sets.py
      records.py
      checkpoints.py
      result_loading.py
      llm_clients.py
      execution/
        python_executor.py
      problems/
      clrs/
      analysis/
      reproductions/
        frontier/
          run.py
          summarize.py
          fixtures/
          provenance/
      scripts/
      results/
        analysis/
    translation_discrimination/
      cli.py
      source_label.py
      analysis/
      experiments/
      prompts/
      reports/
      results/
    translation_additivity/
      cli.py
      information_additivity.py
      native_translation_additivity.py
      prompts/
      reports/
      results/
  tests/
    unit/
    integration/
  results/
    rebuttal/
```

Compatibility policy:

- Temporary `src/exps_*` wrappers are allowed only during migration.
- New docs, tests, and PRs must use canonical names.
- Remove wrappers after canonical package imports have replaced old paths.
- Do not migrate package root `src` to a proper root such as `toolproj` in this
  wave. Plan that separately after domain package names stabilize.

Generated-output policy:

- Compact, validated provenance outputs may stay checked in when they are
  required to reproduce a paper/rebuttal table.
- Raw logs, scratch JSONL files, debug directories, notebooks, vendor trees, and
  large run artifacts must stay out of ordinary rename/refactor PRs.
- Update `.gitignore` only when a repeated generated-output pattern is observed.

## Staged Migration

### 1. Foundation And Domain Rename PR

Create a base refactor PR from `origin/master`, for example:

```text
chore/rebuttal-review-foundation
```

Responsibilities:

- Add canonical package directories.
- Replace old `src/exps_*` imports and CLI entrypoints with canonical package
  imports and module entrypoints.
- Move implementation modules into domain packages under clearer file and
  module names.
- Move duplicated lint/test/import plumbing into one place where needed.
- Preserve current behavior.

Do not retarget #45-#57 or rewrite the stale local
`feat/openrouter-structured-reasoning` branch.

Implementation status in `chore/rebuttal-review-foundation`:

- `src/reasoning_benchmark`, `src/translation_discrimination`, and
  `src/translation_additivity` exist as canonical implementation packages.
- `src/exps_control_again` and `src/exps_functional` no longer exist as import
  surfaces. `src/exps_performance` remains only as a retained raw benchmark
  artifact root for results, notebooks, and legacy generated figures.
- README, paper-reproduction commands, and tests now prefer canonical imports
  and `python -m src.<domain_package>...` module entrypoints.
- Existing checked-in raw results, notebooks, and recovery figures still live
  under `src/exps_performance` to avoid sweeping generated artifacts into the
  interface-refactor slice. Compact discrimination/additivity result artifacts
  moved with their canonical domain packages.
- `tests/integration/test_repository_reorganization_foundation.py` verifies the
  canonical interfaces, removed legacy import surfaces, and CLI entrypoints.

Validation:

- `rtk gh pr diff <domain-refactor-pr> --name-only`
- `rtk gh pr view <domain-refactor-pr> --json headRefName,baseRefName,mergeStateStatus,mergeable`
- Import canonical packages and confirm old package wrappers are gone.
- `rtk git diff --check`
- `rtk uv run pytest -m "not slow" -q`

### 2. Reasoning Benchmark Rename And Interface Split

Move `src/exps_performance` to `src/reasoning_benchmark`.

Public interface targets:

- `main.py` to `runner.py` plus `cli.py`
- `Args` to `BenchmarkRunConfig`
- `arms.py` to `reasoning_strategies.py`
- `Arm1` to `NaturalLanguageReasoning`
- `Arm2` to `CodeSimulationReasoning`
- `Arm3` to `CodeExecutionReasoning`
- `Arm4` to `ControlledSimulationReasoning`
- `dataset.py` to `task_sets.py`
- `make_dataset()` to `load_benchmark_tasks()`
- `logger.py` split into `records.py`, `checkpoints.py`, and `result_loading.py`
- `llm.py` to `llm_clients.py`
- `llm()` to `build_llm_client()`
- `core/executor.py` to `execution/python_executor.py`
- `utils.py` split into `text_parsing.py`, `randomness.py`, and `dimacs.py`

Frontier reproduction target:

```text
src/reasoning_benchmark/reproductions/frontier/
  run.py
  summarize.py
  fixtures/seed1_reference.jsonl
  provenance/seed1.json
```

The old `src/exps_performance/scripts/reproduce_frontier_structured.py` path
should remain a thin wrapper during migration.

Implementation status in `chore/rebuttal-review-foundation`:

- Benchmark implementation modules have moved into `src/reasoning_benchmark`
  under the canonical interface names:
  - `main.py` to `runner.py` plus `cli.py`
  - `arms.py` to `reasoning_strategies.py`
  - `dataset.py` to `task_sets.py`
  - `logger.py` to `records.py` with `checkpoints.py` and
    `result_loading.py` as narrow public interfaces
  - `llm.py` to `llm_clients.py`
  - `core/executor.py` to `execution/python_executor.py`
  - RLM execution helpers to `execution/rlm_executor.py` and
    `execution/rlm_worker.py`
  - parser/random/DIMACS helpers are exposed through `text_parsing.py`,
    `randomness.py`, and `dimacs.py`
- `src/exps_performance` no longer contains compatibility wrappers for imports
  or old script entrypoints. It remains only as the retained raw benchmark
  artifact root.
- Frontier structured reproduction now lives under
  `src/reasoning_benchmark/reproductions/frontier/` with `run.py`,
  `summarize.py`, `fixtures/seed1_reference.jsonl`, and
  `provenance/seed1.json`.
- Raw checked-in `res.jsonl` trees, notebook-backed recovery artifacts, and
  generated figures remain under `src/exps_performance` until the generated
  artifact cleanup/gold-fixture gate is handled separately.
- Canonical modules access those retained artifacts through
  `src.reasoning_benchmark.artifact_paths` instead of scattering hard-coded
  legacy package paths through domain code.
- Tests and docs prefer canonical `src.reasoning_benchmark` imports and module
  entrypoints.

Validation:

- `rtk uv run python -m src.reasoning_benchmark.cli --help`
- `rtk uv run python -m src.reasoning_benchmark.reproductions.frontier.run --help`
- Reasoning benchmark unit tests for task loading, LLM clients, checkpoints,
  reasoning strategies, code execution, and frontier reproduction.
- Existing #45 analyzer/reproduction tests after import updates.
- `rtk rg "src\\.exps_performance|src/exps_performance" src tests README.md`
  should return only the retained artifact-root boundary and migration notes.
- `rtk uv run pytest -q`
- `rtk uv run ruff check src/reasoning_benchmark src/exps_performance tests scripts/reproduce_paper.py`

### 3. Translation Discrimination Rename

Move `src/exps_control_again` to `src/translation_discrimination`.

Responsibilities:

- Source-label classification.
- Translator separability experiments.
- Embedding separability experiments.
- Discrimination prompts.
- Reports and plotting for discrimination outputs.

Interface targets:

- `run_source_discrimination.py` becomes `cli.py` plus `source_label.py`.
- `functional_similarity_experiment.py` becomes
  `experiments/functional_similarity.py`.
- Embedding and classifier scripts become explicit report/analysis modules
  instead of a loose scripts collection.

Implementation status in `chore/rebuttal-review-foundation`:

- Source-label classification moved to
  `src/translation_discrimination/source_label.py`, with
  `src/translation_discrimination/cli.py` as the command entrypoint.
- Translator and source-discrimination experiments moved under
  `src/translation_discrimination/experiments/`.
- Embedding/classifier analysis moved under
  `src/translation_discrimination/analysis/`.
- Plot and report generation moved under
  `src/translation_discrimination/reports/`.
- Prompts and compact checked-in result artifacts moved under
  `src/translation_discrimination/prompts/` and
  `src/translation_discrimination/results/`.
- `src/exps_control_again` no longer exists as an import surface.
- Report modules that previously generated figures at import time now expose
  `main()` and only write outputs when executed as scripts/modules.

Validation:

- `rtk uv run python -m src.translation_discrimination.cli --help`
- Source-label classifier tests.
- Prompt-path tests.
- Report-generation tests.
- `rtk rg "src\\.exps_control_again|src/exps_control_again|exps_control_again" src tests README.md`
  should return only migration notes.

### 4. Translation Additivity Rename

Move `src/exps_functional` to `src/translation_additivity`.

Responsibilities:

- Information-additivity experiment: question, native reasoning, code, and
  combined conditions.
- Native-vs-translated additivity experiment.
- Translation-additivity reports and plots.

Interface targets:

- `run_additivity.py` becomes `information_additivity.py` plus `cli.py`.
- `run_translation_additivity.py` becomes `native_translation_additivity.py`.
- Plot scripts become report modules under `reports/`.

Implementation status in `chore/rebuttal-review-foundation`:

- Information-additivity execution moved to
  `src/translation_additivity/information_additivity.py`.
- Native-vs-translated additivity execution moved to
  `src/translation_additivity/native_translation_additivity.py`.
- Shot-ablation analysis and additivity plots moved under
  `src/translation_additivity/reports/`.
- The shot-ablation shell runner moved to
  `src/translation_additivity/scripts/run_shot_ablation.sh`.
- Prompts and compact checked-in result artifacts moved under
  `src/translation_additivity/prompts/` and
  `src/translation_additivity/results/`.
- `src/exps_functional` no longer exists as an import surface.
- Report modules that previously generated figures at import time now expose
  `main()` and only write outputs when executed as scripts/modules.

Validation:

- `rtk uv run python -m src.translation_additivity.cli --help`
- Tests for information-additivity and native-vs-translated additivity paths.
- Report-generation tests.
- `rtk rg "src\\.exps_functional|src/exps_functional|exps_functional" src tests README.md`
  should return only migration notes.

### 5. Remove Unused Logistic/MI Experiment Surface

Do not restore or rename `src/exps_logistic` in the domain rename wave.

Paper-scope decision:

- Logistic/MI experiments are not used in the paper.
- The current `src/` listing does not show an `exps_logistic` package.
- Remove stale README/test references rather than adding compatibility wrappers
  or a new canonical package.

Implementation status in `chore/rebuttal-review-foundation`:

- `tests/logistic/` was removed.
- `README.md` no longer advertises the legacy logistic test surface.
- `pyproject.toml` no longer carries a stale `src/exps_logistic` mypy
  exclusion.

### 6. Preserve The Merged Rebuttal Baseline

Do not retarget #45-#57. They are merged and now define the behavior that the
domain refactor must preserve.

Post-merge handling:

- Start new refactor work from current `origin/master`, not from the local
  `feat/openrouter-structured-reasoning` branch whose upstream is gone.
- Treat #45-#57 tests, CLI behavior, checked-in compact outputs, and docs as
  regression baselines.
- When renaming packages, migrate the merged analyzer/report tests with the
  code and remove compatibility wrappers after old commands are retired.
- The #49 RLM runtime/reporting ownership remains a domain-boundary issue to
  resolve during the package split, but it is no longer an open PR split.
- Paper reproduction CLI polish from #54-#57 has landed. Future CLI changes
  should be batched as one follow-up instead of recreating a long stack.

### 7. Paper Reproduction Generated-Results Gate

Paper reproduction must generate results through experiments before it
regenerates paper tables, figure data, or copied paper assets. The merged
#52-#57 stack is useful as the current CLI/docs baseline, but it is not the
generated-results gate because `origin/master` still lacks an original-branch
gold fixture manifest.

Required behavior:

- Every paper table or figure source must have a manifest entry that names the
  experiment runner, deterministic shard selector, generated artifact,
  analyzer, and exact comparison target.
- The shard for each result source must cover at least 5% of that source's data.
  Do not use one global 5% sample that can skip smaller result families.
- The reproduction driver must run the real experiment code path with
  deterministic recorded or fake model backends before analyzer commands run.
- Exact CI validation must compare regenerated shard outputs against gold files
  produced by the original source commits/branches that generated the paper
  results, after stable normalization of ordering, line endings, float
  formatting, and volatile metadata.
- Live external model APIs are outside the exact gate because provider behavior,
  latency, and availability can drift. Real-provider reruns may remain as a
  separate slow/manual workflow, but they are not the required exact 5%
  validation path.

Target CLI surface:

```text
python scripts/reproduce_paper.py experiments --shard 5pct --output-dir <dir>
python scripts/reproduce_paper.py tables --input-dir <dir> --output-dir <dir>
python scripts/reproduce_paper.py figures --input-dir <dir> --output-dir <dir>
python scripts/reproduce_paper.py verify-5pct --output-dir <dir>
python scripts/reproduce_paper.py all --shard 5pct --require-generated-results
```

Implementation status in `chore/rebuttal-review-foundation`:

- `scripts/reproduce_paper.py experiments --shard 5pct` now generates
  deterministic recorded experiment shards under
  `<output-dir>/experiments/`.
- `tests/fixtures/paper_reproduction/gold_manifest.json` records table and
  figure-source result sources with source PR, branch, commit, runner, analyzer,
  generated artifact path, row coverage, SHA-256, and exact gold output targets.
- `verify-5pct` compares the observed manifest and artifact hashes exactly
  against the gold manifest, rejects missing or under-coverage artifacts, reruns
  analyzers and figure-source extractors against the generated shards, and
  byte-compares the regenerated markdown or CSV outputs against
  `tests/fixtures/paper_reproduction/gold_outputs/`.
- `tables --input-dir <dir>` feeds generated artifacts into the route-accuracy,
  translation-additivity, RLM, coding-model, code-failure, frontier-no-patch,
  and structured sim/code overlap analyzers.
- `all --require-generated-results` runs generation and verification before
  table analyzers. Its dry run remains inspect-only.
- Figure commands still regenerate deterministic checked-in figure scripts.
  When generated artifacts are required, the driver also writes normalized
  figure-source CSVs for reasoning benchmark figures, judge discrimination,
  native-vs-translated scatter, translation additivity, and recovery-vs-digits.
  `verify-5pct` byte-compares those source CSVs against gold fixtures.
- The benchmark raw artifact root remains centralized in
  `src.reasoning_benchmark.artifact_paths` while raw result trees and recovery
  notebooks stay out of the package rename diff.

Validation:

- `experiments --dry-run --shard 5pct` lists experiment-generation commands,
  not only analyzer commands.
- `verify-5pct` fails if required generated artifacts are missing or if outputs
  were copied from checked-in final artifacts instead of regenerated.
- A manifest such as `tests/fixtures/paper_reproduction/gold_manifest.json`
  records the source PR, branch, commit, input fixture, and gold output for each
  result source.
- Existing "five percent validation" wording must not describe fixture-only or
  analyzer-only tests.
- Figure verification compares generated figure source data exactly; pixel
  equality remains optional because plotting libraries can drift across
  environments.
- Live validation run on 2026-05-27:
  - `rtk uv run python scripts/reproduce_paper.py experiments --shard 5pct --output-dir /tmp/toolproj-paper-gate`
  - `rtk uv run python scripts/reproduce_paper.py verify-5pct --output-dir /tmp/toolproj-paper-gate`
  - `rtk uv run python scripts/reproduce_paper.py tables --input-dir /tmp/toolproj-paper-gate --output-dir /tmp/toolproj-paper-tables`
- Figure-source-data gate tightened on 2026-05-27:
  - `rtk uv run python scripts/reproduce_paper.py experiments --shard 5pct --output-dir /tmp/toolproj-paper-gate-figures`
  - `rtk uv run python scripts/reproduce_paper.py verify-5pct --output-dir /tmp/toolproj-paper-gate-figures`
  - Verification reported `verified 13 generated 5% artifacts` and
    `verified 17 generated 5% output files`.

### 8. Cleanup

- Old `src/exps_*` wrappers were removed after canonical package imports
  replaced old paths.
- README examples use canonical commands.
- Stale comments and migration notes were updated where they referred to old
  package import surfaces.
- The full default test suite is part of final verification.

## Branch Hygiene

Default to non-destructive updates:

- Preserve published branch history unless the owner explicitly asks for clean
  rebases or force pushes.
- Prefer follow-up commits, PR retargeting, or new split branches.
- Do not delete or rewrite local untracked artifacts without an explicit cleanup
  step.
- Before each retarget or split, record the exact source branch and commit.

Before any branch operation, capture:

- `rtk git status --short --branch`
- `rtk git branch --show-current`
- `rtk git rev-parse HEAD`
- `rtk gh pr view <number> --json headRefName,baseRefName,files,commits`

After each branch operation, run:

- `rtk git status --short --branch`
- `rtk gh pr diff <pr-number> --name-only`
- `rtk gh pr view <pr-number> --json headRefName,baseRefName,mergeStateStatus,mergeable`

The operation is not complete if unrelated dirty files were swept into the
branch or if the PR includes files outside its stated responsibility. For new
work, branch from fetched `origin/master` unless explicitly preserving a local
experiment branch.

## Checker Loop

Run this loop after each proposed package rename, PR split, retarget, or
consolidation:

1. Confirm the PR base/head and file list with `gh`.
2. Classify every changed file as foundation plumbing, runtime, analyzer/report,
   docs/CLI, compact provenance, compatibility shim, or unrelated.
3. Check domain ownership, one responsibility per PR, clear names, thin scripts,
   mirrored tests, and generated-output policy.
4. Reject broad chronology-based groupings, vague buckets, invisible stack
   bases, or destructive branch rewriting.
5. Run the targeted tests or regeneration command matching the PR's
   responsibility.

## Completion Checklist

The reorganization is complete only when all of these are true:

- The plan explicitly states that `src/goal.md` was empty at planning time.
- Canonical domain packages exist and are used by new docs/tests.
- Temporary wrappers are present only during migration and removed in cleanup.
- `exps_control_again` no longer exists as a canonical package name.
- `exps_functional` no longer exists as a canonical package name.
- `exps_performance` no longer exists as a canonical package name.
- Historical PRs #45-#57 are documented as merged baseline work, and there are
  no open rebuttal/paper-reproduction PRs left to retarget.
- RLM runtime implementation and RLM result reporting have clear package
  ownership after the domain split.
- Paper reproduction CLI polish from #54-#57 is treated as landed baseline, and
  any future CLI polish is grouped into one follow-up.
- Paper reproduction has a manifest mapping each result source to its runner,
  deterministic 5% shard selector, generated artifact, analyzer, original
  source branch/commit, and exact comparison target.
- `verify-5pct` fails when paper outputs are copied from checked-in final
  artifacts instead of regenerated through experiment shards.
- Existing "five percent validation" wording is not fixture-only or
  analyzer-only.
- Analyzer PRs regenerate their committed compact outputs.
- Tests mirror the behavior they validate.
- Raw logs, raw `res.jsonl` files, debug outputs, notebooks, and vendor trees
  are excluded from ordinary rename/refactor/result PRs.
