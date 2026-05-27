# Repository Reorganization Plan

Authoritative plan: `../docs/repo_reorganization_plan.md`.

This source-local placeholder is retained because it already existed beside
`src/goal.md`. `src/goal.md` was inspected on 2026-05-26 and is currently an
untracked empty file, so the plan falls back to the active GitHub repository
organization guide plus live ToolProj PR evidence.

High-level target:

- Use domain package names:
  - `src/reasoning_benchmark`
  - `src/translation_discrimination`
  - `src/translation_additivity`
- Remove the unused logistic/MI experiment surface from this paper-focused
  refactor instead of repairing or renaming `src/exps_logistic`.
- Remove temporary compatibility wrappers after landed commands and old import
  paths migrate to canonical packages.
- Split overloaded files and interfaces inside `reasoning_benchmark`, including
  `main.py`, `arms.py`, `dataset.py`, `logger.py`, `llm.py`, `core/executor.py`,
  and `utils.py`.
- Treat merged PRs #45-#57 on `origin/master` as the current behavioral
  baseline; there are no open rebuttal/paper-reproduction PRs to retarget.
- Start future refactor work from fetched `origin/master`, not from the local
  `feat/openrouter-structured-reasoning` branch whose upstream is gone.
- Preserve the merged analyzer/report tests, compact outputs, reproduction CLI,
  and paper reproduction docs while package names change.
- Make paper reproduction generated-results-first: run deterministic >=5%
  experiment shards for each paper table and figure source, then regenerate and
  exactly verify tables and normalized figure source data against
  original-branch gold fixtures.
- This branch adds the generated-results gold manifest at
  `tests/fixtures/paper_reproduction/gold_manifest.json`; current
  `origin/master` does not contain it.
- Preserve dirty local work and avoid destructive branch rewrites unless the
  repo owner explicitly approves them.

The refactor branch now contains the canonical implementation moves for
`reasoning_benchmark`, `translation_discrimination`, and
`translation_additivity`. `src.exps_control_again` and `src.exps_functional`
are no longer import surfaces. `src/exps_performance` remains only as a raw
benchmark artifact root for results, notebooks, and recovery figures; canonical
modules reach those retained files through `src.reasoning_benchmark.artifact_paths`.
Do not move those generated artifacts in this interface-refactor branch; the
paper-reproduction gate now verifies normalized figure-source data separately.
