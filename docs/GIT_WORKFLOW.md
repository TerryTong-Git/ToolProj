# Git Workflow for Research Code Review

This document describes the automated Git workflow for validating code-methodology alignment in this research codebase.

## Overview

The workflow ensures that implementation code stays aligned with methodology documentation through:

1. **GitHub Actions** - Automated review on every push/PR to master
2. **Pre-push hooks** - Local validation before pushing
3. **Code mapping** - Explicit relationships between docs and code

## Workflow Architecture

```
                    Local Development
                           |
                           v
                    [pre-push hook]
                    validate_code_mapping.py
                           |
                           v (if passes)
                      git push
                           |
                           v
                    GitHub Actions
                    code-review.yml
                           |
                    +------+------+
                    |             |
                    v             v
               PR Comment    Artifact Upload
               (if PR)       review-report.md
                    |
                    v
              Status Check
              (PASS/WARN/FAIL)
```

## Components

### 1. GitHub Actions Workflow

**File**: `.github/workflows/code-review.yml`

**Triggers**:
- Push to `master` or `main` branches
- Pull requests to `master` or `main` branches

**Monitored paths**:
- `src/**/*.py` - All Python source files
- `docs/**/*.md` - All markdown documentation
- `docs/code_mapping.json` - The mapping configuration

**What it does**:
1. Detects changed files in the commit/PR
2. Runs Claude Code Action to review alignment
3. Generates `review-report.md` artifact
4. Posts review as PR comment (for PRs)
5. Creates GitHub issue on failure (for push events)
6. Sets status check (PASS/WARN/FAIL)

### 2. Local Pre-push Validation

**File**: `scripts/validate_code_mapping.py`

**Usage**:
```bash
# Manual run
python scripts/validate_code_mapping.py

# With verbose output
python scripts/validate_code_mapping.py --verbose

# Strict mode (warnings become errors)
python scripts/validate_code_mapping.py --strict
```

**Automatic via pre-commit**:
The hook runs automatically before `git push` when pre-commit is installed:
```bash
# Install pre-commit hooks
pre-commit install --hook-type pre-push
```

### 3. Code-Documentation Mapping

**File**: `docs/code_mapping.json`

Defines explicit relationships between methodology docs and implementation:

```json
{
  "mappings": [
    {
      "methodology": "docs/experiment_methodologies/01_embedding_ablation.md",
      "implementation": ["src/exps_logistic/featurizer.py"],
      "key_functions": ["Featurizer", "transform", "build_featurizer"],
      "notes": "Description of the mapping",
      "status": "active"  // or "planned"
    }
  ]
}
```

**Fields**:
- `methodology`: Path to the methodology documentation
- `implementation`: List of implementation file paths
- `key_functions`: Required functions/classes that must exist
- `notes`: Optional description
- `status`: "active" (validated) or "planned" (not yet implemented)

## Setup Instructions

### 1. GitHub Repository Secrets

Add the following secret in your GitHub repository settings:

- `ANTHROPIC_API_KEY` - Your Anthropic API key for Claude

**Path**: Repository > Settings > Secrets and variables > Actions > New repository secret

### 2. Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks
pre-commit install
pre-commit install --hook-type pre-push
```

### 3. Branch Protection (Recommended)

Configure branch protection for `master`:

1. Go to Repository > Settings > Branches
2. Add rule for `master`
3. Enable:
   - Require status checks to pass before merging
   - Select "Code-Methodology Alignment Check"
   - Require branches to be up to date

## Review Status Meanings

| Status | Meaning | Action Required |
|--------|---------|-----------------|
| PASS | All code aligns with documentation | None - ready to merge |
| WARN | Minor inconsistencies found | Review warnings, consider fixing |
| FAIL | Critical misalignment detected | Must fix before merge |

## Common Scenarios

### Adding a New Experiment

1. Create methodology doc in `docs/experiment_methodologies/`
2. Add mapping to `docs/code_mapping.json`
3. Implement the code in `src/`
4. Run local validation: `python scripts/validate_code_mapping.py`
5. Push changes

### Modifying Existing Code

1. Make code changes
2. If function names change, update `code_mapping.json`
3. If algorithm changes, update methodology doc
4. Run local validation
5. Push changes

### Handling Review Failures

1. Check the review report artifact in GitHub Actions
2. Identify misalignments
3. Either:
   - Update code to match documentation
   - Update documentation to match code
   - Update `code_mapping.json` to reflect current state
4. Re-push

## Validation Rules

The code mapping validation checks:

1. **Function Existence**: All `key_functions` must exist in implementation files
2. **File Existence**: All referenced files must exist
3. **Methodology Coverage**: Each methodology doc has corresponding implementation

The GitHub Actions review additionally checks:

4. **Signature Consistency**: Function parameters match documentation
5. **Algorithm Steps**: Implementation follows documented steps
6. **Output Format**: Return structures match specifications

## Troubleshooting

### Pre-push hook not running

```bash
# Reinstall hooks
pre-commit uninstall --hook-type pre-push
pre-commit install --hook-type pre-push
```

### GitHub Action not triggering

Check that:
- Changed files match the path patterns in the workflow
- You're pushing to `master` or `main` branch
- The workflow file is on the default branch

### False positives in validation

Add `"status": "planned"` to mappings that aren't yet implemented but are documented for future work.

## Files Reference

| File | Purpose |
|------|---------|
| `.github/workflows/code-review.yml` | GitHub Actions workflow |
| `.pre-commit-config.yaml` | Pre-commit hook configuration |
| `scripts/validate_code_mapping.py` | Local validation script |
| `docs/code_mapping.json` | Code-documentation mapping |
| `docs/experiment_methodologies/` | Methodology documentation |
