# exps_logistic Data Validation Design

**Created:** 2026-01-10
**Status:** Ready for Implementation
**Priority:** High

## Overview

Add comprehensive data validation to exps_logistic to ensure:
1. Data freshness - experiments use latest exps_performance results
2. Data completeness - all expected models/seeds/kinds are present
3. Parsing robustness - catch silent gamma label extraction failures
4. Result reproducibility - same inputs produce same outputs

**Failure Mode:** Fail-fast (raise error if thresholds exceeded)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│  1. DataFreshnessValidator                                      │
│     • Check source JSONL timestamps vs expected                 │
│     • Warn if data is stale (>24h since last update)            │
│     • Compare git hash of source data vs last known good        │
│                                                                 │
│  2. DataCompletenessValidator                                   │
│     • Verify all expected model/seed/kind combinations exist    │
│     • Report missing vs expected coverage matrix                │
│     • Fail if critical combinations are missing                 │
│                                                                 │
│  3. GammaParsingValidator                                       │
│     • Track parse success/failure rates per kind                │
│     • Fail if failure rate > threshold (default 20%)            │
│     • Log sample prompts that failed parsing with diagnostics   │
│                                                                 │
│  4. ReproducibilityValidator                                    │
│     • Hash inputs (data + config) to create fingerprint         │
│     • Compare against previous run fingerprints                 │
│     • Warn if inputs changed but outputs weren't regenerated    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    (If all pass) Run Experiment
```

---

## Component Details

### 1. DataFreshnessValidator

**Purpose:** Ensure logistic experiments use the latest exps_performance results, not stale cached data.

**Implementation:**

```python
# src/exps_logistic/validators/freshness.py

@dataclass
class FreshnessReport:
    source_files: list[Path]
    oldest_file: Path
    oldest_mtime: datetime
    staleness_hours: float
    is_stale: bool  # True if > threshold
    git_hash: str | None  # Hash of results dir if in git

def validate_freshness(
    results_dir: Path,
    max_staleness_hours: float = 24.0,
) -> FreshnessReport:
    """Check if source data is fresh."""
    ...
```

**Checks:**
- [ ] Find all `res.jsonl` files in results_dir
- [ ] Get mtime of oldest file
- [ ] Compare against threshold (default 24h)
- [ ] Optionally check git status of results dir
- [ ] Return structured report

**CLI Flag:** `--max-staleness-hours 24`

**Failure Behavior:** Warning only (data can be intentionally old for reproducibility)

---

### 2. DataCompletenessValidator

**Purpose:** Verify all expected model/seed/kind combinations are present.

**Implementation:**

```python
# src/exps_logistic/validators/completeness.py

@dataclass
class CompletenessReport:
    expected_models: set[str]
    found_models: set[str]
    missing_models: set[str]

    expected_seeds: set[int]
    found_seeds: set[int]
    missing_seeds: set[int]

    expected_kinds: set[str]
    found_kinds: set[str]
    missing_kinds: set[str]

    coverage_matrix: pd.DataFrame  # model x seed x kind
    coverage_pct: float
    is_complete: bool

def validate_completeness(
    df: pd.DataFrame,
    expected_models: set[str] | None = None,
    expected_seeds: set[int] | None = None,
    expected_kinds: str = "extended",  # preset name
    min_coverage_pct: float = 0.8,
) -> CompletenessReport:
    """Check data coverage against expectations."""
    ...
```

**Checks:**
- [ ] Compare found models against expected (from config or auto-detect)
- [ ] Compare found seeds against expected
- [ ] Compare found kinds against preset (fg, clrs, nphard, extended)
- [ ] Build coverage matrix showing gaps
- [ ] Fail if coverage below threshold

**CLI Flags:**
- `--expected-models model1,model2,...` (optional, auto-detect if not provided)
- `--expected-seeds 0,1,2` (optional)
- `--min-coverage-pct 0.8`

**Failure Behavior:** Fail if coverage < threshold

---

### 3. GammaParsingValidator

**Purpose:** Catch when gamma label extraction fails silently (falls back to 'bNA').

**Implementation:**

```python
# src/exps_logistic/validators/parsing.py

@dataclass
class ParseFailure:
    kind: str
    prompt: str
    expected_pattern: str
    error_reason: str

@dataclass
class ParsingReport:
    total_samples: int
    parsed_ok: int
    parsed_bna: int  # Fallback to "bNA"
    failure_rate: float

    failures_by_kind: dict[str, int]
    sample_failures: list[ParseFailure]  # First N failures per kind

    is_valid: bool  # True if failure_rate <= threshold

def validate_parsing(
    df: pd.DataFrame,
    max_failure_rate: float = 0.2,
    samples_per_kind: int = 3,
) -> ParsingReport:
    """Check gamma parsing success rates."""
    ...
```

**Checks:**
- [ ] Count samples where gamma ends with "|bNA"
- [ ] Calculate failure rate per kind
- [ ] Capture sample failed prompts for debugging
- [ ] Fail if overall failure rate > threshold

**CLI Flags:**
- `--max-parse-failure-rate 0.2`
- `--parse-samples-per-kind 3` (for diagnostics)

**Failure Behavior:** Fail if failure_rate > threshold

**Diagnostic Output:**
```
Parsing Validation FAILED
=========================
Overall failure rate: 34.2% (threshold: 20.0%)

Failures by kind:
  knap: 45.0% (90/200 samples)
  rod:  28.3% (17/60 samples)
  ilp_assign: 22.1% (44/199 samples)

Sample failures (knap):
  1. Prompt: "You have a knapsack with capacity..."
     Expected pattern: r"capacity\s*[:=]?\s*(\d+)"
     Error: No capacity value found in prompt

  2. Prompt: "Given items with weights [3,5,7]..."
     Expected pattern: r"weights?\s*[:=]?\s*\[([^\]]+)\]"
     Error: Weights found but no capacity
```

---

### 4. ReproducibilityValidator

**Purpose:** Ensure same inputs produce same outputs across runs.

**Implementation:**

```python
# src/exps_logistic/validators/reproducibility.py

@dataclass
class Fingerprint:
    data_hash: str  # SHA256 of sorted input data
    config_hash: str  # SHA256 of relevant config
    combined_hash: str
    created_at: datetime

@dataclass
class ReproducibilityReport:
    current_fingerprint: Fingerprint
    previous_fingerprint: Fingerprint | None
    inputs_changed: bool
    outputs_exist: bool
    outputs_match_inputs: bool

    is_valid: bool

def validate_reproducibility(
    df: pd.DataFrame,
    config: Config,
    fingerprint_dir: Path,
) -> ReproducibilityReport:
    """Check if inputs match previous run fingerprints."""
    ...
```

**Checks:**
- [ ] Hash input data (sorted by deterministic key)
- [ ] Hash relevant config options
- [ ] Compare against stored fingerprint from last run
- [ ] Check if outputs exist for current fingerprint
- [ ] Warn if inputs changed but outputs weren't regenerated

**CLI Flags:**
- `--fingerprint-dir .cache/fingerprints`
- `--skip-reproducibility-check` (for development)

**Failure Behavior:** Warning only (inputs may intentionally change)

---

## Integration with main.py

```python
# src/exps_logistic/main.py

def main():
    args = parse_args()

    # Load data
    df = load_data(args.results_dir, ...)

    # === NEW: Validation Phase ===
    if not args.skip_validation:
        from .validators import run_all_validators

        validation_result = run_all_validators(
            df=df,
            config=args,
            fail_fast=True,  # Stop on first failure
        )

        if not validation_result.all_passed:
            if args.validate_only:
                # Just print report and exit
                print(validation_result.format_report())
                sys.exit(1)
            else:
                raise ValidationError(validation_result.format_report())

    if args.validate_only:
        print("All validations passed!")
        sys.exit(0)

    # Continue with experiment...
```

---

## New CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--validate-only` | False | Run validation only, don't run experiment |
| `--skip-validation` | False | Skip all validation (for development) |
| `--max-staleness-hours` | 24.0 | Max age of source data before warning |
| `--min-coverage-pct` | 0.8 | Minimum data coverage required |
| `--max-parse-failure-rate` | 0.2 | Max gamma parsing failure rate |
| `--expected-models` | None | Comma-separated list of expected models |
| `--fingerprint-dir` | `.cache/fingerprints` | Where to store run fingerprints |

---

## File Structure

```
src/exps_logistic/
├── validators/
│   ├── __init__.py          # run_all_validators()
│   ├── freshness.py         # DataFreshnessValidator
│   ├── completeness.py      # DataCompletenessValidator
│   ├── parsing.py           # GammaParsingValidator
│   └── reproducibility.py   # ReproducibilityValidator
├── main.py                  # Updated with validation phase
└── config.py                # New CLI arguments
```

---

## Testing Plan

### Unit Tests

```python
# tests/logistic/test_validators.py

class TestFreshnessValidator:
    def test_fresh_data_passes(self): ...
    def test_stale_data_warns(self): ...
    def test_missing_files_fails(self): ...

class TestCompletenessValidator:
    def test_complete_data_passes(self): ...
    def test_missing_model_fails(self): ...
    def test_partial_coverage_threshold(self): ...

class TestParsingValidator:
    def test_all_parsed_passes(self): ...
    def test_high_bna_rate_fails(self): ...
    def test_captures_sample_failures(self): ...
    def test_per_kind_breakdown(self): ...

class TestReproducibilityValidator:
    def test_same_inputs_same_hash(self): ...
    def test_different_inputs_different_hash(self): ...
    def test_warns_on_stale_outputs(self): ...
```

### Integration Tests

```python
# tests/logistic/test_validation_integration.py

def test_validation_pipeline_passes_on_good_data(): ...
def test_validation_pipeline_fails_on_bad_data(): ...
def test_validate_only_mode(): ...
def test_skip_validation_mode(): ...
```

---

## Implementation Order

1. **GammaParsingValidator** (highest priority - user's main concern)
   - Implement parsing failure detection
   - Add detailed diagnostics
   - Add CLI flags

2. **DataCompletenessValidator**
   - Implement coverage matrix
   - Add threshold checking
   - Add CLI flags

3. **DataFreshnessValidator**
   - Implement mtime checking
   - Add git hash tracking
   - Add CLI flags

4. **ReproducibilityValidator**
   - Implement fingerprinting
   - Add storage/retrieval
   - Add CLI flags

5. **Integration**
   - Add validation phase to main.py
   - Add run_all_validators orchestrator
   - Update config.py with new args

6. **Testing**
   - Unit tests for each validator
   - Integration tests for pipeline

---

## Success Criteria

- [ ] Parsing failures are detected and reported with actionable diagnostics
- [ ] Missing model/seed/kind combinations are clearly identified
- [ ] Stale data triggers a warning before experiments run
- [ ] Input changes are tracked via fingerprinting
- [ ] All validation can be run independently via `--validate-only`
- [ ] Validation can be skipped for development via `--skip-validation`
- [ ] Clear error messages guide users to fix issues
