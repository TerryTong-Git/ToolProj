"""Post-hoc validation tests for src/reasoning_benchmark/results data.

These tests validate the integrity of experiment results data:
- All result files have consistent sample counts
- No blank/empty results where data is expected
- All required fields are present
- Data types are correct
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.reasoning_benchmark.artifact_paths import LEGACY_BENCHMARK_RESULTS_DIR

# Path to recorded benchmark results directory.
RESULTS_DIR = LEGACY_BENCHMARK_RESULTS_DIR

# Expected fields in each result record
REQUIRED_FIELDS = {
    "request_id",
    "unique_tag",
    "index_in_kind",
    "model",
    "seed",
    "exp_id",
    "digit",
    "kind",
    "question",
    "answer",
}

# Arm-specific fields (each arm has these suffixed fields)
ARM_NAMES = ["nl", "code", "sim", "controlsim"]
ARM_FIELDS = ["question", "answer", "correct", "parse_err"]

# Exact expected sample count (all files must have this exact count)
EXPECTED_SAMPLE_COUNT = 1580


def get_all_result_files() -> list[Path]:
    """Get active res.jsonl files from the results directory."""
    return sorted(path for path in RESULTS_DIR.rglob("res.jsonl") if "unused" not in path.relative_to(RESULTS_DIR).parts)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file and return list of records."""
    records = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON at {path}:{line_num}: {e}")
    return records


@pytest.fixture(scope="module")
def all_result_files() -> list[Path]:
    """Fixture providing all result file paths."""
    files = get_all_result_files()
    if not files:
        pytest.skip("No result files found in results directory")
    return files


@pytest.fixture(scope="module")
def result_file_data(all_result_files: list[Path]) -> dict[Path, list[dict]]:
    """Fixture providing loaded data from all result files."""
    return {path: load_jsonl(path) for path in all_result_files}


class TestResultFilesExist:
    """Tests for verifying result files exist and are accessible."""

    def test_results_directory_exists(self) -> None:
        """Verify the results directory exists."""
        assert RESULTS_DIR.exists(), f"Results directory not found: {RESULTS_DIR}"
        assert RESULTS_DIR.is_dir(), f"Results path is not a directory: {RESULTS_DIR}"

    def test_result_files_found(self, all_result_files: list[Path]) -> None:
        """Verify at least one result file exists."""
        assert len(all_result_files) > 0, "No result files found"

    def test_minimum_result_files(self, all_result_files: list[Path]) -> None:
        """Verify we have a reasonable number of result files (at least 10 model-seed combos)."""
        assert len(all_result_files) >= 10, f"Expected at least 10 result files, found {len(all_result_files)}"


class TestSampleCounts:
    """Tests for verifying consistent sample counts across result files."""

    def test_all_files_have_data(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify no result files are empty."""
        for path, records in result_file_data.items():
            assert len(records) > 0, f"Empty result file: {path}"

    def test_exact_sample_count(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify all result files have exactly the expected sample count."""
        for path, records in result_file_data.items():
            count = len(records)
            assert (
                count == EXPECTED_SAMPLE_COUNT
            ), f"File {path.relative_to(RESULTS_DIR)} has {count} samples, expected exactly {EXPECTED_SAMPLE_COUNT}"

    def test_consistent_sample_counts(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify all result files have identical sample counts."""
        counts = {path: len(records) for path, records in result_file_data.items()}
        unique_counts = set(counts.values())
        assert len(unique_counts) == 1, f"Inconsistent sample counts: {unique_counts}. All files must have the same count."


class TestRequiredFields:
    """Tests for verifying all required fields are present."""

    def test_base_fields_present(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify all base required fields are present in every record."""
        for path, records in result_file_data.items():
            for i, record in enumerate(records):
                missing = REQUIRED_FIELDS - set(record.keys())
                assert not missing, f"Missing required fields in {path.relative_to(RESULTS_DIR)} record {i}: {missing}"

    def test_arm_fields_present(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify arm-specific fields are present for all arms."""
        for path, records in result_file_data.items():
            for i, record in enumerate(records):
                for arm in ARM_NAMES:
                    for field in ARM_FIELDS:
                        full_field = f"{arm}_{field}"
                        assert full_field in record, f"Missing field '{full_field}' in {path.relative_to(RESULTS_DIR)} record {i}"


class TestNoBlankResults:
    """Tests for verifying no unexpected blank/empty results."""

    def test_no_blank_model_field(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify model field is never blank."""
        for path, records in result_file_data.items():
            for i, record in enumerate(records):
                model = record.get("model", "")
                assert model and str(model).strip(), f"Blank model field in {path.relative_to(RESULTS_DIR)} record {i}"

    def test_no_blank_kind_field(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify kind field is never blank."""
        for path, records in result_file_data.items():
            for i, record in enumerate(records):
                kind = record.get("kind", "")
                assert kind and str(kind).strip(), f"Blank kind field in {path.relative_to(RESULTS_DIR)} record {i}"

    # NP-hard kinds that have blank answers at digit=0 by design
    # (graph/combinatorial problems with 0 nodes have no meaningful answers)
    NP_HARD_KINDS = {"spp", "tsp", "tsp_d", "msp", "ksp", "gcp", "gcp_d", "bsp", "edp"}

    def test_no_blank_ground_truth_answers(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify no blank ground truth answers - all must be populated.

        Note: NP-hard kinds at digit=0 are intentionally blank by design
        (graph/combinatorial problems with 0 nodes have no meaningful answers).
        These are excluded from the check.
        """
        for path, records in result_file_data.items():
            for i, record in enumerate(records):
                # Skip NP-hard kinds at digit=0 (intentionally blank by design)
                if record.get("kind") in self.NP_HARD_KINDS and record.get("digit") == 0:
                    continue
                answer = record.get("answer")
                assert answer is not None and str(answer).strip() != "", (
                    f"Blank ground truth answer in {path.relative_to(RESULTS_DIR)} "
                    f"record {i} (kind={record.get('kind')}, digit={record.get('digit')})"
                )

    def test_all_questions_populated(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify arm questions have acceptable population rates.

        - nl_question and sim_question: 100% population required
        - code_question and controlsim_question: Max 15% blank allowed
          (blanks occur due to model generation/parsing failures)
        """
        # Arms with 100% required vs arms with tolerance for generation failures
        strict_arms = {"nl", "sim"}
        tolerant_arms = {"code", "controlsim"}
        max_blank_rate = 0.15  # 15% max blank rate for tolerant arms

        for path, records in result_file_data.items():
            # Check strict arms (100% required)
            for arm in strict_arms:
                question_field = f"{arm}_question"
                for i, record in enumerate(records):
                    question = record.get(question_field)
                    assert question and str(question).strip(), (
                        f"Blank {question_field} in {path.relative_to(RESULTS_DIR)} "
                        f"record {i} (kind={record.get('kind')}, digit={record.get('digit')})"
                    )

            # Check tolerant arms (max blank rate)
            for arm in tolerant_arms:
                question_field = f"{arm}_question"
                blank_count = sum(1 for r in records if not r.get(question_field) or str(r.get(question_field)).strip() == "")
                blank_rate = blank_count / len(records)
                assert (
                    blank_rate <= max_blank_rate
                ), f"{question_field} in {path.relative_to(RESULTS_DIR)} has {blank_rate:.1%} blank rate (max allowed: {max_blank_rate:.0%})"

    def test_all_answers_populated(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify all arm answers are populated (may be empty string for parse errors, but field must exist)."""
        for path, records in result_file_data.items():
            for i, record in enumerate(records):
                for arm in ARM_NAMES:
                    answer_field = f"{arm}_answer"
                    assert answer_field in record, (
                        f"Missing {answer_field} in {path.relative_to(RESULTS_DIR)} "
                        f"record {i} (kind={record.get('kind')}, digit={record.get('digit')})"
                    )


class TestDataTypes:
    """Tests for verifying correct data types."""

    def test_digit_is_integer(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify digit field is an integer."""
        for path, records in result_file_data.items():
            for i, record in enumerate(records):
                digit = record.get("digit")
                assert isinstance(digit, int), f"digit field is not int in {path.relative_to(RESULTS_DIR)} record {i}: {type(digit)}"

    def test_seed_is_integer(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify seed field is an integer."""
        for path, records in result_file_data.items():
            for i, record in enumerate(records):
                seed = record.get("seed")
                assert isinstance(seed, int), f"seed field is not int in {path.relative_to(RESULTS_DIR)} record {i}: {type(seed)}"

    def test_correct_fields_are_boolean(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify *_correct fields are boolean."""
        for path, records in result_file_data.items():
            for i, record in enumerate(records):
                for arm in ARM_NAMES:
                    correct_field = f"{arm}_correct"
                    value = record.get(correct_field)
                    assert isinstance(
                        value, bool
                    ), f"{correct_field} is not bool in {path.relative_to(RESULTS_DIR)} record {i}: {type(value)} = {value}"

    def test_parse_err_fields_are_boolean(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify *_parse_err fields are boolean."""
        for path, records in result_file_data.items():
            for i, record in enumerate(records):
                for arm in ARM_NAMES:
                    parse_err_field = f"{arm}_parse_err"
                    value = record.get(parse_err_field)
                    assert isinstance(
                        value, bool
                    ), f"{parse_err_field} is not bool in {path.relative_to(RESULTS_DIR)} record {i}: {type(value)} = {value}"


class TestDataConsistency:
    """Tests for data consistency within and across files."""

    def test_model_matches_directory(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify model field matches the directory name."""
        for path, records in result_file_data.items():
            # Extract model from path: results/{model}_seed{N}/tb/run_*/res.jsonl
            dir_name = path.parent.parent.parent.name  # e.g., "gemini-2.0-flash-001_seed0"
            expected_model_prefix = dir_name.rsplit("_seed", 1)[0]

            for i, record in enumerate(records):
                model = record.get("model", "")
                # Model in record might have provider prefix like "google/gemini-2.0-flash-001"
                model_suffix = model.split("/")[-1] if "/" in model else model
                assert (
                    model_suffix == expected_model_prefix or expected_model_prefix in model
                ), f"Model mismatch in {path.relative_to(RESULTS_DIR)} record {i}: directory says '{expected_model_prefix}', record says '{model}'"

    def test_seed_matches_directory(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify seed field matches the directory name."""
        for path, records in result_file_data.items():
            # Extract seed from path
            dir_name = path.parent.parent.parent.name
            expected_seed = int(dir_name.rsplit("_seed", 1)[1])

            for i, record in enumerate(records):
                seed = record.get("seed")
                assert (
                    seed == expected_seed
                ), f"Seed mismatch in {path.relative_to(RESULTS_DIR)} record {i}: directory says {expected_seed}, record says {seed}"

    def test_unique_tags_are_unique_within_file(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify unique_tag field is actually unique within each file."""
        for path, records in result_file_data.items():
            tags = [r.get("unique_tag") for r in records]
            unique_tags = set(tags)
            assert len(tags) == len(
                unique_tags
            ), f"Duplicate unique_tags in {path.relative_to(RESULTS_DIR)}: {len(tags)} records but only {len(unique_tags)} unique tags"


class TestParseErrorRates:
    """Tests for parse error rates being within acceptable bounds."""

    def test_aggregate_parse_error_rate(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify aggregate parse error rate across all files is reasonable."""
        total_records = 0
        total_errors_by_arm = {arm: 0 for arm in ARM_NAMES}

        for path, records in result_file_data.items():
            total_records += len(records)
            for arm in ARM_NAMES:
                parse_err_field = f"{arm}_parse_err"
                total_errors_by_arm[arm] += sum(1 for r in records if r.get(parse_err_field, False))

        # Report error rates (informational)
        print("\nAggregate parse error rates:")
        for arm, errors in total_errors_by_arm.items():
            rate = errors / total_records if total_records > 0 else 0
            print(f"  {arm}: {rate:.1%} ({errors}/{total_records})")

        # At least one arm should have reasonable parse success
        min_success_rate = 0.3  # At least 30% success for best arm
        best_success = max(1 - (errors / total_records) if total_records > 0 else 0 for errors in total_errors_by_arm.values())
        assert best_success >= min_success_rate, f"All arms have very high parse error rates. Best success rate: {best_success:.1%}"


class TestKindCoverage:
    """Tests for verifying problem kinds are valid and covered."""

    # All valid kinds that can appear in results (current experiment config only)
    VALID_KINDS = {
        # Fine-grained arithmetic
        "add",
        "sub",
        "mul",
        # Fine-grained DP
        "lcs",
        "knap",
        "rod",
        # ILP
        "ilp_assign",
        "ilp_prod",
        "ilp_partition",
        # CLRS algorithms
        "activity_selector",
        "articulation_points",
        "bellman_ford",
        "bfs",
        "binary_search",
        "bridges",
        "bubble_sort",
        "dag_shortest_paths",
        "dfs",
        "dijkstra",
        "find_maximum_subarray_kadane",
        "floyd_warshall",
        "graham_scan",
        "heapsort",
        "insertion_sort",
        "jarvis_march",
        "kmp_matcher",
        "lcs_length",
        "matrix_chain_order",
        "minimum",
        "mst_kruskal",
        "mst_prim",
        "naive_string_matcher",
        "optimal_bst",
        "quickselect",
        "quicksort",
        "segments_intersect",
        "strongly_connected_components",
        "task_scheduling",
        "topological_sort",
        # NP-hard (base and decision variants)
        "edp",
        "gcp",
        "gcp_d",
        "ksp",
        "spp",
        "tsp",
        "tsp_d",
        # NP-hard additional variants
        "msp",
        "bsp",
    }

    def test_all_kinds_are_valid(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify all kinds in results are from the valid set."""
        for path, records in result_file_data.items():
            kinds_in_file = {r.get("kind") for r in records}
            invalid_kinds = kinds_in_file - self.VALID_KINDS
            assert not invalid_kinds, f"Invalid kinds in {path.relative_to(RESULTS_DIR)}: {invalid_kinds}"

    def test_minimum_kinds_per_file(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify each file has at least some problem variety."""
        min_kinds = 3  # Each file should have at least 3 different problem kinds
        for path, records in result_file_data.items():
            kinds_in_file = {r.get("kind") for r in records}
            assert (
                len(kinds_in_file) >= min_kinds
            ), f"Too few kinds in {path.relative_to(RESULTS_DIR)}: found {len(kinds_in_file)}, expected at least {min_kinds}"

    def test_digit_range_coverage(self, result_file_data: dict[Path, list[dict]]) -> None:
        """Verify digit range is reasonable (0-100 to accommodate various experiments)."""
        for path, records in result_file_data.items():
            digits = {r.get("digit") for r in records if r.get("digit") is not None}
            if not digits:
                continue
            min_digit, max_digit = min(digits), max(digits)
            # CLRS can have digit=0, some experiments go up to 64+
            assert min_digit >= 0, f"Unexpected negative digit {min_digit} in {path.relative_to(RESULTS_DIR)}"
            assert max_digit <= 100, f"Unexpected max digit {max_digit} in {path.relative_to(RESULTS_DIR)}"
