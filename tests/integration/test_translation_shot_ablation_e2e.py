import csv
import json
import random
import subprocess
import sys
from pathlib import Path

from src.exps_functional.run_translation_additivity import (
    build_arg_parser,
    build_translate_prompt,
    load_samples,
    output_stem,
)


def _trial(sample_id: str, condition: str, correct: bool) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "kind": "minimum",
        "condition": condition,
        "gold_answer": "1",
        "predicted_answer": "1" if correct else "0",
        "correct": correct,
        "raw_response": "",
    }


FINAL_LEGACY_TRIAL_ROWS = 600
FIVE_PERCENT_TRIAL_ROWS = 30
TRIALS_PER_CONDITION = FIVE_PERCENT_TRIAL_ROWS // 3


def _condition_trials(condition: str, correct_count: int) -> list[dict[str, object]]:
    return [_trial(f"{condition}-{index}", condition, index < correct_count) for index in range(TRIALS_PER_CONDITION)]


def _write_trials(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_translation_shot_ablation_report_cli(tmp_path: Path) -> None:
    results_dir = tmp_path / "functional_results"
    shot_rows = [
        *_condition_trials("x", 5),
        *_condition_trials("x_nl_native", 7),
        *_condition_trials("x_nl_translated", 4),
    ]
    legacy_rows = [
        *_condition_trials("x", 6),
        *_condition_trials("x_nl_native", 8),
        *_condition_trials("x_nl_translated", 7),
    ]
    assert len(shot_rows) == FIVE_PERCENT_TRIAL_ROWS
    assert len(legacy_rows) == FIVE_PERCENT_TRIAL_ROWS
    assert FIVE_PERCENT_TRIAL_ROWS * 20 == FINAL_LEGACY_TRIAL_ROWS
    _write_trials(
        results_dir / "translation_anthropic_claude-haiku-4.5_source_claude-haiku-4.5_shots0_subset25_20260101_trials.jsonl",
        shot_rows,
    )
    legacy_path = tmp_path / "legacy_trials.jsonl"
    _write_trials(legacy_path, legacy_rows)

    report_path = tmp_path / "report.md"
    csv_path = tmp_path / "summary.csv"
    subprocess.run(
        [
            sys.executable,
            "src/exps_functional/scripts/analyze_translation_shot_ablation.py",
            "--results-dir",
            str(results_dir),
            "--legacy-trials",
            str(legacy_path),
            "--report-path",
            str(report_path),
            "--csv-path",
            str(csv_path),
        ],
        check=True,
    )

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    report = report_path.read_text(encoding="utf-8")

    assert rows[0]["shots"] == "0"
    assert rows[0]["x"] == "50.00%"
    assert rows[0]["x_native_nl"] == "70.00%"
    assert rows[0]["x_translated_nl"] == "40.00%"
    assert rows[0]["delta_native"] == "+20.00pp"
    assert rows[0]["delta_translated"] == "-10.00pp"
    assert rows[0]["gap"] == "-30.00pp"
    assert rows[1]["shots"] == "10 (paper legacy)"
    assert "| 0 | 50.00% | 70.00% | 40.00% | +20.00pp | -10.00pp | -30.00pp |" in report
    assert "| 10 (paper legacy) | 60.00% | 80.00% | 70.00% | +20.00pp | +10.00pp | -10.00pp |" in report


def test_translation_runner_accepts_shot_ablation_flags(tmp_path: Path) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model",
            "anthropic/claude-haiku-4.5",
            "--source_model",
            "claude-haiku-4.5",
            "--subset_fraction",
            "0.25",
            "--n_shots",
            "2",
        ]
    )

    prompt = build_translate_prompt("def solution():\n    return 1", args.n_shots)
    stem = output_stem(args, "20260101_000000")

    assert "### Example 1" in prompt
    assert "### Example 2" in prompt
    assert "### Example 3" not in prompt
    assert "def solution()" in prompt
    assert stem == "translation_anthropic_claude-haiku-4.5_source_claude-haiku-4.5_shots2_subset25_20260101_000000"


def test_load_samples_applies_subset_fraction(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    path = results_dir / "claude-haiku-4.5_seed0" / "tb" / "run_fixture" / "res.jsonl"
    path.parent.mkdir(parents=True)
    rows = []
    for idx in range(4):
        rows.append(
            {
                "sim_code": "def solution():\n    return 1\n" + ("#" * 60),
                "nl_reasoning": "This is a long enough native reasoning trace. " * 2,
                "question": "What is one?",
                "answer": "1",
                "kind": "minimum",
                "index_in_kind": idx,
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    random.seed(0)
    samples = load_samples(
        results_dir,
        source_model_filter="claude-haiku-4.5",
        max_samples=10,
        max_per_kind=10,
        subset_fraction=0.5,
    )

    assert len(samples) == 2
    assert {sample.source_model for sample in samples} == {"claude-haiku-4.5"}
