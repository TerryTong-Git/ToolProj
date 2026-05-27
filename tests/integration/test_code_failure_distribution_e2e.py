import csv
import json
import subprocess
import sys
from pathlib import Path

from src.reasoning_benchmark.scripts.analyze_code_failure_distribution import DEFAULT_MODELS

FINAL_ROWS_PER_MODEL = 4740
FIVE_PERCENT_ROWS_PER_MODEL = 237


def _record(*, correct: bool, err_msg: str) -> dict[str, object]:
    return {"code_correct": correct, "code_err_msg": err_msg}


def _write_fixture_results(results_dir: Path) -> None:
    rows = [
        *[_record(correct=True, err_msg="ok,ok") for _ in range(117)],
        *[_record(correct=False, err_msg="type_check_failed,ok") for _ in range(20)],
        *[_record(correct=False, err_msg="ok,ok") for _ in range(40)],
        *[_record(correct=False, err_msg="type_check_failed,invalid syntax (<string>, line 1)") for _ in range(25)],
        *[_record(correct=False, err_msg="ok,division by zero") for _ in range(20)],
        *[_record(correct=False, err_msg="ok,timeout") for _ in range(15)],
    ]
    assert len(rows) == FIVE_PERCENT_ROWS_PER_MODEL
    assert FIVE_PERCENT_ROWS_PER_MODEL * 20 == FINAL_ROWS_PER_MODEL

    for model_dir in DEFAULT_MODELS.values():
        path = results_dir / f"{model_dir}_seed0" / "tb" / "run_fixture" / "res.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_code_failure_distribution_cli_regenerates_table(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    output_path = tmp_path / "analysis" / "code_failure_distribution.csv"
    _write_fixture_results(results_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.reasoning_benchmark.scripts.analyze_code_failure_distribution",
            "--results-dir",
            str(results_dir),
            "--exclude-parse-only",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))

    assert "Total: failures=600" in completed.stdout
    assert len(rows) == 7
    assert rows[0]["model"] == "Haiku"
    assert rows[0]["total"] == "237"
    assert rows[0]["success"] == "117"
    assert rows[0]["failures"] == "100"
    assert rows[0]["excluded_parse_only"] == "20"
    assert rows[0]["wrong_answer_pct"] == "40.0"
    assert rows[0]["syntax_error_pct"] == "25.0"
    assert rows[0]["runtime_error_pct"] == "20.0"
    assert rows[0]["time_limit_pct"] == "15.0"
    assert rows[-1]["model"] == "Total"
    assert rows[-1]["total"] == "1422"
    assert rows[-1]["failures"] == "600"
    assert rows[-1]["wrong_answer_pct"] == "40.0"
