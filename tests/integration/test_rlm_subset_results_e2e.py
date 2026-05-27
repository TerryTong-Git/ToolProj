import json
import subprocess
import sys
from pathlib import Path

FINAL_OUTCOME_ROWS = 1372
FIVE_PERCENT_OUTCOME_ROWS = 69


def test_rlm_subset_results_rebuilds_report_from_raw_runs(tmp_path: Path) -> None:
    raw_path = tmp_path / "res.jsonl"
    raw_rows = [
        {
            "kind": "add",
            "digit": index + 1,
            "index_in_kind": index,
            "request_id": f"req-{index:03d}",
            "rlmcode_correct": index < 35,
            "rlmnl_correct": index < 34,
        }
        for index in range(FIVE_PERCENT_OUTCOME_ROWS)
    ]
    assert FIVE_PERCENT_OUTCOME_ROWS * 20 >= FINAL_OUTCOME_ROWS
    with raw_path.open("w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row))
            f.write("\n")

    outcomes_csv = tmp_path / "outcomes.csv"
    report_path = tmp_path / "rlm_results.md"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.reasoning_benchmark.scripts.analyze_rlm_subset_results",
            "--raw-run",
            f"Toy Model={raw_path}",
            "--outcomes-csv",
            str(outcomes_csv),
            "--report-path",
            str(report_path),
        ],
        check=True,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "| Toy Model | 35 / 69 | 50.7% | 34 / 69 | 49.3% | Code |" in report
    outcomes_text = outcomes_csv.read_text(encoding="utf-8")
    assert "req-000" in outcomes_text
    assert "req-068" in outcomes_text
