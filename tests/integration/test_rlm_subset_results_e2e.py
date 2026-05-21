import json
import subprocess
import sys
from pathlib import Path


def test_rlm_subset_results_rebuilds_report_from_raw_runs(tmp_path: Path) -> None:
    raw_path = tmp_path / "res.jsonl"
    raw_rows = [
        {
            "kind": "add",
            "digit": 2,
            "index_in_kind": 1,
            "request_id": "req-1",
            "rlmcode_correct": True,
            "rlmnl_correct": False,
        },
        {
            "kind": "sub",
            "digit": 3,
            "index_in_kind": 1,
            "request_id": "req-2",
            "rlmcode_correct": False,
            "rlmnl_correct": True,
        },
        {
            "kind": "mul",
            "digit": 4,
            "index_in_kind": 1,
            "request_id": "req-3",
            "rlmcode_correct": True,
            "rlmnl_correct": True,
        },
    ]
    with raw_path.open("w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row))
            f.write("\n")

    outcomes_csv = tmp_path / "outcomes.csv"
    report_path = tmp_path / "rlm_results.md"

    subprocess.run(
        [
            sys.executable,
            "src/exps_performance/scripts/analyze_rlm_subset_results.py",
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
    assert "| Toy Model | 2 / 3 | 66.7% | 2 / 3 | 66.7% | Tie |" in report
    assert "req-1" in outcomes_csv.read_text(encoding="utf-8")
