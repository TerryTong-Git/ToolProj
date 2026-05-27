import csv
import json
import subprocess
import sys
from pathlib import Path

FRONTIER_STRUCTURED_FINAL_ROWS = 350
FRONTIER_STRUCTURED_FIVE_PERCENT_ROWS = 18


def test_analyze_sim_code_overlap_reproduces_five_percent_frontier_table_shard(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert FRONTIER_STRUCTURED_FIVE_PERCENT_ROWS / FRONTIER_STRUCTURED_FINAL_ROWS >= 0.05
    rows = [
        {
            "nl_correct": index < 9,
            "nl_parse_err": False,
            "sim_correct": index < 12,
            "sim_parse_err": False,
            "code_correct": index < 15,
            "code_err_msg": "ok,ok",
        }
        for index in range(FRONTIER_STRUCTURED_FIVE_PERCENT_ROWS)
    ]
    run_path = tmp_path / "res.jsonl"
    run_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report_path = tmp_path / "report.md"
    csv_path = tmp_path / "summary.csv"
    repo_root = Path(__file__).resolve().parents[2]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.reasoning_benchmark.scripts.analyze_sim_code_overlap",
            "--results-root",
            str(tmp_path),
            "--report-path",
            str(report_path),
            "--csv-path",
            str(csv_path),
            "--patching-run",
            f"Frontier 5% shard={run_path}",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    report = report_path.read_text(encoding="utf-8")
    assert ("| Frontier 5% shard | 18 | 50.00% | 66.67% | 83.33% | 50.00% | 66.67% | 83.33% | 18 |") in report

    with csv_path.open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[0]["label"] == "Frontier 5% shard"
    assert csv_rows[0]["rows"] == "18"
    assert csv_rows[0]["eligible_rows"] == "18"
