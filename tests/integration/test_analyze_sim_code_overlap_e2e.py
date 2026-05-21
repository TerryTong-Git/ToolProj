import csv
import json
import subprocess
import sys
from pathlib import Path


def test_analyze_sim_code_overlap_cli(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rows = [
        {
            "nl_correct": True,
            "nl_parse_err": False,
            "sim_correct": True,
            "sim_parse_err": False,
            "code_correct": True,
            "code_err_msg": "ok,ok",
        },
        {
            "nl_correct": False,
            "nl_parse_err": True,
            "sim_correct": False,
            "sim_parse_err": False,
            "code_correct": False,
            "code_err_msg": "runtime,error",
        },
    ]
    run_path = tmp_path / "res.jsonl"
    run_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report_path = tmp_path / "report.md"
    csv_path = tmp_path / "summary.csv"
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "src" / "exps_performance" / "scripts" / "analyze_sim_code_overlap.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--results-root",
            str(tmp_path),
            "--report-path",
            str(report_path),
            "--csv-path",
            str(csv_path),
            "--patching-run",
            f"Tiny GPT={run_path}",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "| Tiny GPT | 2 | 50.00% | 50.00% | 50.00% | 100.00% | 50.00% | 100.00% | 1 |" in report

    with csv_path.open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert csv_rows[0]["label"] == "Tiny GPT"
    assert csv_rows[0]["rows"] == "2"
    assert csv_rows[0]["eligible_rows"] == "1"
