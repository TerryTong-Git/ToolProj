import csv
import subprocess
import sys
from pathlib import Path


def _row(
    model_label: str,
    source_run: str,
    nl_correct: str,
    sim_correct: str,
    code_correct: str,
) -> dict[str, str]:
    return {
        "model_label": model_label,
        "source_run": source_run,
        "source_file": f"{source_run}/res.jsonl",
        "row_index": "0",
        "kind": "add",
        "digit": "2",
        "index_in_kind": "1",
        "nl_correct": nl_correct,
        "sim_correct": sim_correct,
        "code_correct": code_correct,
        "nl_parse_err": "false",
        "sim_parse_err": "false",
        "code_err_msg": "ok,ok",
    }


def test_frontier_nopatch_table_rebuilds_from_compact_outcomes(tmp_path: Path) -> None:
    input_csv = tmp_path / "frontier_nopatch_outcomes.csv"
    output_md = tmp_path / "frontier_nopatch_table.md"
    rows = [
        _row("GPT-5.4", "run_20260406_nopatch_gpt54_seed1_subset350_py310", "true", "false", "true"),
        _row("Claude Opus 4.6", "run_20260406_nopatch_opus46_seed1_subset350_py310", "true", "false", "true"),
        _row("Claude Opus 4.6", "run_20260406_nopatch_opus46_seed1_subset350_rerun1_py310", "false", "true", "false"),
    ]
    with input_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    subprocess.run(
        [
            sys.executable,
            "src/exps_performance/scripts/analyze_frontier_nopatch_table.py",
            "--input-csv",
            str(input_csv),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )

    report = output_md.read_text(encoding="utf-8")
    assert "| GPT-5.4 | 100.00% | 0.00% | 100.00% |" in report
    assert "| Claude Opus 4.6 | 100.00% | 100.00% | 100.00% |" in report
    assert "| Claude Opus 4.6 | Route 2 | 1/1 | `run_20260406_nopatch_opus46_seed1_subset350_rerun1_py310` |" in report
