import csv
import subprocess
import sys
from pathlib import Path

FINAL_COMPACT_OUTCOME_ROWS = 1050
FIVE_PERCENT_COMPACT_ROWS = 53


def _row(
    model_label: str,
    source_run: str,
    row_index: int,
    nl_correct: str,
    sim_correct: str,
    code_correct: str,
) -> dict[str, str]:
    return {
        "model_label": model_label,
        "source_run": source_run,
        "source_file": f"{source_run}/res.jsonl",
        "row_index": str(row_index),
        "kind": "add",
        "digit": "2",
        "index_in_kind": str(row_index),
        "nl_correct": nl_correct,
        "sim_correct": sim_correct,
        "code_correct": code_correct,
        "nl_parse_err": "false",
        "sim_parse_err": "false",
        "code_err_msg": "ok,ok",
    }


def _rows(
    model_label: str,
    source_run: str,
    total: int,
    nl_correct: int,
    sim_correct: int,
    code_correct: int,
) -> list[dict[str, str]]:
    return [
        _row(
            model_label,
            source_run,
            index,
            str(index < nl_correct).lower(),
            str(index < sim_correct).lower(),
            str(index < code_correct).lower(),
        )
        for index in range(total)
    ]


def test_frontier_nopatch_table_rebuilds_from_compact_outcomes(tmp_path: Path) -> None:
    input_csv = tmp_path / "frontier_nopatch_outcomes.csv"
    output_md = tmp_path / "frontier_nopatch_table.md"
    rows = [
        *_rows("GPT-5.4", "run_20260406_nopatch_gpt54_seed1_subset350_py310", 18, 9, 12, 15),
        *_rows("Claude Opus 4.6", "run_20260406_nopatch_opus46_seed1_subset350_py310", 17, 10, 9, 14),
        *_rows("Claude Opus 4.6", "run_20260406_nopatch_opus46_seed1_subset350_rerun1_py310", 18, 6, 13, 7),
    ]
    assert len(rows) == FIVE_PERCENT_COMPACT_ROWS
    assert FIVE_PERCENT_COMPACT_ROWS * 20 >= FINAL_COMPACT_OUTCOME_ROWS
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
    assert "| GPT-5.4 | 50.00% | 66.67% | 83.33% |" in report
    assert "| Claude Opus 4.6 | 58.82% | 72.22% | 82.35% |" in report
    assert "| GPT-5.4 | Route 3 | 15/18 | `run_20260406_nopatch_gpt54_seed1_subset350_py310` |" in report
    assert "| Claude Opus 4.6 | Route 2 | 13/18 | `run_20260406_nopatch_opus46_seed1_subset350_rerun1_py310` |" in report
