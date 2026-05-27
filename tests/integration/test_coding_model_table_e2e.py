import csv
import subprocess
import sys
from pathlib import Path

FINAL_COMPACT_OUTCOME_ROWS = 5440
FIVE_PERCENT_COMPACT_ROWS = 272


def _model_rows(model_label: str, source_run: str, total: int, nl_correct: int, sim_correct: int, code_correct: int) -> list[dict[str, str]]:
    return [
        {
            "model_label": model_label,
            "source_run": source_run,
            "source_file": f"{source_run}/res.jsonl",
            "row_index": str(index),
            "kind": "add",
            "digit": "2",
            "index_in_kind": str(index),
            "nl_correct": str(index < nl_correct).lower(),
            "sim_correct": str(index < sim_correct).lower(),
            "code_correct": str(index < code_correct).lower(),
            "nl_parse_err": "false",
            "sim_parse_err": "false",
            "code_err_msg": "ok,ok",
        }
        for index in range(total)
    ]


def test_coding_model_table_rebuilds_from_compact_outcomes(tmp_path: Path) -> None:
    input_csv = tmp_path / "coding_model_outcomes.csv"
    output_md = tmp_path / "coding_model_table.md"
    rows = [
        *_model_rows("x-ai/grok-code-fast-1 (25% data)", "grok", 91, 45, 60, 75),
        *_model_rows("qwen/qwen3-coder (25% data)", "qwen", 91, 30, 45, 60),
        *_model_rows("codestral-2508 (original)", "codestral", 90, 15, 30, 45),
    ]
    assert len(rows) == FIVE_PERCENT_COMPACT_ROWS
    assert FIVE_PERCENT_COMPACT_ROWS * 20 == FINAL_COMPACT_OUTCOME_ROWS
    with input_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    subprocess.run(
        [
            sys.executable,
            "src/exps_performance/scripts/analyze_coding_model_table.py",
            "--input-csv",
            str(input_csv),
            "--output-md",
            str(output_md),
        ],
        check=True,
    )

    report = output_md.read_text(encoding="utf-8")
    assert "| x-ai/grok-code-fast-1 (25% data) | 49.45% | 65.93% | 82.42% |" in report
    assert "| qwen/qwen3-coder (25% data) | 32.97% | 49.45% | 65.93% |" in report
    assert "| codestral-2508 (original) | 16.67% | 33.33% | 50.00% |" in report
    assert "| x-ai/grok-code-fast-1 (25% data) | 91 | 45/91 | 60/91 | 75/91 |" in report
