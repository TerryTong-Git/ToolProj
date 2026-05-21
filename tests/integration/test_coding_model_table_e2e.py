import csv
import subprocess
import sys
from pathlib import Path


def test_coding_model_table_rebuilds_from_compact_outcomes(tmp_path: Path) -> None:
    input_csv = tmp_path / "coding_model_outcomes.csv"
    output_md = tmp_path / "coding_model_table.md"
    rows = [
        {
            "model_label": "x-ai/grok-code-fast-1 (25% data)",
            "source_run": "grok",
            "source_file": "grok/res.jsonl",
            "row_index": "0",
            "kind": "add",
            "digit": "2",
            "index_in_kind": "1",
            "nl_correct": "true",
            "sim_correct": "true",
            "code_correct": "true",
            "nl_parse_err": "false",
            "sim_parse_err": "false",
            "code_err_msg": "ok,ok",
        },
        {
            "model_label": "qwen/qwen3-coder (25% data)",
            "source_run": "qwen",
            "source_file": "qwen/res.jsonl",
            "row_index": "0",
            "kind": "add",
            "digit": "2",
            "index_in_kind": "1",
            "nl_correct": "false",
            "sim_correct": "true",
            "code_correct": "true",
            "nl_parse_err": "false",
            "sim_parse_err": "false",
            "code_err_msg": "ok,ok",
        },
        {
            "model_label": "codestral-2508 (original)",
            "source_run": "codestral",
            "source_file": "codestral/res.jsonl",
            "row_index": "0",
            "kind": "add",
            "digit": "2",
            "index_in_kind": "1",
            "nl_correct": "false",
            "sim_correct": "false",
            "code_correct": "false",
            "nl_parse_err": "false",
            "sim_parse_err": "false",
            "code_err_msg": "ok,ok",
        },
    ]
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
    assert "| x-ai/grok-code-fast-1 (25% data) | 100.00% | 100.00% | 100.00% |" in report
    assert "| qwen/qwen3-coder (25% data) | 0.00% | 100.00% | 100.00% |" in report
    assert "| codestral-2508 (original) | 0.00% | 0.00% | 0.00% |" in report
