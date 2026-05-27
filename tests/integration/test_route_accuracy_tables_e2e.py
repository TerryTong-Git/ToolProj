import csv
import json
import subprocess
import sys
from pathlib import Path

from src.reasoning_benchmark.scripts.analyze_route_accuracy_tables import COMPLEXITY_TASKS, MODEL_ORDER

FINAL_ROWS_PER_MODEL = 4020
FIVE_PERCENT_ROWS_PER_MODEL = 201


def _write_fixture_results(results_root: Path) -> None:
    tasks = [task for tasks_for_class in COMPLEXITY_TASKS.values() for task in tasks_for_class]
    assert FIVE_PERCENT_ROWS_PER_MODEL * 20 == FINAL_ROWS_PER_MODEL
    for model_index, model in enumerate(MODEL_ORDER):
        path = results_root / f"fixture_model_{model_index}" / "tb" / "run_fixture" / "res.jsonl"
        path.parent.mkdir(parents=True)
        rows = [
            {
                "kind": tasks[row_index % len(tasks)],
                "digit": row_index,
                "index_in_kind": row_index,
                "model": model,
                "nl_correct": row_index < 67,
                "sim_correct": row_index < 101,
                "code_correct": row_index < 151,
            }
            for row_index in range(FIVE_PERCENT_ROWS_PER_MODEL)
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_route_accuracy_tables_cli_regenerates_outputs(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    report_path = tmp_path / "route_accuracy_tables.md"
    complexity_csv = tmp_path / "accuracy_by_asymptotic_class.csv"
    model_csv = tmp_path / "accuracy_by_model.csv"
    _write_fixture_results(results_root)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.reasoning_benchmark.scripts.analyze_route_accuracy_tables",
            "--results-root",
            str(results_root),
            "--report-path",
            str(report_path),
            "--complexity-csv",
            str(complexity_csv),
            "--model-csv",
            str(model_csv),
        ],
        check=True,
    )

    complexity_rows = list(csv.DictReader(complexity_csv.open(encoding="utf-8")))
    model_rows = list(csv.DictReader(model_csv.open(encoding="utf-8")))
    report = report_path.read_text(encoding="utf-8")

    assert complexity_rows[0]["group"] == "O(1)"
    assert complexity_rows[0]["tasks"] == "4"
    assert complexity_rows[0]["instances"] == "20"
    assert complexity_rows[0]["rows"] == "120"
    assert complexity_rows[0]["route1"] == "0.400000"
    assert complexity_rows[0]["route2"] == "0.600000"
    assert complexity_rows[0]["route3"] == "0.800000"
    assert complexity_rows[0]["n01_route2_route1"] == "24"
    assert complexity_rows[0]["n01_route3_route2"] == "24"
    assert model_rows[0]["group"] == MODEL_ORDER[0]
    assert model_rows[0]["instances"] == "201"
    assert model_rows[0]["rows"] == "201"
    assert model_rows[0]["route1"] == "0.333333"
    assert model_rows[0]["route2"] == "0.502488"
    assert model_rows[0]["route3"] == "0.751244"
    assert "| O(1) | 4 | 20 | 40.0 | 60.0 | 80.0 | 20.0 | 1.2e-07 | 20.0 | 1.2e-07 |" in report
