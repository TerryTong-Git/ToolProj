import csv
import json
import subprocess
import sys
from pathlib import Path

from src.exps_performance.scripts.analyze_route_accuracy_tables import COMPLEXITY_TASKS, MODEL_ORDER


def _write_fixture_results(results_root: Path) -> None:
    tasks = [task for tasks_for_class in COMPLEXITY_TASKS.values() for task in tasks_for_class]
    for model_index, model in enumerate(MODEL_ORDER):
        path = results_root / f"fixture_model_{model_index}" / "tb" / "run_fixture" / "res.jsonl"
        path.parent.mkdir(parents=True)
        rows = [
            {
                "kind": task,
                "digit": 0,
                "index_in_kind": task_index,
                "model": model,
                "nl_correct": False,
                "sim_correct": False,
                "code_correct": True,
            }
            for task_index, task in enumerate(tasks)
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
            "src/exps_performance/scripts/analyze_route_accuracy_tables.py",
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
    assert complexity_rows[0]["instances"] == "4"
    assert complexity_rows[0]["route1"] == "0.000000"
    assert complexity_rows[0]["route2"] == "0.000000"
    assert complexity_rows[0]["route3"] == "1.000000"
    assert model_rows[0]["group"] == MODEL_ORDER[0]
    assert model_rows[0]["instances"] == "44"
    assert model_rows[0]["route3"] == "1.000000"
    assert "| O(1) | 4 | 4 | 0.0 | 0.0 | 100.0 | 0.0 | 1 | 100.0 |" in report
