#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stage_counts(rows: list[dict[str, Any]], stage: str) -> dict[str, int]:
    err_field = f"{stage}_err_msg"
    parse_field = f"{stage}_parse_err"
    correct_field = f"{stage}_correct"
    question_field = f"{stage}_question"
    answer_field = f"{stage}_answer"
    counts = Counter(str(row.get(err_field, "")) for row in rows)
    return {
        "rows": len(rows),
        "ok": sum(1 for row in rows if row.get(err_field) == "ok"),
        "parse_err_true": sum(1 for row in rows if bool(row.get(parse_field, False))),
        "correct": sum(1 for row in rows if bool(row.get(correct_field, False))),
        "question_populated": sum(1 for row in rows if bool(str(row.get(question_field, "") or "").strip())),
        "answer_populated": sum(1 for row in rows if bool(str(row.get(answer_field, "") or "").strip())),
        "err_msg_populated": sum(1 for row in rows if row.get(err_field) not in (None, "")),
        "distinct_errs": len(counts),
    }


def aggregate_llm_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    per_stage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        stage = str(row.get("stage_set_name", row.get("stage", "")))
        for key in (
            "llm_requests",
            "llm_total_attempts",
            "llm_errors",
            "structured_requested",
            "reasoning_requested",
            "reasoning_visible",
            "reasoning_details_visible",
            "reasoning_tokens_visible",
            "empty_text_responses",
            "content_parse_failures_after_retry",
        ):
            per_stage[stage][key] += int(row.get(key, 0) or 0)
    return {stage: dict(values) for stage, values in per_stage.items()}


def validate(summary: dict[str, Any]) -> None:
    for stage in ("nl", "sim"):
        counts = summary["stages"][stage]
        if counts["question_populated"] != counts["rows"]:
            raise RuntimeError(f"{stage} stage has missing prompts in res.jsonl: {counts}")
        if counts["ok"] != counts["rows"]:
            raise RuntimeError(f"{stage} stage has schema parse failures: {counts}")
        if counts["parse_err_true"] != 0:
            raise RuntimeError(f"{stage} stage still records parse_err rows: {counts}")

        llm = summary["llm"].get(stage, {})
        if llm.get("structured_requested", 0) != llm.get("llm_requests", 0):
            raise RuntimeError(f"{stage} stage did not request structured outputs")
        if llm.get("reasoning_requested", 0) != llm.get("llm_requests", 0):
            raise RuntimeError(f"{stage} stage did not request reasoning")
        if llm.get("reasoning_visible", 0) == 0 and llm.get("reasoning_tokens_visible", 0) == 0 and llm.get("reasoning_details_visible", 0) == 0:
            raise RuntimeError(f"{stage} stage did not surface any reasoning signal")

    code_counts = summary["stages"]["code"]
    if code_counts["question_populated"] != code_counts["rows"]:
        raise RuntimeError(f"code stage did not run for every row: {code_counts}")
    if code_counts["err_msg_populated"] != code_counts["rows"]:
        raise RuntimeError(f"code stage is missing execution outcomes for some rows: {code_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a structured-output run directory.")
    parser.add_argument("run_dir", help="Run directory containing res.jsonl and llm_stage_stats.jsonl")
    parser.add_argument("--write-summary", action="store_true", help="Write validation_summary.json next to the run artifacts.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    res_path = run_dir / "res.jsonl"
    llm_path = run_dir / "llm_stage_stats.jsonl"
    rows = load_jsonl(res_path)
    llm_rows = load_jsonl(llm_path) if llm_path.exists() else []

    summary = {
        "run_dir": str(run_dir),
        "rows": len(rows),
        "stages": {
            "nl": stage_counts(rows, "nl"),
            "sim": stage_counts(rows, "sim"),
            "code": stage_counts(rows, "code"),
        },
        "llm": aggregate_llm_stats(llm_rows),
    }

    validate(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.write_summary:
        out_path = run_dir / "validation_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
