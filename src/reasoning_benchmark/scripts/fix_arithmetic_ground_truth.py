#!/usr/bin/env python3
"""
Fix ground truth values for arithmetic tasks (add, sub, mul) in result files.

The original data generation used float arithmetic which loses precision for large integers.
This script recomputes the correct answers using Python's arbitrary-precision integers.
"""

import json
import re
from pathlib import Path

from src.reasoning_benchmark.artifact_paths import LEGACY_BENCHMARK_RESULTS_DIR


def extract_operands(question: str, kind: str) -> tuple[int, int] | None:
    """Extract operands from a question string."""
    if kind == "add":
        match = re.search(r"Compute:\s*(\d+)\s*\+\s*(\d+)", question)
    elif kind == "sub":
        match = re.search(r"Compute:\s*(\d+)\s*-\s*(\d+)", question)
    elif kind == "mul":
        match = re.search(r"Compute:\s*(\d+)\s*\*\s*(\d+)", question)
    else:
        return None

    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def compute_correct_answer(a: int, b: int, kind: str) -> int:
    """Compute the correct answer using integer arithmetic."""
    if kind == "add":
        return a + b
    elif kind == "sub":
        return a - b if a >= b else b - a
    elif kind == "mul":
        return a * b
    else:
        raise ValueError(f"Unknown kind: {kind}")


def extract_model_answer(raw_answer: str, arm: str) -> str | None:
    """Extract the actual answer value from model output.

    For code arm: answer is typically just the value
    For nl/sim/controlsim: answer is in JSON format with "Answer" field
    """
    if not raw_answer:
        return None

    raw_answer = raw_answer.strip()

    # For code arm, the answer is usually just the number
    if arm == "code":
        return raw_answer

    # For other arms, try to extract from JSON
    # First, try to find JSON in the response (may be wrapped in ```json ... ```)
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_answer, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    elif raw_answer.startswith("{"):
        # Find the end of the JSON object
        brace_count = 0
        end_idx = 0
        for i, c in enumerate(raw_answer):
            if c == "{":
                brace_count += 1
            elif c == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        json_str = raw_answer[:end_idx]
    else:
        return None

    try:
        parsed = json.loads(json_str)
        answer = parsed.get("Answer", "")
        return str(answer).strip()
    except (json.JSONDecodeError, TypeError):
        return None


def fix_jsonl_file(filepath: Path, dry_run: bool = False) -> dict:
    """Fix a single jsonl file and return stats."""
    stats = {"total": 0, "fixed": 0, "arithmetic": 0, "code_flipped": 0, "nl_flipped": 0, "sim_flipped": 0, "controlsim_flipped": 0}

    rows = []
    with open(filepath) as f:
        for line in f:
            rows.append(json.loads(line))

    for row in rows:
        stats["total"] += 1
        kind = row.get("kind", "")

        if kind not in ("add", "sub", "mul"):
            continue

        stats["arithmetic"] += 1
        question = row.get("question", "")
        operands = extract_operands(question, kind)

        if operands is None:
            continue

        a, b = operands
        correct_answer = str(compute_correct_answer(a, b, kind))
        old_answer = row.get("answer", "")

        if old_answer != correct_answer:
            stats["fixed"] += 1
            row["answer"] = correct_answer

        # Re-evaluate correctness for each arm (even if GT wasn't changed,
        # we should check if model got correct answer that was marked wrong)
        for arm in ["code", "nl", "sim", "controlsim"]:
            ans_field = f"{arm}_answer"
            correct_field = f"{arm}_correct"

            if ans_field in row and correct_field in row:
                raw_answer = row[ans_field]
                model_answer = extract_model_answer(str(raw_answer), arm)
                was_correct = row[correct_field]

                # Check if model answer matches correct answer
                is_correct = model_answer is not None and model_answer == correct_answer

                if was_correct != is_correct:
                    row[correct_field] = is_correct
                    stats[f"{arm}_flipped"] += 1

    # Write if any changes were made (ground truth or correctness)
    any_changes = stats["fixed"] > 0 or any(stats[f"{arm}_flipped"] > 0 for arm in ["code", "nl", "sim", "controlsim"])
    if not dry_run and any_changes:
        with open(filepath, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=str(LEGACY_BENCHMARK_RESULTS_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes, just report")
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    jsonl_files = sorted(results_root.rglob("*.jsonl"))

    print(f"Found {len(jsonl_files)} jsonl files")

    total_stats = {"total": 0, "fixed": 0, "arithmetic": 0, "code_flipped": 0, "nl_flipped": 0, "sim_flipped": 0, "controlsim_flipped": 0}

    for filepath in jsonl_files:
        stats = fix_jsonl_file(filepath, dry_run=args.dry_run)
        for k, v in stats.items():
            total_stats[k] += v

        any_flips = stats["code_flipped"] + stats["nl_flipped"] + stats["sim_flipped"] + stats["controlsim_flipped"]
        if stats["fixed"] > 0 or any_flips > 0:
            print(
                f"  {filepath.relative_to(results_root)}: fixed {stats['fixed']} answers, "
                f"code={stats['code_flipped']}, nl={stats['nl_flipped']}, "
                f"sim={stats['sim_flipped']}, controlsim={stats['controlsim_flipped']}"
            )

    print(f"\n{'DRY RUN - ' if args.dry_run else ''}Summary:")
    print(f"  Total records: {total_stats['total']}")
    print(f"  Arithmetic records: {total_stats['arithmetic']}")
    print(f"  Fixed GT answers: {total_stats['fixed']}")
    print(f"  Code correctness flipped: {total_stats['code_flipped']}")
    print(f"  NL correctness flipped: {total_stats['nl_flipped']}")
    print(f"  Sim correctness flipped: {total_stats['sim_flipped']}")
    print(f"  ControlSim correctness flipped: {total_stats['controlsim_flipped']}")


if __name__ == "__main__":
    main()
