#!/usr/bin/env python3
"""Backfill sim_reasoning from sim_answer JSON in existing result files.

The sim_answer field contains the full JSON response which includes the simulation
field from CodeReasoning. This script extracts it and populates sim_reasoning.
"""

import argparse
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_simulation_from_sim_answer(sim_answer: str) -> str | None:
    """Extract simulation field from raw sim_answer JSON string.

    The sim_answer contains escaped JSON like:
    {"Answer": "...", "code": "...", "simulation": "Step 1: ..."}
    """
    if not sim_answer:
        return None

    # Try to parse as JSON first (handles well-formed responses)
    try:
        data = json.loads(sim_answer)
        if isinstance(data, dict) and "simulation" in data:
            return data["simulation"]
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction for malformed JSON
    match = re.search(r'"simulation"\s*:\s*"((?:[^"\\]|\\.)*)"', sim_answer, re.DOTALL)
    if match:
        try:
            return match.group(1).encode().decode("unicode_escape")
        except (UnicodeDecodeError, ValueError):
            return match.group(1)

    return None


def backfill_jsonl_file(input_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Backfill sim_reasoning in a JSONL file.

    Returns: (records_updated, total_records)
    """
    records = []
    updated_count = 0
    total_count = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_count += 1
            record = json.loads(line)

            # Check if sim_reasoning is empty but sim_answer exists
            sim_reasoning = record.get("sim_reasoning", "")
            sim_answer = record.get("sim_answer", "")

            if not sim_reasoning and sim_answer:
                extracted = extract_simulation_from_sim_answer(sim_answer)
                if extracted:
                    record["sim_reasoning"] = extracted
                    updated_count += 1

            records.append(record)

    if not dry_run and updated_count > 0:
        with open(input_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return updated_count, total_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill sim_reasoning from sim_answer in result files"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Root directory containing result files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without modifying files",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    total_updated = 0
    total_records = 0
    files_modified = 0

    jsonl_files = list(results_dir.rglob("*.jsonl"))
    logger.info(f"Found {len(jsonl_files)} JSONL files to process")

    for jsonl_path in jsonl_files:
        updated, total = backfill_jsonl_file(jsonl_path, dry_run=args.dry_run)
        total_records += total
        if updated > 0:
            files_modified += 1
            total_updated += updated
            action = "Would update" if args.dry_run else "Updated"
            logger.info(f"{action} {updated}/{total} records in {jsonl_path.relative_to(results_dir)}")

    logger.info(f"Summary: {total_updated} records updated in {files_modified} files (total: {total_records} records)")
    if args.dry_run:
        logger.info("(Dry run - no files were modified)")


if __name__ == "__main__":
    main()
