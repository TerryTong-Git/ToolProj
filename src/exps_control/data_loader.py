"""
Data loader for exps_control experiments.

Loads existing results from exps_performance to get:
- Generated code (sim_code)
- Original NL reasoning (nl_reasoning)
- Problem questions
- Gold answers
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Sample:
    """A single sample for the prefilled CoT experiment."""

    kind: str  # Problem type (e.g., "dijkstra", "bubble_sort")
    digit: int  # Problem size/complexity
    question: str  # Original problem question
    answer: str  # Gold answer

    # From existing results
    sim_code: str  # Generated code from Arm2
    nl_reasoning: str  # Original NL reasoning from Arm1

    # To be filled by translator
    translated_reasoning: str = ""

    # Results
    nl_prefill_answer: str = ""
    nl_prefill_correct: bool = False
    translated_prefill_answer: str = ""
    translated_prefill_correct: bool = False

    # Metadata
    model: str = ""
    seed: int = -1
    request_id: str = ""


def load_results_jsonl(path: Path) -> list[dict]:
    """Load results from a JSONL file."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_result_files(results_dir: Path) -> list[Path]:
    """Find all res.jsonl files in the results directory."""
    return list(results_dir.glob("**/res.jsonl"))


def load_samples(
    results_dir: Path,
    model_filter: Optional[str] = None,
    kind_filter: Optional[list[str]] = None,
    require_code: bool = True,
    require_nl: bool = True,
    max_samples: Optional[int] = None,
) -> list[Sample]:
    """Load samples from existing results.

    Args:
        results_dir: Path to results directory (e.g., src/exps_performance/results)
        model_filter: Only load results from this model (partial match)
        kind_filter: Only load these problem kinds
        require_code: Only include samples with non-empty sim_code
        require_nl: Only include samples with non-empty nl_reasoning
        max_samples: Maximum number of samples to load

    Returns:
        List of Sample objects
    """
    samples = []
    result_files = find_result_files(results_dir)

    for result_file in result_files:
        # Check model filter
        if model_filter and model_filter.lower() not in str(result_file).lower():
            continue

        rows = load_results_jsonl(result_file)

        for row in rows:
            # Check kind filter
            kind = row.get("kind", "")
            if kind_filter and kind not in kind_filter:
                continue

            # Check required fields
            sim_code = row.get("sim_code", "")
            nl_reasoning = row.get("nl_reasoning", "")

            if require_code and not sim_code:
                continue
            if require_nl and not nl_reasoning:
                continue

            # Skip if no question or answer
            question = row.get("question", "")
            answer = row.get("answer", "")
            if not question or not answer:
                continue

            sample = Sample(
                kind=kind,
                digit=row.get("digit", 0),
                question=question,
                answer=answer,
                sim_code=sim_code,
                nl_reasoning=nl_reasoning,
                model=row.get("model", ""),
                seed=row.get("seed", -1),
                request_id=row.get("request_id", ""),
            )
            samples.append(sample)

            if max_samples and len(samples) >= max_samples:
                return samples

    return samples


def get_unique_kinds(samples: list[Sample]) -> list[str]:
    """Get unique problem kinds from samples."""
    return sorted(set(s.kind for s in samples))


def filter_by_kind(samples: list[Sample], kinds: list[str]) -> list[Sample]:
    """Filter samples by problem kind."""
    return [s for s in samples if s.kind in kinds]


def sample_per_kind(samples: list[Sample], n_per_kind: int) -> list[Sample]:
    """Sample n samples per kind."""
    import random
    from collections import defaultdict

    by_kind = defaultdict(list)
    for s in samples:
        by_kind[s.kind].append(s)

    result = []
    for kind, kind_samples in by_kind.items():
        if len(kind_samples) <= n_per_kind:
            result.extend(kind_samples)
        else:
            result.extend(random.sample(kind_samples, n_per_kind))

    return result


if __name__ == "__main__":
    # Quick test
    results_dir = Path(__file__).parent.parent / "exps_performance" / "results"
    print(f"Looking for results in: {results_dir}")

    samples = load_samples(results_dir, max_samples=10)
    print(f"Loaded {len(samples)} samples")

    if samples:
        s = samples[0]
        print("\nFirst sample:")
        print(f"  Kind: {s.kind}")
        print(f"  Model: {s.model}")
        print(f"  Question: {s.question[:100]}...")
        print(f"  Code: {s.sim_code[:100]}..." if s.sim_code else "  Code: (empty)")
        print(f"  NL: {s.nl_reasoning[:100]}..." if s.nl_reasoning else "  NL: (empty)")

    kinds = get_unique_kinds(samples)
    print(f"\nUnique kinds: {kinds}")
