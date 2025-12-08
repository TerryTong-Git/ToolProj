from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import pandas as pd
from simple_parsing import parse

from src.exps_performance.arms import Arm1, Arm2, Arm3, Arm4
from src.exps_performance.dataset import make_dataset
from src.exps_performance.llm import llm
from src.exps_performance.metrics import accuracy_by_noise
from src.exps_performance.noise import NOISE_FUNCS, clamp_sigma, perturb
from src.exps_performance.utils import seed_all_and_setup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class NoiseArgs:
    model: str = "dummy"
    backend: str = "dummy"
    batch_size: int = 4
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    checkpoint_every: int = 8
    exec_workers: int = 4
    seed: int = 1
    kinds: List[str] = field(
        default_factory=lambda: [
            "add",
            "sub",
            "mul",
            "lcs",
            "knap",
            "rod",
            "ilp_assign",
            "ilp_prod",
            "ilp_partition",
        ]
    )
    digits_list: List[int] = field(default_factory=lambda: [2, 4])
    n: int = 4
    sigma: List[float] = field(default_factory=lambda: [round(x * 0.1, 2) for x in range(0, 11)])
    noise_types: List[str] = field(default_factory=lambda: list(NOISE_FUNCS.keys()))
    root: str = "results_noise"


def _run_all_arms(data: List[Any], args: NoiseArgs, client: Any) -> List[Any]:
    """Run the four arms sequentially on the given dataset copy."""
    pipeline = (Arm2, Arm3, Arm4, Arm1)
    updated = data
    for arm_cls in pipeline:
        arm = arm_cls(updated, args, client)
        _, updated = arm.run()
    return updated


def _ensure_outdir(root: str) -> Path:
    outdir = Path(root)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _evaluate_noise(base_data: List[Any], args: NoiseArgs, client: Any, noise_type: str, sigma: float) -> pd.DataFrame:
    data_copy = copy.deepcopy(base_data)
    sigma = clamp_sigma(sigma)
    for q in data_copy:
        q.question = perturb(q.question, noise_type, sigma, args.seed)
    completed = _run_all_arms(data_copy, args, client)
    records = [q.record for q in completed]
    acc_df = accuracy_by_noise(records, noise_type, sigma)
    return acc_df


def run(args: NoiseArgs) -> None:
    seed_all_and_setup(args)
    client = llm(args)
    logger.info(f"Using backend={args.backend} model={args.model}")

    base_data = list(make_dataset(args.kinds, args.n, args.digits_list))
    outdir = _ensure_outdir(args.root)
    outfile = outdir / "noise_results.jsonl"

    results: List[pd.DataFrame] = []
    for noise_type in args.noise_types:
        for sigma in args.sigma:
            logger.info(f"Running noise_type={noise_type} sigma={sigma}")
            acc_df = _evaluate_noise(base_data, args, client, noise_type, sigma)
            acc_df["noise_type"] = noise_type
            acc_df["sigma"] = sigma
            results.append(acc_df)

    final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    final_df.to_json(outfile, orient="records", lines=True)
    logger.info(f"Wrote noise results to {outfile} (JSONL)")


def main() -> None:
    args = parse(NoiseArgs)
    run(args)


if __name__ == "__main__":
    main()
