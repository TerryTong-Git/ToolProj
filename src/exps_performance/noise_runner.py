from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

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


# Kind presets for noise robustness experiments
FG_KINDS = ["add", "sub", "mul", "lcs", "knap", "rod"]
CLRS_SUBSET_KINDS = ["binary_search", "bellman_ford", "dijkstra", "dfs", "segments_intersect"]
NPHARD_KINDS = ["gcp", "spp", "tsp"]
EXPERIMENT_KINDS = FG_KINDS + CLRS_SUBSET_KINDS + NPHARD_KINDS  # 14 total

# Finer sigma levels for capturing degradation curves
FINE_SIGMA_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


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
    kinds: List[str] = field(default_factory=lambda: EXPERIMENT_KINDS.copy())
    digits_list: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    n: int = 10  # 10 samples per condition for statistical power
    sigma: List[float] = field(default_factory=lambda: FINE_SIGMA_LEVELS.copy())
    noise_types: List[str] = field(default_factory=lambda: list(NOISE_FUNCS.keys()))
    root: str = "results_noise_v3"
    save_path: str | None = None
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Pilot mode for quick calibration runs
    pilot: bool = False


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


def _default_save_path(args: NoiseArgs) -> Path:
    """Construct a save path similar to logistic runs, enriched with noise metadata."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = str(args.model).replace("/", "-")
    seed = str(args.seed)
    outdir = _ensure_outdir(args.root)
    fname = f"{model}_seed{seed}_noise_{ts}.json"
    return outdir / fname


def _serialize_results(final_df: pd.DataFrame, args: NoiseArgs, save_path: Path) -> None:
    """Write records plus a summary block containing run metadata and noise settings."""
    records = final_df.to_dict(orient="records")
    summary = {
        "model": args.model,
        "backend": args.backend,
        "seed": args.seed,
        "noise_types": args.noise_types,
        "noise_levels": [float(clamp_sigma(s)) for s in args.sigma],
        "kinds": args.kinds,
        "digits_list": args.digits_list,
        "n": args.n,
        "timestamp": datetime.now().isoformat(),
        "path": str(save_path),
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(records + [summary], ensure_ascii=False))


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
    # Apply pilot mode overrides for quick calibration
    if args.pilot:
        logger.info("PILOT MODE: Using reduced configuration for calibration")
        args.kinds = ["add", "sub", "binary_search"]  # 3 kinds only
        args.digits_list = [2, 4]  # 2 digit levels only
        args.n = 4  # Fewer samples
        args.sigma = [0.0, 0.05, 0.10, 0.15, 0.20]  # Finer focus on early degradation
        args.root = "results_noise_pilot"

    seed_all_and_setup(args)
    client = llm(args)
    logger.info(f"Using backend={args.backend} model={args.model}")
    logger.info(f"Kinds: {args.kinds}")
    logger.info(f"Digits: {args.digits_list}")
    logger.info(f"Sigma levels: {args.sigma}")
    logger.info(f"Noise types: {args.noise_types}")
    logger.info(f"n={args.n} samples per condition")

    base_data = list(make_dataset(args.kinds, args.n, args.digits_list))
    logger.info(f"Generated {len(base_data)} base questions")

    results: List[pd.DataFrame] = []
    total_conditions = len(args.noise_types) * len(args.sigma)
    current = 0
    for noise_type in args.noise_types:
        for sigma in args.sigma:
            current += 1
            logger.info(f"[{current}/{total_conditions}] Running noise_type={noise_type} sigma={sigma}")
            acc_df = _evaluate_noise(base_data, args, client, noise_type, sigma)
            acc_df["noise_type"] = noise_type
            acc_df["sigma"] = sigma
            results.append(acc_df)

    final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    save_path = Path(args.save_path) if args.save_path else _default_save_path(args)
    _serialize_results(final_df, args, save_path)
    logger.info(f"Wrote noise results to {save_path} (JSON)")
    logger.info(f"Total records: {len(final_df)}")


def main() -> None:
    args = parse(NoiseArgs)
    run(args)


if __name__ == "__main__":
    main()
