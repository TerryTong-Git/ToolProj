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
    sigma: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    noise_types: List[str] = field(default_factory=lambda: list(NOISE_FUNCS.keys()))
    root: str = "results_noise"
    save_path: str | None = None
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"


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
    seed_all_and_setup(args)
    client = llm(args)
    logger.info(f"Using backend={args.backend} model={args.model}")

    base_data = list(make_dataset(args.kinds, args.n, args.digits_list))

    results: List[pd.DataFrame] = []
    for noise_type in args.noise_types:
        for sigma in args.sigma:
            logger.info(f"Running noise_type={noise_type} sigma={sigma}")
            acc_df = _evaluate_noise(base_data, args, client, noise_type, sigma)
            acc_df["noise_type"] = noise_type
            acc_df["sigma"] = sigma
            results.append(acc_df)

    final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    save_path = Path(args.save_path) if args.save_path else _default_save_path(args)
    _serialize_results(final_df, args, save_path)
    logger.info(f"Wrote noise results to {save_path} (JSON)")


def main() -> None:
    args = parse(NoiseArgs)
    run(args)


if __name__ == "__main__":
    main()
