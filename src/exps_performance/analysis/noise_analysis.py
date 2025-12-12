from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(f"Noise results not found at {results_path}")
    df = pd.read_json(results_path, lines=True)
    required_cols = {"noise_type", "sigma", "arm", "accuracy", "kind"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in results: {missing}")
    return df


def plot_noise_curves(df: pd.DataFrame, outdir: Path) -> None:
    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(df, col="noise_type", hue="arm", col_wrap=2, sharey=True, height=3.5)
    g.map_dataframe(sns.lineplot, x="sigma", y="accuracy", marker="o")
    g.set_titles("{col_name}")
    g.add_legend()
    g.set_axis_labels("sigma", "accuracy")
    g.fig.suptitle("Noise vs Accuracy by Arm", y=1.02)
    outfile = outdir / "noise_vs_accuracy.png"
    g.fig.savefig(outfile, bbox_inches="tight")


def plot_faceted_kinds(df: pd.DataFrame, outdir: Path) -> None:
    sns.set_theme(style="ticks")
    g = sns.FacetGrid(df, col="kind", col_wrap=4, hue="arm", sharey=True, height=3)
    g.map_dataframe(sns.lineplot, x="sigma", y="accuracy", marker="o")
    g.set_titles("{col_name}")
    g.add_legend()
    g.set_axis_labels("sigma", "accuracy")
    outfile = outdir / "noise_by_kind.png"
    g.fig.savefig(outfile, bbox_inches="tight")


def _compute_auc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (noise_type, arm), sub in df.groupby(["noise_type", "arm"]):
        sub_sorted = sub.sort_values("sigma")
        sigma_vals = sub_sorted["sigma"].to_numpy()
        acc_vals = sub_sorted["accuracy"].to_numpy()
        if len(sigma_vals) < 2:
            auc = float(acc_vals.mean()) if len(acc_vals) else 0.0
        else:
            span = sigma_vals.max() - sigma_vals.min()
            raw = float(np.trapz(acc_vals, sigma_vals))
            auc = raw / span if span > 0 else float(acc_vals.mean())
        rows.append({"noise_type": noise_type, "arm": arm, "auc": auc})
    return pd.DataFrame(rows)


def plot_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    auc_df = _compute_auc(df)
    pivot = auc_df.pivot(index="arm", columns="noise_type", values="auc")
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="magma", cbar_kws={"label": "normalized AUC"})
    plt.xlabel("noise_type")
    plt.ylabel("arm")
    plt.title("Robustness (AUC across sigma)")
    outfile = outdir / "noise_heatmap.png"
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot noise vs accuracy curves.")
    parser.add_argument("--results", type=Path, default=Path("results_noise/noise_results.jsonl"))
    parser.add_argument("--outdir", type=Path, default=Path("results_noise"))
    parsed = parser.parse_args(args=args)

    outdir = _ensure_outdir(parsed.outdir)
    df = _load_results(parsed.results)

    plot_noise_curves(df, outdir)
    plot_faceted_kinds(df, outdir)
    plot_heatmap(df, outdir)


if __name__ == "__main__":
    main()
