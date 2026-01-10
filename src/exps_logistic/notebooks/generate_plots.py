#!/usr/bin/env python3
"""Generate main plots for extended kinds logistic experiment."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, pearsonr, wilcoxon

# Base paths
LOGISTIC_RESULTS_DIR = Path("/nlpgpu/data/terry/ToolProj/src/exps_logistic/results")
PERF_RESULTS_DIR = Path("/nlpgpu/data/terry/ToolProj/src/exps_performance/results")
PLOTS_DIR = Path("/nlpgpu/data/terry/ToolProj/src/exps_logistic/notebooks")

FILENAME_RE = re.compile(r"^(?P<model>.+)_seed(?P<seed>\d+)_(?P<rep>nl|code)_(?P<feats>[^_]+)-(?P<embed>[^_]+)_(?P<ts>\d{8}_\d{6})\.json$")


def parse_logistic_filename(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return m.group("model"), int(m.group("seed")), m.group("rep")


def main():
    # Only load most recent files (today's run) to get extended kinds results
    logistic_dfs = {}
    latest_files = {}

    for path in sorted(LOGISTIC_RESULTS_DIR.glob("*.json")):
        parsed = parse_logistic_filename(path)
        if not parsed:
            continue
        model, seed, rep = parsed
        # Only use files from today's extended run (starting with 16 or 17)
        if "20260109_16" in path.name or "20260109_17" in path.name:
            key = (model, seed, rep)
            # Keep only the most recent file for each (model, seed, rep)
            if key not in latest_files or path.name > latest_files[key]:
                latest_files[key] = path.name
                with path.open("r") as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                df["model"] = model
                df["seed"] = seed
                df["rep"] = rep
                logistic_dfs[key] = df

    print(f"Loaded {len(logistic_dfs)} logistic result files")
    print(f"Models: {sorted(set(k[0] for k in logistic_dfs.keys()))}")

    # Load performance accuracy (excluding gsm8k)
    REP_TO_COL = {"nl": "nl_correct", "code": "code_correct"}
    perf_accuracy = {}

    for model, seed, _ in sorted({(m, s, r)[:2] + ("",) for m, s, r in logistic_dfs.keys()}):
        perf_dir = PERF_RESULTS_DIR / f"{model}_seed{seed}"
        res_files = sorted(perf_dir.glob("tb/run_*/res.jsonl"))
        if not res_files:
            continue
        perf_df_list = []
        for rf in res_files:
            try:
                perf_df_list.append(pd.read_json(rf, lines=True))
            except Exception:
                pass
        if not perf_df_list:
            continue
        perf_df = pd.concat(perf_df_list, ignore_index=True)
        perf_df = perf_df[perf_df["kind"] != "gsm8k"]

        for rep, col in REP_TO_COL.items():
            if col not in perf_df.columns:
                continue
            by_kind = perf_df.groupby("kind")[col].mean()
            acc = by_kind.mean() if len(by_kind) else float("nan")
            perf_accuracy[(model, seed, rep)] = acc

    print(f"Loaded performance accuracy for {len(perf_accuracy)} (model, seed, rep) combinations")

    # Merge and build summary
    merged_dfs = {}
    for key, df in logistic_dfs.items():
        acc = perf_accuracy.get(key, float("nan"))
        df = df.copy()
        df["accuracy_performance"] = acc
        merged_dfs[key] = df

    # Build summary dataframe
    def build_summary_df(merged):
        rows = []
        for (model, seed, rep), df in merged.items():
            metrics_row = df[df["mutual_info_lower_bound_bits"].notna()].tail(1)
            if metrics_row.empty:
                continue
            mi_bits = float(metrics_row["mutual_info_lower_bound_bits"].iloc[0])
            acc_perf = float(df["accuracy_performance"].iloc[0])
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "rep": rep,
                    "mutual_info_lower_bound_bits": mi_bits,
                    "accuracy_performance": acc_perf,
                }
            )
        return pd.DataFrame(rows)

    summary_corr_df = build_summary_df(merged_dfs)
    print(f"Summary has {len(summary_corr_df)} entries")
    print(summary_corr_df[["model", "rep", "mutual_info_lower_bound_bits", "accuracy_performance"]].head(10))

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Plot 1: Boxplot with p-values: Code vs NL MI lower bound
    fig, ax = plt.subplots(figsize=(6, 6))
    mi_code = summary_corr_df[summary_corr_df["rep"] == "code"]["mutual_info_lower_bound_bits"].dropna()
    mi_nl = summary_corr_df[summary_corr_df["rep"] == "nl"]["mutual_info_lower_bound_bits"].dropna()

    palette = sns.color_palette("Set2", n_colors=2)
    colors = [palette[0], palette[1]]

    bp = ax.boxplot([mi_code, mi_nl], tick_labels=["code", "nl"], patch_artist=True, widths=0.6)
    bp["boxes"][0].set_facecolor(colors[0])
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(colors[1])
    bp["boxes"][1].set_alpha(0.7)

    mi_code_mean = mi_code.mean()
    mi_nl_mean = mi_nl.mean()
    mean_diff = mi_code_mean - mi_nl_mean

    paired_df_box = summary_corr_df.pivot_table(index=["model", "seed"], columns="rep", values="mutual_info_lower_bound_bits").dropna(
        subset=["code", "nl"]
    )

    p_value = None
    if len(paired_df_box) >= 3:
        statistic, p_value = wilcoxon(paired_df_box["code"], paired_df_box["nl"], alternative="two-sided")
        max_height = max(mi_code.max(), mi_nl.max())
        y_pval = max_height + 0.1 * max_height
        ax.plot([1, 1, 2, 2], [max_height + 0.05 * max_height, y_pval, y_pval, max_height + 0.05 * max_height], color="steelblue", linewidth=1.5)
        ax.text(1.5, y_pval + 0.02 * max_height, f"p={p_value:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold", color="steelblue")
        ax.text(
            1.5,
            max_height * 0.5,
            f"Δ = {mean_diff:.3f} bits",
            ha="center",
            va="center",
            fontsize=11,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )

    ax.set_ylabel("MI Lower Bound (bits)", fontsize=12)
    ax.set_xlabel("Representation", fontsize=12)
    ax.set_title("Variational Lower Bound: Code vs NL (Extended)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "boxplot_extended.png", dpi=150)
    print(f"Saved boxplot to {PLOTS_DIR / 'boxplot_extended.png'}")
    plt.close()

    # Plot 2: KDE density plot
    fig, ax = plt.subplots(figsize=(6, 2.5))
    palette = sns.color_palette("Set2", n_colors=2)
    colors = [palette[0], palette[1]]

    if len(mi_code) > 1 and len(mi_nl) > 1:
        kde_code = gaussian_kde(mi_code)
        kde_nl = gaussian_kde(mi_nl)
        x_range = np.linspace(min(mi_code.min(), mi_nl.min()) - 0.5, max(mi_code.max(), mi_nl.max()) + 0.5, 200)

        ax.plot(x_range, kde_code(x_range), color=colors[0], linewidth=4, label="code", alpha=0.8)
        ax.plot(x_range, kde_nl(x_range), color=colors[1], linewidth=4, label="nl", alpha=0.8)

        ax.axvline(mi_code.mean(), color=colors[0], linestyle="--", linewidth=2.5, alpha=0.7)
        ax.axvline(mi_nl.mean(), color=colors[1], linestyle="--", linewidth=2.5, alpha=0.7)

        if p_value is not None:
            ax.text(
                0.98,
                0.15,
                f"Δ = {mean_diff:.3f} bits\np = {p_value:.4f}",
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="black", linewidth=1.5),
            )

    ax.set_ylabel("Density", fontsize=14)
    ax.set_xlabel("Mutual information lower bound (bits)", fontsize=14)
    ax.set_title("Variational Lower Bound: Code vs NL (Extended)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kde_extended.png", dpi=150)
    print(f"Saved KDE plot to {PLOTS_DIR / 'kde_extended.png'}")
    plt.close()

    # Plot 3: MI vs Accuracy scatter
    clean_df = summary_corr_df.dropna(subset=["mutual_info_lower_bound_bits", "accuracy_performance"])

    if len(clean_df) >= 2:
        r, p = pearsonr(clean_df["mutual_info_lower_bound_bits"], clean_df["accuracy_performance"])
        x = clean_df["mutual_info_lower_bound_bits"].to_numpy()
        y = clean_df["accuracy_performance"].to_numpy()

        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        plt.figure(figsize=(4.5, 3))
        scatter = plt.scatter(x, y, c=y, cmap="magma", alpha=0.8, s=90, edgecolors="white", linewidth=0.6, label="runs")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Accuracy", fontsize=11)

        plt.plot(x_line, y_line, color=sns.color_palette("dark")[2], linewidth=2, label=f"fit (slope={slope:.3f})")
        plt.xlabel("Mutual information lower bound (bits)", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.title("MI vs. Accuracy (Extended)", fontsize=16, fontweight="bold")
        plt.legend(fontsize=11, loc="upper left", framealpha=0.9)
        plt.grid(True, alpha=0.2)
        plt.annotate(
            f"r={r:.3f}\np={p:.3g}",
            xy=(0.97, 0.05),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            fontsize=12,
        )
        plt.tick_params(labelsize=12)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "mi_vs_accuracy_extended.png", dpi=150)
        print(f"Saved MI vs accuracy plot to {PLOTS_DIR / 'mi_vs_accuracy_extended.png'}")
        print(f"Pearson r = {r:.3f}, p = {p:.3g}, N = {len(clean_df)}")
        plt.close()

    # Plot 4: Contrast plot with scattered x positions by group
    paired_df = summary_corr_df.pivot_table(index=["model", "seed"], columns="rep", values="mutual_info_lower_bound_bits").reset_index()
    paired_df = paired_df.dropna(subset=["code", "nl"])
    paired_df["difference"] = paired_df["code"] - paired_df["nl"]

    if len(paired_df) >= 2:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

        models = sorted(paired_df["model"].unique())
        n_models = len(models)
        palette_colors = sns.color_palette("Set2", n_colors=len(models))
        color_map = {model: palette_colors[i] for i, model in enumerate(models)}

        x_positions = []
        y_values = []
        colors = []

        np.random.seed(42)
        for i, (idx, row) in enumerate(paired_df.iterrows()):
            model = row["model"]
            model_idx = list(models).index(model)
            jitter = np.random.uniform(-0.15, 0.15)
            x_pos = model_idx + 1 + jitter
            x_positions.append(x_pos)
            y_values.append(row["difference"])
            colors.append(color_map[model])

        ax.scatter(x_positions, y_values, c=colors, alpha=0.8, s=100, edgecolors="black", linewidth=2.0, zorder=3)

        mean_diff = paired_df["difference"].mean()
        if len(paired_df) >= 3:
            statistic, p_value = wilcoxon(paired_df["code"], paired_df["nl"], alternative="two-sided")

        within_group_means = []
        within_group_x = []
        within_group_colors = []
        for model in models:
            model_data = paired_df[paired_df["model"] == model]["difference"]
            if len(model_data) > 0:
                within_group_means.append(model_data.mean())
                model_idx = list(models).index(model)
                within_group_x.append(model_idx + 1)
                within_group_colors.append(color_map[model])

        ax.scatter(
            within_group_x,
            within_group_means,
            marker="^",
            s=200,
            c=within_group_colors,
            edgecolors="black",
            linewidth=3.0,
            zorder=5,
            label="Within-group mean",
        )

        ax.axhline(0, color="black", linestyle="-", linewidth=4.5, alpha=0.9, label="Zero difference", zorder=2)
        mean_line = ax.axhline(
            mean_diff, color=sns.color_palette("dark")[2], linestyle="--", linewidth=4.5, alpha=1.0, label=f"Overall mean={mean_diff:.3f}", zorder=2
        )

        ax.set_xticks(range(1, n_models + 1))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("Model", fontsize=11, fontweight="bold")
        ax.set_ylabel("Contrast (Code - NL) in bits", fontsize=11, fontweight="bold")
        ax.set_title("Variational Lower Bound Contrasts (Extended)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y", linewidth=1.0)
        ax.set_ylim(-0.05, None)

        legend_elements = [Patch(facecolor=color_map[model], label=model, edgecolor="black", linewidth=1.5) for model in models]
        triangle_legend = Line2D([0], [0], marker="^", color="black", linestyle="None", markersize=10, markeredgewidth=2.5, label="Within-group mean")
        ax.legend(handles=legend_elements + [triangle_legend, mean_line], loc="upper right", fontsize=6, frameon=True, framealpha=0.95)

        plt.subplots_adjust(right=0.97, left=0.15, top=0.94, bottom=0.25)
        plt.savefig(PLOTS_DIR / "contrast_extended.png", dpi=150)
        print(f"Saved contrast plot to {PLOTS_DIR / 'contrast_extended.png'}")
        plt.close()

    # =========================================================================
    # Label Distribution Plots
    # =========================================================================

    # Combine all prediction data to analyze label distributions
    all_preds = []
    for key, df in logistic_dfs.items():
        # Filter out the summary row (which has NaN in 'kind')
        pred_df = df[df["kind"].notna()].copy()
        if len(pred_df) > 0:
            all_preds.append(pred_df)

    if all_preds:
        combined_df = pd.concat(all_preds, ignore_index=True)
        # Remove duplicates (same sample may appear in multiple runs)
        combined_df = combined_df.drop_duplicates(subset=["kind", "digits", "prompt", "true_label"])

        print("\n=== Label Distribution Analysis ===")
        print(f"Total unique samples: {len(combined_df)}")
        print(f"Unique labels: {combined_df['true_label'].nunique()}")
        print(f"Unique kinds: {combined_df['kind'].nunique()}")

        # Parse true_label to extract kind, digits, bin
        def parse_label(label):
            if pd.isna(label):
                return None, None, None
            parts = str(label).split("|")
            kind = parts[0] if len(parts) > 0 else None
            digits = None
            bin_val = None
            for p in parts[1:]:
                if p.startswith("d"):
                    try:
                        digits = int(p[1:])
                    except Exception:
                        pass
                elif p.startswith("b"):
                    bin_val = p[1:]
            return kind, digits, bin_val

        parsed = combined_df["true_label"].apply(parse_label)
        combined_df["label_kind"] = [p[0] for p in parsed]
        combined_df["label_digits"] = [p[1] for p in parsed]
        combined_df["label_bin"] = [p[2] for p in parsed]

        # Plot 5: Distribution by Kind (horizontal bar chart)
        fig, ax = plt.subplots(figsize=(10, 8))
        kind_counts = combined_df["kind"].value_counts().sort_values(ascending=True)

        # Color by category
        fg_kinds = {"add", "sub", "mul", "lcs", "knap", "rod", "ilp_assign", "ilp_prod", "ilp_partition"}
        nphard_kinds = {"edp", "gcp", "ksp", "spp", "tsp"}

        colors = []
        for kind in kind_counts.index:
            if kind in fg_kinds:
                colors.append(sns.color_palette("Set2")[0])  # Green for fine-grained
            elif kind in nphard_kinds:
                colors.append(sns.color_palette("Set2")[1])  # Orange for NP-hard
            else:
                colors.append(sns.color_palette("Set2")[2])  # Blue for CLRS

        ax.barh(kind_counts.index, kind_counts.values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Count", fontsize=12)
        ax.set_ylabel("Problem Kind", fontsize=12)
        ax.set_title("Sample Distribution by Problem Kind (Extended)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        # Add legend
        legend_elements = [
            Patch(facecolor=sns.color_palette("Set2")[0], label="Fine-grained", edgecolor="black"),
            Patch(facecolor=sns.color_palette("Set2")[1], label="NP-hard", edgecolor="black"),
            Patch(facecolor=sns.color_palette("Set2")[2], label="CLRS", edgecolor="black"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "label_dist_by_kind.png", dpi=150)
        print(f"Saved kind distribution to {PLOTS_DIR / 'label_dist_by_kind.png'}")
        plt.close()

        # Plot 6: Distribution by Digits (for fine-grained problems)
        fg_df = combined_df[combined_df["kind"].isin(fg_kinds)]
        if len(fg_df) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            digit_counts = fg_df["digits"].value_counts().sort_index()

            ax.bar(
                digit_counts.index.astype(str),
                digit_counts.values,
                color=sns.color_palette("viridis", len(digit_counts)),
                edgecolor="black",
                linewidth=0.5,
            )
            ax.set_xlabel("Digits", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title("Fine-grained Sample Distribution by Digits", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "label_dist_by_digits.png", dpi=150)
            print(f"Saved digits distribution to {PLOTS_DIR / 'label_dist_by_digits.png'}")
            plt.close()

        # Plot 7: Heatmap of Kind x Digits (for fine-grained)
        if len(fg_df) > 0:
            pivot_df = fg_df.groupby(["kind", "digits"]).size().unstack(fill_value=0)

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot_df, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5, cbar_kws={"label": "Count"})
            ax.set_xlabel("Digits", fontsize=12)
            ax.set_ylabel("Problem Kind", fontsize=12)
            ax.set_title("Fine-grained: Kind × Digits Distribution", fontsize=14, fontweight="bold")

            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "label_dist_heatmap.png", dpi=150)
            print(f"Saved heatmap to {PLOTS_DIR / 'label_dist_heatmap.png'}")
            plt.close()

        # Plot 8: Top 30 most frequent labels
        fig, ax = plt.subplots(figsize=(10, 8))
        top_labels = combined_df["true_label"].value_counts().head(30).sort_values(ascending=True)

        # Color by kind category
        label_colors = []
        for label in top_labels.index:
            kind = str(label).split("|")[0]
            if kind in fg_kinds:
                label_colors.append(sns.color_palette("Set2")[0])
            elif kind in nphard_kinds:
                label_colors.append(sns.color_palette("Set2")[1])
            else:
                label_colors.append(sns.color_palette("Set2")[2])

        ax.barh(range(len(top_labels)), top_labels.values, color=label_colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(top_labels)))
        ax.set_yticklabels(top_labels.index, fontsize=8)
        ax.set_xlabel("Count", fontsize=12)
        ax.set_ylabel("Label (kind|digits|bin)", fontsize=12)
        ax.set_title("Top 30 Most Frequent Labels", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "label_dist_top30.png", dpi=150)
        print(f"Saved top 30 labels to {PLOTS_DIR / 'label_dist_top30.png'}")
        plt.close()

        # Plot 9: Label frequency distribution (log-log)
        fig, ax = plt.subplots(figsize=(8, 5))
        label_counts = combined_df["true_label"].value_counts()

        # Plot rank vs frequency
        ranks = np.arange(1, len(label_counts) + 1)
        freqs = label_counts.values

        ax.scatter(ranks, freqs, alpha=0.6, s=20, color=sns.color_palette("deep")[0])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rank", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Label Frequency Distribution (Zipf-like)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add annotation
        ax.annotate(
            f"Total labels: {len(label_counts)}\nMax freq: {freqs[0]}\nMin freq: {freqs[-1]}",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=10,
        )

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "label_dist_zipf.png", dpi=150)
        print(f"Saved Zipf plot to {PLOTS_DIR / 'label_dist_zipf.png'}")
        plt.close()

        # Plot 10: Distribution by category (pie chart)
        fig, ax = plt.subplots(figsize=(8, 8))

        category_counts = {
            "Fine-grained": len(combined_df[combined_df["kind"].isin(fg_kinds)]),
            "NP-hard": len(combined_df[combined_df["kind"].isin(nphard_kinds)]),
            "CLRS": len(combined_df[~combined_df["kind"].isin(fg_kinds | nphard_kinds)]),
        }

        colors = [sns.color_palette("Set2")[0], sns.color_palette("Set2")[1], sns.color_palette("Set2")[2]]
        wedges, texts, autotexts = ax.pie(
            category_counts.values(),
            labels=category_counts.keys(),
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            wedgeprops={"edgecolor": "black", "linewidth": 1},
        )
        ax.set_title("Sample Distribution by Category", fontsize=14, fontweight="bold")

        # Add counts in legend
        legend_labels = [f"{k}: {v:,}" for k, v in category_counts.items()]
        ax.legend(wedges, legend_labels, loc="lower right", fontsize=10)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "label_dist_pie.png", dpi=150)
        print(f"Saved pie chart to {PLOTS_DIR / 'label_dist_pie.png'}")
        plt.close()

        # Print summary statistics
        print("\n=== Label Distribution Summary ===")
        print(f"Fine-grained samples: {category_counts['Fine-grained']:,}")
        print(f"NP-hard samples: {category_counts['NP-hard']:,}")
        print(f"CLRS samples: {category_counts['CLRS']:,}")
        print(f"Total unique labels: {combined_df['true_label'].nunique()}")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
