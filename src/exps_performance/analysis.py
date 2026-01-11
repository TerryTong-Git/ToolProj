import itertools
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import wilcoxon

from src.exps_performance.logger import create_big_df

goodset = [
    "clrs30",
    "add",
    "sub",
    "mul",
    "lcs",
    "rod",
    "knap",
    "ilp_assign",
    "ilp_prod",
    "ilp_partition",
    "spp",
    "tsp",
    "tsp_d",
    "msp",
    "ksp",
    "gcp",
    "gcp_d",
    "bsp",
    "edp",
]


def plot_main_fig(df: pd.DataFrame) -> None:
    # train_lengths_dict = {}
    # for alg, train_length in _DEFAULT_VAL_ALGOS_AND_LENGTHS.items():
    #     train_lengths_dict[alg] = np.array(train_length)
    sns.reset_defaults()
    # import pdb; pdb.set_trace()
    df1 = df
    df2 = df1[df1["kind"].isin(["add", "mul", "lcs", "rod", "knap", "ilp_assign", "ilp_prod", "ilp_partition"])]
    name_map = {
        "nl_correct": "Arm 1 \n (NL)",
        "sim_correct": "Arm 2 \n (Code Sim)",
        "controlsim_correct": "Arm 2.5 \n (Controlled Code Sim)",
        "code_correct": "Arm 3 \n (Code Exec)",
    }
    dfnew = df2.rename(columns=name_map)

    cols = list(name_map.values())
    mdf = pd.melt(dfnew, value_vars=cols, id_vars=["kind", "digit"])
    # mdf1 = mdf.groupby(["variable", "digit", "kind"]).mean().reset_index()
    g = sns.FacetGrid(mdf, col="kind", col_wrap=4, hue="variable", hue_order=cols, sharex=False)
    g.map(sns.lineplot, "digit", "value")
    g.set_titles("{col_name}")

    for ax in g.axes:
        alg = ax.title.get_text()
        ax.set_title(alg.replace("_", " "))
        # train_lengths = train_lengths_dict[alg]
        train_lengths = [2, 4, 8, 10, 12, 14, 16, 18, 20]
        ax.scatter(train_lengths, np.ones(len(train_lengths)) + 0.05, color="red", s=1.0)
        ax.set_xlim(None, 20)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    g.axes[7].legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), fontsize=9, title="")
    g.set_xlabels("test length")
    plt.show()
    plt.savefig("figures/main.png")


def plot_v_graph(df: pd.DataFrame) -> None:
    from matplotlib import rcParams

    rcParams["figure.dpi"] = 500
    rcParams["savefig.dpi"] = 500
    rcParams["font.family"] = "Arial"
    rcParams["axes.labelsize"] = 18
    rcParams["axes.titlesize"] = 18
    rcParams["legend.fontsize"] = 18
    rcParams["figure.titlesize"] = 18
    rcParams["markers.fillstyle"] = "none"
    fig, ax = plt.subplots(figsize=(6, 6))
    cols = ["nl_correct", "sim_correct", "controlsim_correct", "code_correct"]
    df1 = df[df["kind"].isin(["gsm8k"])]
    melted_df = pd.melt(df1, value_vars=cols, id_vars=["model", "kind"])
    # Classify models by provider prefix to split markers between open/closed
    closed_providers = {"anthropic", "openai", "xai"}
    open_providers = {"meta-llama", "mistral", "mistralai", "qwen", "deepseek", "microsoft", "allenai", "zhipuai"}

    def _is_open_model(model_name: str) -> bool:
        prefix = str(model_name).split("/")[0].lower()
        if prefix in closed_providers:
            return False
        if prefix in open_providers:
            return True
        return False

    melted_df = melted_df.copy()
    melted_df["model_type"] = melted_df["model"].apply(lambda m: "open" if _is_open_model(m) else "closed")
    open_models_df = melted_df[melted_df["model_type"] == "open"]
    closed_models_df = melted_df[melted_df["model_type"] == "closed"]

    hue_order = sorted(melted_df["model"].unique())
    palette_base = sorted(sns.color_palette("tab20", n_colors=20), key=lambda x: x[0] - x[2])

    closed_candidates = [m for m in hue_order if not _is_open_model(m)]
    open_candidates = [m for m in hue_order if _is_open_model(m)]

    closed_palette_map = {model: palette_base[i % len(palette_base)] for i, model in enumerate(closed_candidates)}
    # Offset open model colors slightly to reduce closeness with closed-model colors.
    open_palette_rotated = palette_base[2:] + palette_base[:2]
    open_palette_map = {model: open_palette_rotated[i % len(open_palette_rotated)] for i, model in enumerate(open_candidates)}

    palette_map = {**closed_palette_map, **open_palette_map}

    # Keep original coloring; draw separately with different markers but shared palette.
    if not closed_models_df.empty:
        sns.pointplot(
            data=closed_models_df,
            ax=ax,
            x="variable",
            y="value",
            hue="model",
            hue_order=hue_order,
            linestyle="",
            alpha=0.8,
            marker="x",  # closed models -> x
            markersize=12,
            linewidth=1.6,
            palette=palette_map,
            errorbar=None,
            legend=False,
        )
    if not open_models_df.empty:
        sns.pointplot(
            data=open_models_df,
            ax=ax,
            x="variable",
            y="value",
            hue="model",
            hue_order=hue_order,
            linestyle="",
            alpha=0.8,
            marker="o",  # open models -> o
            markersize=12,
            linewidth=1.6,
            palette=palette_map,
            errorbar=None,
            legend=False,
        )
    sns.lineplot(
        data=melted_df, ax=ax, x="variable", y="value", color="black", marker="o", markersize=10, fillstyle="full", label="All models", errorbar=None
    )
    # Build legend: first three closed models (x), then three open models (o), then All models.
    closed_priority = ["openai/gpt-4o-mini", "anthropic/claude-haiku-4.5", "google/gemini-2.5-flash"]
    open_priority = ["mistralai/ministral-14b-2512", "meta-llama/llama-3.1-405b-instruct", "qwen/qwen-2.5-coder-32b-instruct"]

    def _pick_models(priority: list[str], candidates: list[str], k: int) -> list[str]:
        picked: list[str] = []
        for m in priority:
            if m in candidates and m not in picked:
                picked.append(m)
            if len(picked) >= k:
                return picked
        for m in candidates:
            if m not in picked:
                picked.append(m)
            if len(picked) >= k:
                break
        return picked

    closed_candidates = [m for m in hue_order if not _is_open_model(m)]
    open_candidates = [m for m in hue_order if _is_open_model(m)]
    closed_order = _pick_models(closed_priority, closed_candidates, 3)
    open_order = _pick_models(open_priority, open_candidates, 3)

    from matplotlib.lines import Line2D

    handles_custom: list[Line2D] = []
    labels_custom: list[str] = []
    for m in closed_order:
        handles_custom.append(Line2D([0], [0], marker="x", color=palette_map[m], linestyle="", markersize=12, markeredgewidth=1.6))
        labels_custom.append(m)
    for m in open_order:
        handles_custom.append(Line2D([0], [0], marker="o", color=palette_map[m], linestyle="", markersize=12, markeredgewidth=1.6))
        labels_custom.append(m)

    # Append the "All models" entry from the lineplot
    line_handles, line_labels = plt.gca().get_legend_handles_labels()
    for h, label in zip(line_handles, line_labels):
        if label == "All models":
            handles_custom.append(h)
            labels_custom.append(label)
            break

    ax.legend(
        handles_custom,
        labels_custom,
        title="Model (o=open, x=closed)",
        markerscale=1.3,
        fontsize="large",
        title_fontsize="x-large",
    )
    ax.set_ylim([0, 1])
    plt.xlabel("Arm")
    ax.set_xticklabels(["NL", "Sim", "ControlSim", "Code"])
    plt.savefig("figures/line.png", bbox_inches="tight")


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    return (lower, upper)


def plot_v_graph_closed(df: pd.DataFrame) -> None:
    """
    Line figure for closed models only, with bootstrap CIs and p-value annotations.
    Uses CLRS30, NPHard, and fine-grained tasks (excludes gsm8k).
    """
    from matplotlib import rcParams
    from matplotlib.lines import Line2D

    rcParams["figure.dpi"] = 500
    rcParams["savefig.dpi"] = 500
    rcParams["font.family"] = "Arial"
    rcParams["axes.labelsize"] = 18
    rcParams["axes.titlesize"] = 18
    rcParams["legend.fontsize"] = 14
    rcParams["figure.titlesize"] = 18
    rcParams["markers.fillstyle"] = "none"

    fig, ax = plt.subplots(figsize=(8, 7))

    cols = ["nl_correct", "sim_correct", "controlsim_correct", "code_correct"]
    arm_labels = ["NL", "Sim", "ControlSim", "Code"]

    # Define task sets
    CLRS_KINDS = {
        "activity_selector", "articulation_points", "bellman_ford", "bfs",
        "binary_search", "bridges", "bubble_sort", "dag_shortest_paths",
        "dfs", "dijkstra", "find_maximum_subarray_kadane", "floyd_warshall",
        "graham_scan", "heapsort", "insertion_sort", "jarvis_march",
        "kmp_matcher", "lcs_length", "matrix_chain_order", "minimum",
        "mst_kruskal", "mst_prim", "naive_string_matcher", "optimal_bst",
        "quickselect", "quicksort", "segments_intersect",
        "strongly_connected_components", "task_scheduling", "topological_sort",
    }
    NPHARD_KINDS = {"edp", "gcp", "ksp", "spp", "tsp"}
    FG_KINDS = {"add", "sub", "mul", "lcs", "rod", "knap", "ilp_assign", "ilp_prod", "ilp_partition"}

    # Combine all task sets (exclude gsm8k)
    target_kinds = CLRS_KINDS | NPHARD_KINDS | FG_KINDS
    df1 = df[df["kind"].isin(target_kinds)]

    print(f"[plot_v_graph_closed] Filtering to {len(target_kinds)} task kinds")
    print(f"[plot_v_graph_closed] Kinds in data: {sorted(df1['kind'].unique())}")

    # Define closed providers
    closed_providers = {"anthropic", "openai", "google", "xai"}

    def _is_closed_model(model_name: str) -> bool:
        prefix = str(model_name).split("/")[0].lower()
        return prefix in closed_providers

    # Filter to closed models only
    df_closed = df1[df1["model"].apply(_is_closed_model)]

    if df_closed.empty:
        print("[plot_v_graph_closed] No closed models found in data.")
        return

    melted_df = pd.melt(df_closed, value_vars=cols, id_vars=["model", "kind"])
    melted_df = melted_df.copy()

    # Get unique models and set up palette
    unique_models = sorted(melted_df["model"].unique())
    palette_base = sns.color_palette("tab10", n_colors=len(unique_models))
    palette_map = {model: palette_base[i] for i, model in enumerate(unique_models)}

    # Compute aggregated stats per arm with bootstrap CIs
    arm_stats = []
    for col in cols:
        values = df_closed[col].dropna().values
        mean_val = np.mean(values) if len(values) > 0 else np.nan
        ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap=1000, ci=0.95)
        arm_stats.append({
            "arm": col,
            "mean": mean_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })
    stats_df = pd.DataFrame(arm_stats)

    # Plot individual model points
    for model in unique_models:
        model_data = melted_df[melted_df["model"] == model]
        model_means = model_data.groupby("variable")["value"].mean().reindex(cols)
        x_positions = list(range(len(cols)))
        ax.scatter(
            x_positions,
            model_means.values,
            marker="x",
            s=120,
            color=palette_map[model],
            alpha=0.8,
            linewidths=2,
            label=model,
        )

    # Plot aggregated line with bootstrap CI error bars
    x_positions = list(range(len(cols)))
    means = stats_df["mean"].values
    ci_lower = stats_df["ci_lower"].values
    ci_upper = stats_df["ci_upper"].values
    yerr_lower = means - ci_lower
    yerr_upper = ci_upper - means

    ax.errorbar(
        x_positions,
        means,
        yerr=[yerr_lower, yerr_upper],
        fmt="o-",
        color="black",
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        label="All closed models (mean)",
    )

    # Compute and display p-values between adjacent arms only (cleaner visualization)
    # Adjacent pairs: NL-Sim, Sim-ControlSim, ControlSim-Code
    adjacent_pairs = [(cols[i], cols[i + 1]) for i in range(len(cols) - 1)]
    mdf_grouped = melted_df.groupby(["variable", "model"])["value"].mean().reset_index()
    mdf_pivot = mdf_grouped.pivot(index="model", columns="variable", values="value")

    p_values = []
    for pair in adjacent_pairs:
        col1, col2 = pair
        if col1 in mdf_pivot.columns and col2 in mdf_pivot.columns:
            x = mdf_pivot[col1].dropna()
            y = mdf_pivot[col2].dropna()
            common_idx = x.index.intersection(y.index)
            if len(common_idx) >= 2:
                try:
                    stat, p_val = wilcoxon(x.loc[common_idx], y.loc[common_idx])
                except Exception:
                    p_val = 1.0
            else:
                p_val = 1.0
        else:
            p_val = 1.0
        p_values.append(p_val)

    # Draw p-value annotations for adjacent pairs
    max_val = max(ci_upper) if len(ci_upper) > 0 else 1.0
    offset = 0.05
    col_to_x = {col: i for i, col in enumerate(cols)}

    for i, (pair, p_val) in enumerate(zip(adjacent_pairs, p_values)):
        col1, col2 = pair
        x1 = col_to_x[col1]
        x2 = col_to_x[col2]
        y = max_val + offset
        h = 0.02
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, color="steelblue")

        # Format p-value
        if p_val < 0.001:
            p_text = "p<0.001"
        elif p_val < 0.01:
            p_text = f"p={p_val:.3f}"
        else:
            p_text = f"p={p_val:.2f}"
        ax.text((x1 + x2) * 0.5, y + h + 0.01, p_text, ha="center", va="bottom", fontsize=9)
        offset += 0.08

    # Set up axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(arm_labels)
    ax.set_xlabel("Arm")
    ax.set_ylabel("Accuracy")
    # Fix y-axis at [0, 1] for accuracy, extend slightly above for annotations
    y_max = max(1.0, max_val + offset + 0.1)
    ax.set_ylim([0, y_max])
    ax.set_title("Closed Models: Accuracy Across Arms\n(CLRS + NPHard + Fine-grained)")

    # Build legend
    handles_custom: list[Line2D] = []
    labels_custom: list[str] = []
    for model in unique_models:
        handles_custom.append(
            Line2D([0], [0], marker="x", color=palette_map[model], linestyle="", markersize=10, markeredgewidth=2)
        )
        labels_custom.append(model)
    # Add aggregated line entry
    handles_custom.append(
        Line2D([0], [0], marker="o", color="black", linestyle="-", markersize=10, linewidth=2)
    )
    labels_custom.append("All closed models (mean)")

    ax.legend(
        handles_custom,
        labels_custom,
        title="Closed Models",
        loc="upper left",
        fontsize=10,
        title_fontsize=12,
    )

    plt.tight_layout()
    plt.savefig("figures/line_closed.png", bbox_inches="tight")
    print("[plot_v_graph_closed] Saved figures/line_closed.png")


def wilcoxon_test(mdf: pd.DataFrame, complexity_pairs: List[tuple[str, str]]) -> List[float]:
    # Perform Wilcoxon test for each pair
    p_values = []
    for pair in complexity_pairs:
        complexity1 = pair[0]
        complexity2 = pair[1]
        x = mdf[mdf["variable"] == complexity1]["value"]
        y = mdf[mdf["variable"] == complexity2]["value"]
        try:
            stat, p_value = wilcoxon(x, y)
        except Exception:
            p_value = 1.0
        p_values.append(p_value)
    return p_values


def plot_p_vals(df: pd.DataFrame) -> None:
    from matplotlib import rcParams

    rcParams["figure.dpi"] = 500
    rcParams["savefig.dpi"] = 500
    rcParams["font.family"] = "Arial"
    rcParams["axes.labelsize"] = 18
    rcParams["axes.titlesize"] = 18
    rcParams["legend.fontsize"] = 18
    rcParams["figure.titlesize"] = 18
    rcParams["markers.fillstyle"] = "none"
    # df1 = df[df["model"] == "Qwen/Qwen2.5-14B-Instruct"]
    df1 = df
    # df1 = df[df["model"].isin(["Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501"])]
    df2 = df1[df1["kind"].isin(["gsm8k"])]
    name_map = {
        "nl_correct": "Arm 1 \n (NL)",
        "sim_correct": "Arm 2 \n (Code Sim)",
        "controlsim_correct": "Arm 2.5 \n (Controlled Code Sim)",
        "code_correct": "Arm 3 \n (Code Exec)",
    }
    dfnew = df2.rename(columns=name_map)

    # Debug: inspect incoming data
    print("[plot_p_vals] raw df shape:", df.shape)
    print("[plot_p_vals] models:", df["model"].unique())
    print("[plot_p_vals] kinds:", df["kind"].unique())
    print("[plot_p_vals] sample rows:\n", df.head(5))

    cols = list(name_map.values())
    mdf = pd.melt(dfnew, value_vars=cols, id_vars=["model", "kind"])
    fig, ax = plt.subplots(figsize=(6, 6))
    # mdf = mdf.sort_values(by=["variable"], key=lambda x: x.map({color: ind for color,ind in zip(range(len(colors)), list(colors.keys()))}))
    # import pdb; pdb.set_trace()
    mdf1 = mdf.groupby(["variable", "model", "kind"]).mean().reset_index()
    mdf2 = mdf1.drop(["model", "kind"], axis=1)

    # Drop non-finite values to avoid NaN/inf in downstream plotting
    print("[plot_p_vals] grouped sample before finite filter:\n", mdf2.head(10))
    mdf2 = mdf2[np.isfinite(mdf2["value"])]
    print("[plot_p_vals] grouped shape after finite filter:", mdf2.shape)
    print("[plot_p_vals] grouped sample after finite filter:\n", mdf2.head(10))
    if mdf2.empty:
        print("[plot_p_vals] No finite values available for p-value plot after filtering.")
        raise ValueError("No finite values available for p-value plot.")

    arm_pairs = list(itertools.combinations(cols, 2))
    # import pdb; pdb.set_trace()
    p_values = wilcoxon_test(mdf2, arm_pairs)
    sns.boxplot(x="variable", y="value", data=mdf2, gap=0.3, palette=sns.color_palette("vlag", n_colors=8)[:4], ax=ax)

    offset = 0
    max_val = mdf2["value"].max()
    if pd.isna(max_val):
        max_val = 0.0
    for i, pair in enumerate(arm_pairs):
        complexity1 = pair[0]
        complexity2 = pair[1]
        order = {x: i for i, x in enumerate(cols)}
        x1 = order[complexity1]
        x2 = order[complexity2]
        y = max_val + offset
        h = 0.03
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color="steelblue")
        ax.text((x1 + x2) * 0.5, y + h, f"p={p_values[i]:.4f}", ha="center", va="bottom")
        offset += 0.08  # type: ignore[assignment]

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # Set y-axis ticks
    ax.set_xlabel("Arms")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, max(1.0, y + 0.5))
    plt.ylim(0, 1.5)
    plt.title("Accuracy across Arms on Fine-grained Tasks")
    plt.tight_layout()
    plt.show()
    plt.savefig("figures/pval.png", bbox_inches="tight")


models = [
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "openai/gpt-5.1-codex",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/o3-mini",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-distill-llama-70b",
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "xai/grok-code-fast-1",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-4-scout",
    "mistral/devstral-medium",
    "mistral/ministral-14b-2512",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct",
    "qwen/qwen3-coder-30b-a3b-instruct",
    "zhipuai/glm-4.6",
    "allenai/olmo-2-0325-32b-instruct",
    "microsoft/phi-4",
    "microsoft/phi-4-reasoning-plus",
]


def analysis() -> None:
    results_root = Path("/nlpgpu/data/terry/ToolProj/src/exps_performance/results")
    jsonl_files = sorted(results_root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under {results_root}")
    df = create_big_df(jsonl_files)
    # Target models from scaled experiment (plan-1.md)
    df = df[
        df["model"].isin(
            [
                "mistralai/codestral-2508",
                "mistralai/mistral-large-2411",
                "google/gemini-2.0-flash-001",
                "mistralai/mixtral-8x22b-instruct",
            ]
        )
    ]
    plot_p_vals(df)
    plot_main_fig(df)
    plot_v_graph(df)

    # rows = df.to_dict("records")
    # import pdb

    # pdb.set_trace()
    # return rows


def analysis_closed_models() -> None:
    """Run analysis on closed models only with the new line_closed figure."""
    # Use local results directory
    results_root = Path(__file__).parent / "results"
    jsonl_files = sorted(results_root.rglob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under {results_root}")
    df = create_big_df(jsonl_files)
    print(f"[analysis_closed_models] Loaded {len(df)} rows from {len(jsonl_files)} files")
    print(f"[analysis_closed_models] Models: {df['model'].unique()}")
    print(f"[analysis_closed_models] Kinds: {df['kind'].unique()}")

    # Generate the closed models figure
    plot_v_graph_closed(df)


if __name__ == "__main__":
    analysis_closed_models()
