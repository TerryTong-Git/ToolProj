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
    closed_providers = {"anthropic", "openai", "google", "xai"}
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


analysis()
