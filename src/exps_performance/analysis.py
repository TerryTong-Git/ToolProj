import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import wilcoxon

from src.exps_performance.logger import create_big_df, walk_results_folder


def plot_main_fig(df):
    # train_lengths_dict = {}
    # for alg, train_length in _DEFAULT_VAL_ALGOS_AND_LENGTHS.items():
    #     train_lengths_dict[alg] = np.array(train_length)
    sns.reset_defaults()
    # import pdb; pdb.set_trace()
    df1 = df[df["model"].isin(["Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501"])]
    # df1 = df
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
        train_lengths = [2, 4, 8, 16]
        ax.scatter(train_lengths, np.ones(len(train_lengths)) + 0.05, color="red", s=1.0)
        ax.set_xlim(None, 16)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    g.axes[7].legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), fontsize=9, title="")
    g.set_xlabels("test length")
    plt.show()
    plt.savefig("main.png")


def plot_v_graph(df):
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
    df1 = df[df["kind"].isin(["add", "mul", "lcs", "rod", "knap", "ilp_assign", "ilp_prod", "ilp_partition"])]

    melted_df = pd.melt(df1, value_vars=cols, id_vars=["model"])
    # import pdb; pdb.set_trace()
    sns.pointplot(
        data=melted_df,
        ax=ax,
        x="variable",
        y="value",
        hue="model",
        linestyle="",
        alpha=0.8,
        marker="^",
        palette=sorted(sns.color_palette("tab20", n_colors=5), key=lambda x: x[0] - x[2]),
        errorbar=None,
    )
    sns.lineplot(
        data=melted_df, ax=ax, x="variable", y="value", color="black", marker="o", markersize=10, fillstyle="full", label="All models", errorbar=None
    )
    # can classify by size
    plt.ylabel("Accuracy")
    handles, labels = plt.gca().get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title="Model",
        markerscale=1.3,
        fontsize="large",
        title_fontsize="x-large",
    )
    ax.set_ylim([0, 1])
    plt.xlabel("Arm")
    ax.set_xticklabels(["NL", "Sim", "ControlSim", "Code"])
    plt.savefig("line", bbox_inches="tight")


def wilcoxon_test(mdf, complexity_pairs):
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


def plot_p_vals(df):
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
    # df1 = df
    df1 = df[df["model"].isin(["Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501"])]
    df2 = df1
    # df2 = df1[df1["kind"].isin(["add", "mul", "lcs", "rod", "knap", "ilp_assign", "ilp_prod", "ilp_partition"])]
    name_map = {
        "nl_correct": "Arm 1 \n (NL)",
        "sim_correct": "Arm 2 \n (Code Sim)",
        "controlsim_correct": "Arm 2.5 \n (Controlled Code Sim)",
        "code_correct": "Arm 3 \n (Code Exec)",
    }
    dfnew = df2.rename(columns=name_map)

    cols = list(name_map.values())
    mdf = pd.melt(dfnew, value_vars=cols, id_vars=["model", "kind"])
    fig, ax = plt.subplots(figsize=(6, 6))
    # mdf = mdf.sort_values(by=["variable"], key=lambda x: x.map({color: ind for color,ind in zip(range(len(colors)), list(colors.keys()))}))
    # import pdb; pdb.set_trace()
    mdf1 = mdf.groupby(["variable", "model", "kind"]).mean().reset_index()
    mdf2 = mdf1.drop(["model", "kind"], axis=1)

    arm_pairs = list(itertools.combinations(cols, 2))
    # import pdb; pdb.set_trace()
    p_values = wilcoxon_test(mdf2, arm_pairs)
    sns.boxplot(x="variable", y="value", data=mdf2, gap=0.3, palette=sns.color_palette("vlag", n_colors=8)[:4], ax=ax)

    offset = 0
    for i, pair in enumerate(arm_pairs):
        complexity1 = pair[0]
        complexity2 = pair[1]
        order = {x: i for i, x in enumerate(cols)}
        x1 = order[complexity1]
        x2 = order[complexity2]
        y = mdf2["value"].max() + offset
        h = 0.03
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color="steelblue")
        ax.text((x1 + x2) * 0.5, y + h, f"p={p_values[i]:.4f}", ha="center", va="bottom")
        offset += 0.08

    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # Set y-axis ticks
    ax.set_xlabel("Arms")
    ax.set_ylabel("Accuracy")

    ax.set_ylim(0, y + 0.5)
    plt.ylim(0, 1.5)
    plt.title("Accuracy across Arms on Fine-grained Tasks")
    plt.tight_layout()
    plt.show()
    plt.savefig("pval", bbox_inches="tight")


def analysis():
    files = walk_results_folder("/nlpgpu/data/terry/ToolProj/src/exps_performance/results")  # check files are deepseek and gemma, seed 1 and 2
    df = create_big_df(files)
    plot_p_vals(df)
    plot_main_fig(df)
    plot_v_graph(df)

    # rows = df.to_dict("records")
    # import pdb

    # pdb.set_trace()
    # return rows


analysis()
