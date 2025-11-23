import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.exps_performance.logger import create_big_df, walk_results_folder

ALGOS_AND_LENGTHS = {
    "articulation_points": [4, 5, 10, 11, 12, 15, 19],
    "activity_selector": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    "bellman_ford": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    "bfs": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    "binary_search": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    "bridges": [4, 5],
    "bubble_sort": [4, 5, 10],
    "dag_shortest_paths": [4, 5, 10, 11, 12, 15, 19],
    "dfs": [4, 5, 10, 11, 12, 15, 19, 23],
    "dijkstra": [4, 5, 10, 11, 12, 15, 19, 23, 28],
    "find_maximum_subarray_kadane": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    "floyd_warshall": [4, 5, 10],
    "graham_scan": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    "heapsort": [4, 5, 10],
    "insertion_sort": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    "jarvis_march": [4, 5, 10, 11, 12],
    "kmp_matcher": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    "lcs_length": [4, 5, 10],
    "matrix_chain_order": [4, 5, 10],
    "minimum": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    "mst_kruskal": [4, 5, 10],
    "mst_prim": [4, 5, 10, 11, 12, 15, 19, 23, 28],
    "naive_string_matcher": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    "optimal_bst": [4, 5, 10],
    "quickselect": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    "quicksort": [4, 5, 10],
    "segments_intersect": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    "strongly_connected_components": [4, 5, 10, 11, 12, 15],
    "task_scheduling": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
    "topological_sort": [4, 5, 10, 11, 12, 15, 19, 23],
}


def plot_main_fig(df):
    train_lengths_dict = {}
    for alg, train_length in ALGOS_AND_LENGTHS.items():
        train_lengths_dict[alg] = np.array(train_length)
    sns.reset_defaults()

    hue_order = ["NL", "Sim", "Code", "ControlSim"]

    g = sns.FacetGrid(df, col="algorithm", col_wrap=6, hue="experiment", hue_order=hue_order, sharex=False)
    g.map(sns.lineplot, "test_length", "accuracy")
    g.set_titles("{col_name}")

    max_len = df.groupby("algorithm").test_length.max()

    for ax in g.axes:
        alg = ax.title.get_text()
        ax.set_title(alg.replace("_", " "))
        train_lengths = train_lengths_dict[alg]
        ax.scatter(train_lengths, np.ones(len(train_lengths)) + 0.05, color="red", s=1.0)
        ax.set_xlim(None, max_len[alg])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    g.axes[11].legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), fontsize=9, title="")
    g.set_xlabels("test length")
    plt.show()


def analysis():
    files = walk_results_folder("/nlpgpu/data/terry/ToolProj/src/exps_performance/results")  # check files are deepseek and gemma, seed 1 and 2
    df = create_big_df(files)
    rows = df.to_dict("records")
    return rows


analysis()
