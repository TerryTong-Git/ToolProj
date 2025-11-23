from src.exps_performance.logger import create_big_df, walk_results_folder

# ALGOS_AND_LENGTHS = {
#     "articulation_points": [4, 5, 10, 11, 12, 15, 19],
#     "activity_selector": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
#     "bellman_ford": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
#     "bfs": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
#     "binary_search": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
#     "bridges": [4, 5],
#     "bubble_sort": [4, 5, 10],
#     "dag_shortest_paths": [4, 5, 10, 11, 12, 15, 19],
#     "dfs": [4, 5, 10, 11, 12, 15, 19, 23],
#     "dijkstra": [4, 5, 10, 11, 12, 15, 19, 23, 28],
#     "find_maximum_subarray_kadane": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
#     "floyd_warshall": [4, 5, 10],
#     "graham_scan": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
#     "heapsort": [4, 5, 10],
#     "insertion_sort": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
#     "jarvis_march": [4, 5, 10, 11, 12],
#     "kmp_matcher": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
#     "lcs_length": [4, 5, 10],
#     "matrix_chain_order": [4, 5, 10],
#     "minimum": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
#     "mst_kruskal": [4, 5, 10],
#     "mst_prim": [4, 5, 10, 11, 12, 15, 19, 23, 28],
#     "naive_string_matcher": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
#     "optimal_bst": [4, 5, 10],
#     "quickselect": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
#     "quicksort": [4, 5, 10],
#     "segments_intersect": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
#     "strongly_connected_components": [4, 5, 10, 11, 12, 15],
#     "task_scheduling": [4, 5, 10, 11, 12, 15, 19, 23, 28, 31],
#     "topological_sort": [4, 5, 10, 11, 12, 15, 19, 23],
# }


# def plot_main_fig(df):
#     train_lengths_dict = {}
#     for alg, train_length in ALGOS_AND_LENGTHS.items():
#         train_lengths_dict[alg] = np.array(train_length)
#     sns.reset_defaults()

#     hue_order = ["NL", "Sim", "Code", "ControlSim"]

#     g = sns.FacetGrid(df, col="algorithm", col_wrap=6, hue="experiment", hue_order=hue_order, sharex=False)
#     g.map(sns.lineplot, "test_length", "accuracy")
#     g.set_titles("{col_name}")

#     max_len = df.groupby("algorithm").test_length.max()

#     for ax in g.axes:
#         alg = ax.title.get_text()
#         ax.set_title(alg.replace("_", " "))
#         train_lengths = train_lengths_dict[alg]
#         ax.scatter(train_lengths, np.ones(len(train_lengths)) + 0.05, color="red", s=1.0)
#         ax.set_xlim(None, max_len[alg])
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))

#     g.axes[11].legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), fontsize=9, title="")
#     g.set_xlabels("test length")
#     plt.show()

# def plot_line_chart(df,  col_name, x_name, x_order_mapper, ax):
#     tmp_df = df
#     tmp_df["x_order"] = tmp_df[x_name].map(x_order_mapper)
#     tmp_df.sort_values(by=["x_order", "model_type"], inplace=True)
#     open_model_df = tmp_df[tmp_df["model_type"] == 0]
#     close_model_df = tmp_df[tmp_df["model_type"] == 1]
#     number_of_close_models = close_model_df["model_name"].nunique()
#     number_of_open_models = open_model_df["model_name"].nunique()

#     # make one red palette and one blue palette
#     palette = sns.color_palette("tab20", n_colors=number_of_close_models + number_of_open_models)
#     palette = sorted(palette, key=lambda x: x[0] - x[2])
#     palette_map = {}

#     sns.pointplot(
#         data=open_model_df,
#         ax=ax,
#         x=x_name,
#         y=col_name,
#         hue="model_name",
#         linestyle="",
#         alpha=0.8,
#         marker="s",
#         palette=palette[:number_of_open_models],
#     )
#     palette_map = {model: palette[i] for i, model in enumerate(open_model_df["model_name"].unique())}

#     sns.pointplot(
#         data=close_model_df,
#         ax=ax,
#         x=x_name,
#         y=col_name,
#         hue="model_name",
#         linestyle="",
#         alpha=0.8,
#         marker="^",
#         palette=palette[number_of_open_models:],
#     )
#     palette_map.update({model: palette[i + number_of_open_models] for i, model in enumerate(close_model_df["model_name"].unique())})

#     sns.lineplot(
#         data=close_model_df,
#         ax=ax,
#         x=x_name,
#         y=col_name,
#         color="darkred",
#         marker="o",
#         markersize=10,
#         fillstyle="full",
#         label="Close models",
#         errorbar=None,
#     )

#     sns.lineplot(
#         data=open_model_df, ax=ax, x=x_name, y=col_name, color="red", marker="o", markersize=10, fillstyle="full", label="Open models", errorbar=None
#     )

#     sns.lineplot(
#         data=tmp_df, ax=ax, x=x_name, y=col_name, color="black", marker="o", markersize=10, fillstyle="full", label="All models", errorbar=None
#     )

#     leg = ax.legend()

#     if col_name == "weighted_acc":
#         plt.title("a.", loc="left")
#         plt.ylabel("Aggregate Accuracy")
#         # for text in plt.legend().get_texts():
#         #     text.set_fontsize('xx-small')
#         # edit markerscale and fontsize to small to accommodate the graph
#         leg.remove()

#     else:
#         plt.title("b.", loc="left")
#         plt.ylabel("Instruction Following \n Effective Rate")
#         # get the label and name and sort them based on model_order_mapper
#         handles, labels = plt.gca().get_legend_handles_labels()
#         sorted_labels_handles = sorted(zip(labels, handles), key=lambda lh: model_order_mapper.get(lh[0], 0))
#         sorted_labels, sorted_handles = zip(*sorted_labels_handles)
#         ax.legend(
#             sorted_handles,
#             sorted_labels,
#             title="Model",
#             bbox_to_anchor=(1.05, 1),
#             markerscale=1.3,
#             fontsize="large",
#             loc="upper left",
#             title_fontsize="x-large",
#         )

#     ax.set_ylim([0, 1])
#     plt.xlabel("Complexity")
#     # plt.tight_layout()
#     ax.set_xticklabels(["P", "NP-Complete", "NP-Hard"])
#     plt.ylim(0, 1.05)


# def plot_v_graph():
#     # rq1
#     rq1_drawer = plot_line_chart(summary_info3)
#     ## rq1.1
#     fig, axs = plt.subplots(figsize=(12, 6))
#     gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1], wspace=0.30)

#     ax1 = plt.subplot(gs[0, 0])
#     rq1_drawer.plot_line_chart("weighted_acc", "problem_type", nphard_order_mapper, ax1)
#     ax1.set_aspect(2)

#     ax2 = plt.subplot(gs[0, 1])
#     rq1_drawer.plot_line_chart("effective_rate", "problem_type", nphard_order_mapper, ax2)
#     ax2.set_aspect(2)

#     plt.savefig("figures/weighted_accuracy_effective_rate.png", bbox_inches="tight")

# import itertools

# from scipy.stats import wilcoxon

# colors = {"P": "lightblue", "NP-Complete": "steelblue", "NP-Hard": "navy"}
# complexities = ["P", "NP-Complete", "NP-Hard"]


# def wilcoxon_test(mdf, complexity_pairs):
#     # Perform Wilcoxon test for each pair
#     p_values = []
#     for pair in complexity_pairs:
#         complexity1 = pair[0]
#         complexity2 = pair[1]
#         x = mdf[mdf["complexity"] == complexity1]["Average accuracy"]
#         y = mdf[mdf["complexity"] == complexity2]["Average accuracy"]
#         try:
#             stat, p_value = wilcoxon(x, y)
#         except:
#             p_value = 1.0
#         p_values.append(p_value)
#     return p_values


# def create_figure(model_df, model, ax):
#     # Create a figure for the model
#     mdf = model_df[model_df["model"] == model]
#     mdf = mdf.sort_values(by=["complexity"], key=lambda x: x.map({"P": 0, "NP-Complete": 1, "NP-Hard": 2}))
#     mdf = mdf.reset_index()
#     mdf = mdf.explode("Average accuracy")
#     complexity_pairs = list(itertools.combinations(complexities, 2))

#     p_values = wilcoxon_test(mdf, complexity_pairs)
#     print(p_values)
#     sns.boxplot(x="complexity", y="Average accuracy", data=mdf, gap=0.4, palette=colors, ax=ax)

#     offset = 0
#     for i, pair in enumerate(complexity_pairs):
#         complexity1 = pair[0]
#         complexity2 = pair[1]
#         x1 = mdf[mdf["complexity"] == complexity1]["complexity"].index[0]
#         x2 = mdf[mdf["complexity"] == complexity2]["complexity"].index[0]
#         y = mdf["Average accuracy"].max() + offset
#         h = 0.05
#         ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color="steelblue")
#         ax.text((x1 + x2) * 0.5, y + h, f"p={p_values[i]:.4f}", ha="center", va="bottom")
#         offset += 0.15

#     ax.set_title(f"Model: {model}")
#     ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # Set y-axis ticks
#     ax.set_xlabel("Complexity")
#     ax.set_ylim(0, 1.5)

# def plot_p_vals():
#     models = df["model"].unique()
#     order = [
#         "GPT 4 Turbo",
#         "Claude 2",
#         "GPT 3.5 Turbo",
#         "Claude Instant 1.2",
#         "PaLM 2",
#         "Yi-34b",
#         "Qwen-14b",
#         "Mistral-7b",
#         "Phi-2",
#         "MPT-30b",
#         "Vicuna-13b",
#         "Phi-1.5",
#     ]
#     sorted_models = sorted(models, key=lambda x: order.index(x))

#     fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))
#     labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]

#     # Iterate over the models and call create_figure() for each model
#     for i, model in enumerate(sorted_models):
#         row = i // 4  # Calculate the row index
#         col = i % 4  # Calculate the column index
#         create_figure(model_df, model, axs[row, col])

#         axs[row, col].text(0.05, 0.95, labels[i], transform=axs[row, col].transAxes, fontsize=12, fontweight="bold", va="top", ha="left")

#     plt.tight_layout()
#     plt.show()


def analysis():
    files = walk_results_folder("/nlpgpu/data/terry/ToolProj/results")  # check files are deepseek and gemma, seed 1 and 2
    df = create_big_df(files)
    rows = df.to_dict("records")
    import pdb

    pdb.set_trace()
    return rows


analysis()
