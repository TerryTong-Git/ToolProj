from src.exps_performance.logger import create_big_df, walk_results_folder


def analysis():
    files = walk_results_folder("/nlpgpu/data/terry/ToolProj/src/exps_performance/results")  # check files are deepseek and gemma, seed 1 and 2
    df = create_big_df(files)
    rows = df.to_dict("records")
    return rows


analysis()
