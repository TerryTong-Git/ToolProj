import os
import re
from types import SimpleNamespace

from tbparse import SummaryReader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.exps_performance.logger import (
    create_big_df,
    create_dir,
    init_tensorboard,
    read_from_csv,
    walk_results_folder,
    write_text_to_tensorboard,
    write_to_csv,
)

"""
File Naming convention? 
"""


def search(res):
    pat = r"```\n(.*)\n```"
    return re.search(pat, res).group(1)


def test_log_text(tmp_path_factory, default_args, mock_records):
    base = tmp_path_factory.mktemp("base")
    args = default_args
    exp_dir = create_dir(args, base)
    writer = init_tensorboard(args, exp_dir)
    write_text_to_tensorboard(mock_records, writer, args)
    events = EventAccumulator(exp_dir)
    events.Reload()
    reader = SummaryReader(exp_dir)
    mock_data = ["abc", "1", "True"]
    for data in reader.text["value"]:
        result = search(data)
        assert result in mock_data, "wrong mock data"


def test_csv_log(tmp_path_factory, default_args, mock_records):
    base = tmp_path_factory.mktemp("base")
    args = default_args
    exp_dir = create_dir(args, base)
    csv_path = os.path.join(exp_dir, "res.csv")
    write_to_csv(csv_path, mock_records)
    records = read_from_csv(csv_path)
    assert records == mock_records, "serialization write and read not the same"


def test_aggregate_results(tmp_path_factory, default_args, mock_records, mock_records_1):
    base = tmp_path_factory.mktemp("base")
    logdirs = []
    arg_list = [default_args, SimpleNamespace(model="deepseek/deepseek", seed=2)]
    data = [mock_records, mock_records_1]
    for args, rec in zip(arg_list, data):
        exp_dir = create_dir(args, base)
        csv_path = os.path.join(exp_dir, "res.csv")
        write_to_csv(csv_path, rec)
        logdirs.append(exp_dir)
    files = walk_results_folder(base)  # check files are deepseek and gemma, seed 1 and 2
    for f in files:
        assert "deepseek" in f or "gemma" in f, "wrong models"
        assert "seed1" in f or "seed2" in f, "wrong seed"
    df = create_big_df(files)
    rows = df.to_dict("records")
    # mock_data = ["abc", "1", "True"]
    # mock_data1 = ['efg', "2", "False"]
    for r in rows:
        assert r["code_question"] in ["abc", "efg"], "not loaded correctly"
