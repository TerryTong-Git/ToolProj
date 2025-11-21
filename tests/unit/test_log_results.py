import os
import re

from tbparse import SummaryReader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.exps_performance.logger import init_tensorboard, read_from_csv, write_text_to_tensorboard, write_to_csv

"""
File Naming convention? 
"""


def search(res):
    pat = r"```\n(.*)\n```"
    return re.search(pat, res).group(1)


def test_log_text(tmp_path_factory, default_args, mock_records):
    fn = tmp_path_factory.mktemp("results")
    args = default_args
    writer, logdir = init_tensorboard(args, fn)

    write_text_to_tensorboard(mock_records, writer, args)
    events = EventAccumulator(logdir)
    events.Reload()
    reader = SummaryReader(logdir)
    mock_data = ["abc", "1", "True"]
    for data in reader.text["value"]:
        result = search(data)
        assert result in mock_data, "wrong mock data"


def test_csv_log(tmp_path_factory, default_args, mock_records):
    fn = tmp_path_factory.mktemp("results")
    args = default_args
    writer, logdir = init_tensorboard(args, fn)
    csv_path = os.path.join(logdir, "res.csv")
    write_to_csv(csv_path, mock_records)
    records = read_from_csv(csv_path)
    assert records == mock_records, "serialization write and read not the same"
