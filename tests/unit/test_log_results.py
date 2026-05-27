import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence, Union

from tbparse import SummaryReader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.reasoning_benchmark.records import (
    RLM_ONLY_RESULT_FIELDS,
    CheckpointManager,
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


def search(res: str) -> str:
    pat = r"```\n(.*)\n```"
    match = re.search(pat, res)
    assert match is not None, f"Pattern not found in: {res}"
    return match.group(1)


def test_log_text(tmp_path_factory: Any, default_args: Any, mock_records: Any) -> None:
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


def test_csv_log(tmp_path_factory: Any, default_args: Any, mock_records: Any) -> None:
    base = tmp_path_factory.mktemp("base")
    args = default_args
    exp_dir = create_dir(args, base)
    csv_path = os.path.join(exp_dir, "res.csv")
    write_to_csv(csv_path, mock_records)
    records = read_from_csv(csv_path)
    assert records == mock_records, "serialization write and read not the same"


def test_aggregate_results(tmp_path_factory: Any, default_args: Any, mock_records: Any, mock_records_1: Any) -> None:
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
    typed_files: Sequence[Union[str, Path]] = files
    for f in files:
        assert "deepseek" in f or "gemma" in f, "wrong models"
        assert "seed1" in f or "seed2" in f, "wrong seed"
    df = create_big_df(typed_files)
    rows = df.to_dict("records")
    # mock_data = ["abc", "1", "True"]
    # mock_data1 = ['efg', "2", "False"]
    for r in rows:
        assert r["sim_parse_err"] in ["abc", "efg"], "not loaded correctly"


def test_checkpoint_rlm_only_schema(tmp_path_factory: Any, mock_records: Any) -> None:
    base = tmp_path_factory.mktemp("base")
    path = Path(base) / "res.jsonl"
    checkpoint = CheckpointManager(str(path), only_rlm=True)
    checkpoint.save_batch(mock_records, flush=True)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "expected serialized rows"
    for row in rows:
        assert set(row.keys()) == set(RLM_ONLY_RESULT_FIELDS)
        assert "nl_question" not in row
        assert "sim_question" not in row
