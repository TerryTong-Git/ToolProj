from src.exps_performance.logger import Record
from src.exps_performance.metrics import accuracy


def test_accuracy(mock_records: list[Record]) -> None:
    df = accuracy(mock_records)
    assert list(df.columns) == ["nl_correct", "code_correct", "sim_correct", "controlsim_correct"]
    for d in df.values[0]:  # can just do 0 because only one kind
        assert d == 1.0, "mock data gone wrong"
