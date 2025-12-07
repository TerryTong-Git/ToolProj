from src.exps_performance.dataset import NPHARD


def test_dataloading():
    data = NPHARD().load()
    assert data is not None, "no data"
