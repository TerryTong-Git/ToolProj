from src.exps_performance.main import parse_args


def test_parse():
    args = parse_args()
    assert args is not None, "no args"
