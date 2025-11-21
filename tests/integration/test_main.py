from src.exps_performance.main import parse_args


def test_e2e():
    args = parse_args()
    assert args is not None, "no args"


# run start to end fast
