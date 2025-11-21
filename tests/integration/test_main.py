from src.exps_performance.logger import read_from_csv
from src.exps_performance.main import run


def test_e2e(tmp_path_factory, default_args):
    base = tmp_path_factory.mktemp("base")
    default_args.root = base
    results_path = run(default_args)
    read_from_csv(results_path)
    import pdb

    pdb.set_trace()


# run start to end fast
