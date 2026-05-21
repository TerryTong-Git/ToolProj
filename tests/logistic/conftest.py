import importlib.util

collect_ignore_glob = ["test_*.py"] if importlib.util.find_spec("src.exps_logistic") is None else []
