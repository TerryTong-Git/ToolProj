from typing import Any

import pytest

from src.exps_performance.logger import create_big_df, walk_results_folder
from src.exps_performance.main import run


@pytest.mark.slow
# on add questions of 2 digits, we should have > 80% on all areas. set this result here.
def test_e2e(tmp_path_factory: Any, default_args: Any) -> None:
    base = tmp_path_factory.mktemp("base")
    default_args.root = base
    run(default_args)
    files = walk_results_folder(str(base))
    df = create_big_df(files)
    assert df[["nl_correct"]].sum() >= int(0.8 * default_args.n), "nl surpringly wrong"
    assert df[["sim_correct"]].sum() >= int(0.8 * default_args.n), "sim surpringly wrong"
    assert df[["code_correct"]].sum() >= int(0.8 * default_args.n), "code surpringly wrong"
    assert df[["controlsim_correct"]].sum() >= int(0.1 * default_args.n), "controlsim surpringly wrong"
