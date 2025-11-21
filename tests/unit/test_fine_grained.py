import pytest

from src.exps_performance.runners import Arm1, Arm2, Arm3, Arm4
from tests.conftest import EXAMPLES, check


@pytest.mark.parametrize("data_name", ["add", "sub", "mul", "lcs", "rod", "knap", "ilp_assign", "ilp_prod", "ilp_partition"])
def test_fg(llm, data_name, fg_subset, default_args):
    data = fg_subset([data_name])
    client = llm
    arm2 = Arm2(data[:EXAMPLES], default_args, client)
    arm2.run()
    check(arm2, data, "code")

    problems_w_code = arm2.set_code()
    blanks = 0
    for p in problems_w_code:
        if p.code == "":
            blanks += 1
    assert blanks <= EXAMPLES - 1, "too many no code generations"

    arm3 = Arm3(problems_w_code)
    arm3.run()
    assert arm3.errs <= EXAMPLES - 1, "too many errors"

    arm4 = Arm4(problems_w_code, default_args, client)
    arm4.run()
    check(arm4, data, "sim")

    arm1 = Arm1(data[:EXAMPLES], default_args, client)
    arm1.run()
    check(arm1, data, "nl")
