from src.exps_performance.runners import Arm1, Arm2, Arm3, Arm4
from tests.conftest import check


def test_arm2(llm, npdata, default_args):
    num_examples = 5
    data = npdata
    client = llm
    arm2 = Arm2(data[:num_examples], default_args, client)
    arm2.run()
    check(arm2, data, "code")


def test_arm3(llm, npdata, default_args):
    num_examples = 5
    data = npdata
    client = llm
    arm2 = Arm2(data[:num_examples], default_args, client)
    arm2.run()
    problems_w_code = arm2.set_code()
    arm3 = Arm3(problems_w_code)
    arm3.run()
    assert arm3.errs < 3, "too many errors"


def test_arm1(llm, npdata, default_args):
    num_examples = 5
    data = npdata
    client = llm
    arm1 = Arm1(data[:num_examples], default_args, client)
    arm1.run()
    check(arm1, data, "nl")


def test_arm4(llm, npdata, default_args):
    num_examples = 5
    data = npdata
    client = llm
    arm2 = Arm2(data[:num_examples], default_args, client)
    arm2.run()
    problems_w_code = arm2.set_code()
    arm4 = Arm4(problems_w_code, default_args, client)
    arm4.run()
    check(arm4, data, "sim")
