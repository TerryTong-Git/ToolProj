from src.exps_performance.dataset import make_dataset
from src.exps_performance.runners import Arm1, Arm2, Arm3, Arm4
from tests.conftest import EXAMPLES, check


# sim
def test_arm2(llm, default_args):
    data = make_dataset(["spp"])
    client = llm
    data_subset = data[:EXAMPLES]
    arm2 = Arm2(data_subset, default_args, client)
    accuracy = arm2.run()
    check(arm2, data, "code")
    data_subset = arm2.problems
    correct = 0
    for d in data_subset:
        assert d.record.sim_answer != "", "sim answer is not string"
        assert d.record.sim_question != "", "sim question is not string"
        assert d.record.question != "", "sim question is not string"
        assert d.record.answer != "", "sim question is not string"
        correct += int(d.record.sim_correct)
    assert accuracy == correct / EXAMPLES, "accuracy record keeping is wrong"


def test_arm3(llm, default_args):
    data = make_dataset(["spp"])
    client = llm
    data_subset = data[:EXAMPLES]
    arm2 = Arm2(data_subset, default_args, client)
    arm2.run()
    problems_w_code = arm2.set_code()
    arm3 = Arm3(problems_w_code)
    accuracy = arm3.run()
    assert arm3.errs < 3, "too many errors"
    correct = 0
    for d in data_subset:
        assert d.record.code_answer != "", "code answer is not string"
        assert d.record.code_question != "", "code question is not string"
        correct += int(d.record.code_correct)
    assert accuracy == correct / EXAMPLES, "accuracy record keeping is wrong"


def test_arm1(llm, default_args):
    data = make_dataset(["spp"])
    client = llm
    data_subset = data[:EXAMPLES]
    arm1 = Arm1(data_subset, default_args, client)
    accuracy = arm1.run()
    check(arm1, data, "nl")
    correct = 0
    for d in data_subset:
        assert d.record.nl_answer != "", "nl answer is not string"
        assert d.record.nl_question != "", "nl question is not string"
        correct += int(d.record.nl_correct)
    assert accuracy == correct / EXAMPLES, "accuracy record keeping is wrong"


def test_arm4(llm, default_args):
    data = make_dataset(["spp"])
    client = llm
    data_subset = data[:EXAMPLES]
    arm2 = Arm2(data_subset, default_args, client)
    arm2.run()
    problems_w_code = arm2.set_code()
    arm4 = Arm4(problems_w_code, default_args, client)
    accuracy = arm4.run()
    check(arm4, data, "sim")
    correct = 0
    for d in data_subset:
        assert d.record.controlsim_answer != "", "control sim answer is not string"
        assert d.record.controlsim_question != "", "control sim question is not string"
        correct += int(d.record.controlsim_correct)
    assert accuracy == correct / EXAMPLES, "accuracy record keeping is wrong"
