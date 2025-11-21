import pytest

from src.exps_performance.dataset import make_dataset
from src.exps_performance.runners import Arm1, Arm2, Arm3, Arm4
from tests.conftest import EXAMPLES, check


@pytest.mark.parametrize(
    "data_name",
    [
        "spp",
        "tsp",
        "tsp_d",
        "msp",
        "ksp",
        "gcp",
        "gcp_d",
        "bsp",
        "edp",
        "clrs",
        "gsm8k",
        "add",
        "sub",
        "mul",
        "lcs",
        "rod",
        "knap",
        "ilp_assign",
        "ilp_prod",
        "ilp_partition",
    ],
)
def test_nphard(llm, data_name, default_args):
    # should also test the seed
    # should also test the exp_id logged correctly

    data = make_dataset([data_name])
    client = llm
    data_subset = data[:EXAMPLES]
    arm2 = Arm2(data_subset, default_args, client)
    accuracy = arm2.run()
    check(arm2, data, "code")
    data_subset = arm2.edited_problems
    correct = 0
    for d in data_subset:
        # parse_error = d.record.sim_parse_err
        # ans_ok = d.record.sim_reasoning != ""
        # assert parse_error != ans_ok, "cannot have ok answer and parse error"
        assert d.record.seed != -1, "did not seed the problem"
        assert d.record.sim_answer != "", "sim answer is not string"
        assert d.record.sim_question != "", "sim question is not string"
        correct += int(d.record.sim_correct)
    assert accuracy == correct / EXAMPLES, "accuracy record keeping is wrong"

    problems_w_code = arm2.set_code()
    blanks = 0
    for p in problems_w_code:
        if p.code == "":
            blanks += 1
    assert blanks <= EXAMPLES - 1, "too many no code generations"

    arm3 = Arm3(problems_w_code)
    accuracy = arm3.run()
    assert arm3.errs <= EXAMPLES - 1, "too many errors"

    data_subset = arm3.edited_problems
    correct = 0
    for d in data_subset:
        # parse_error = d.record.code_parse_err
        # code_ok = d.record.code_answer != "-1"
        # gen_error = d.record.code_gen_err
        # assert parse_error != code_ok, "cannot have ok code and parse error"
        # assert parse_error != gen_error, "cannot have gen_err and parse error, only one"
        # assert gen_error != code_ok, "cannot have gen error and ok code" -> this is possible when code is not syntax error but type error
        assert isinstance(d.record.code_answer, str), "code answer is not string"
        assert isinstance(d.record.code_question, str), "code question is not string"
        correct += int(d.record.code_correct)
    assert accuracy == correct / EXAMPLES, "accuracy record keeping is wrong"

    arm4 = Arm4(problems_w_code, default_args, client)
    accuracy = arm4.run()
    check(arm4, data, "sim")

    data_subset = arm4.edited_problems
    correct = 0
    for d in data_subset:
        parse_error = d.record.controlsim_parse_err
        ans_ok = d.record.controlsim_reasoning != ""
        assert parse_error != ans_ok, "cannot have ok answer and parse error"
        assert d.record.controlsim_answer != "", "control sim answer is not string"
        assert d.record.controlsim_question != "", "control sim question is not string"
        correct += int(d.record.controlsim_correct)
    assert accuracy == correct / EXAMPLES, "accuracy record keeping is wrong"

    arm1 = Arm1(data_subset, default_args, client)
    accuracy = arm1.run()
    check(arm1, data, "nl")
    data_subset = arm1.edited_problems
    correct = 0
    for d in data_subset:
        parse_error = d.record.nl_parse_err
        ans_ok = d.record.nl_answer != ""
        assert parse_error != ans_ok, "cannot have ok answer and parse error"
        assert d.record.nl_answer != "", "nl answer is not string"
        assert d.record.nl_question != "", "nl question is not string"
        correct += int(d.record.nl_correct)
    assert accuracy == correct / EXAMPLES, "accuracy record keeping is wrong"
