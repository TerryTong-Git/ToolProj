import pytest

from src.exps_performance.runners import Arm1, Arm2, Arm3, Arm4


def check(arm, data, types):
    parsed_answer = arm.parsed_answer
    assert arm.parse_fail < 5, "parse failed too much"
    pUtil = data[0].util_pointer(types)
    classtype = pUtil.PROB_TYPES[types]
    empties = 0
    for parsed in parsed_answer:
        assert isinstance(parsed, classtype), "no output, all wrong output types"
        if parsed == classtype():
            empties += 1
    assert empties < 2, "too many no parse"


@pytest.mark.parametrize("data_name", ["spp"])
def test_fine_grained(instantiate_llm, data_name, subset_data, default_args):
    num_examples = 5
    data = subset_data([data_name])
    client = instantiate_llm
    arm2 = Arm2(data[:num_examples], default_args, client)
    arm2.run()
    check(arm2, data, "code")

    problems_w_code = arm2.set_code()
    blanks = 0
    for p in problems_w_code:
        if p.code == "":
            blanks += 1
    assert blanks < 3, "too many no code generations"

    arm3 = Arm3(problems_w_code)
    arm3.run()
    assert arm3.errs < 3, "too many errors"

    arm4 = Arm4(problems_w_code, default_args, client)
    arm4.run()
    check(arm4, data, "sim")

    arm1 = Arm1(data[:num_examples], default_args, client)
    arm1.run()
    check(arm1, data, "nl")
