from src.exps_performance.runners import Arm2


# load data as a fixture
def test_arm2(instantiate_llm, instantiate_data, default_args):
    num_examples = 5
    data = instantiate_data
    client = instantiate_llm
    arm2 = Arm2(data[:num_examples], default_args, client)
    parsed_answer, accuracy = arm2.run()
    assert accuracy > 0, "all wrong"
    pUtil = data[0].util_pointer("code")
    classtype = pUtil.PROB_TYPES["code"]
    for parsed in parsed_answer:
        assert isinstance(parsed, classtype), "no output, all wrong output types"
