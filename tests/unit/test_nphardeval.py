import ast

from src.exps_performance.llm import run_batch
from src.exps_performance.problems.nphard.spp import SPP


# only count instances that were parsed?
def test_spp_llm(instantiate_llm, default_args):
    spp = SPP("code")
    data = spp.load_data("/nlpgpu/data/terry/ToolProj/src/exps_performance/Data_V2/SPP/")

    # check the first 3 examples and see if they work.
    examples = [spp.format_one(d) for d in data[:3]]
    messages = [[{"role": "user", "content": example}] for example in examples]
    client = instantiate_llm
    answers = run_batch(messages, default_args, client)
    parsed_answer = [spp.parse_output(answer) for answer in answers]

    # test code
    code_to_run = []
    for q, p in zip(data[:3], parsed_answer):
        correct, reason = spp.decision_check(q, p)
        assert not correct, "should be wrong for all three for deepseekcoder if deterministic"
        code = p.imports + "\n" + p.code + "\n" + p.print_statement
        code_to_run.append(code)

    # try except runtime error. Then log the runtime error.
    code_ran = [exec(code) for code in code_to_run]  # extract the local variable from prompt.

    for q, p in zip(data[:3], code_ran):
        json_dict = ast.literal_eval(p)  # could be error if LLM generated wrong code
        correct, reason = spp.decision_check(q, json_dict)

    # test sim
    examples = [spp.format_one(d) for d in code_to_run]
    messages = [[{"role": "user", "content": example}] for example in examples]
    answers = run_batch(messages, default_args, client)
    parsed_answer = [spp.parse_output(answer) for answer in answers]

    for q, p in zip(data[:3], parsed_answer):
        correct, reason = spp.decision_check(q, p)
        import pdb

        pdb.set_trace()
        assert not correct, "should be wrong for all three for deepseekcoder if deterministic"


# def test_spp_llm_NL(instantiate_llm, default_args):
#     spp = SPP("nl")
#     data = spp.load_data("/nlpgpu/data/terry/ToolProj/src/exps_performance/Data_V2/SPP/")

#     # check the first 3 examples and see if they work.
#     examples = [spp.format_one(d) for d in data[:3]]
#     messages = [[{"role": "user", "content": example}] for example in examples]
#     client = instantiate_llm
#     answers = run_batch(messages, default_args, client)
#     parsed_answer = [spp.parse_output(answer) for answer in answers]

#     # test code
#     for q, p in zip(data[:3], parsed_answer):
#         correct, reason = spp.decision_check(q, p)
#         assert correct == False, "should be wrong for all three for deepseekcoder if deterministic"

# test spp llm code executable

# test spp llm code sim runnable

# assert result is seeded and reproducible and get the same thing everytime from local model / deterministic
