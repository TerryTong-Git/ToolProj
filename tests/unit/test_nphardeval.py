import pickle as pkl

from src.exps_performance.core.runtime import ProgramChatInterface
from src.exps_performance.llm import run_batch
from src.exps_performance.problems.nphard.spp import SPP, SPPCodeReasoning
from src.exps_performance.utils import clean_code_llm, remove_json_backticks

# TODO: terminal choose based off of fast vs slow


# def run(messages, default, client):


# only count instances that were parsed?
def test_spp_llm(instantiate_llm, instantiate_data, default_args):
    num_examples = 5
    data = instantiate_data
    # check the first 3 examples and see if they work.
    examples = [(d.util_pointer)("code").format_one(d) for d in data[:num_examples]]
    messages = [[{"role": "user", "content": example}] for example in examples]
    client = instantiate_llm
    answers = run_batch(messages, default_args, client)

    parsed_answer = []
    for i, pairs in enumerate(zip(data[:num_examples], answers)):  # with retries for formatting
        d, answer = pairs
        parsed = (d.util_pointer)("code").parse_output(remove_json_backticks(answer))
        count = 0
        while parsed == SPPCodeReasoning() and count < 3:
            rerun = run_batch([messages[i]], default_args, client)[0]
            reparsed = (d.util_pointer)("code").parse_output(remove_json_backticks(rerun))
            with open("/nlpgpu/data/terry/ToolProj/tests/log.txt", "a+") as f:
                f.write(f"Retrying {i}")
            parsed = reparsed
            count += 1
        parsed_answer.append(parsed)
    # test code
    code_to_run = []
    for q, p in zip(data[:num_examples], parsed_answer):
        # assert parsed_answer != SPPCodeReasoning(), "rerun parse failure"
        correct, reason = (q.util_pointer)("code").decision_check(q, p)  # integrate err handling here or nah, do like wilcoxon.
        # assert not correct, "should be wrong for all three for deepseekcoder if deterministic"
        code = p.code
        cleaned_code = clean_code_llm(str(code))
        assert "```" not in cleaned_code, "parse err"
        code_to_run.append(cleaned_code)  # exclude instruction following parse errors. Although there may be some interactions present there?

    with open("/nlpgpu/data/terry/ToolProj/tests/integration/fixtures/saved_code_spp.pkl", "wb+") as f:
        pkl.dump(code_to_run, f)

    def extract_locals(code):
        itf = ProgramChatInterface(err_log="/nlpgpu/data/terry/ToolProj/tests/log.txt")
        code = itf.process_generation_to_code(code)
        return itf.run(code)  # -1 if err

    code_ran = [extract_locals(code) for code in code_to_run]  # extract the local variable from prompt.

    for q, p in zip(data[:num_examples], code_ran):
        if p == -1 or len(p) < 2:
            sol = SPPCodeReasoning(prefix="", code="", code_answer="", simulation="", Path="", TotalDistance="")
        else:
            sol = SPPCodeReasoning(prefix="", code="", code_answer="", simulation="", Path=str(p[0]), TotalDistance=str(p[1]))
        correct, reason = (q.util_pointer)("code").decision_check(q, sol)
        import pdb

        pdb.set_trace()

    # test sim
    spp_sim = SPP("sim")
    examples = [spp_sim.format_one(d) for d in code_to_run]
    messages = [[{"role": "user", "content": example}] for example in examples]
    answers = run_batch(messages, default_args, client)
    parsed_answer_sim = [spp_sim.parse_output(answer) for answer in answers]

    for q, p in zip(data[:3], parsed_answer_sim):
        correct, reason = spp_sim.decision_check(q, p)
        import pdb

        pdb.set_trace()
        # assert not correct, "should be wrong for all three for deepseekcoder if deterministic"


def test_spp_llm_NL(instantiate_llm, default_args):
    spp = SPP("nl")
    data = spp.load_data("/nlpgpu/data/terry/ToolProj/src/exps_performance/Data_V2/SPP/")

    # check the first 3 examples and see if they work.
    examples = [spp.format_one(d) for d in data[:3]]
    messages = [[{"role": "user", "content": example}] for example in examples]
    client = instantiate_llm
    answers = run_batch(messages, default_args, client)
    parsed_answer = [spp.parse_output(answer) for answer in answers]

    # test code
    for q, p in zip(data[:3], parsed_answer):
        correct, reason = spp.decision_check(q, p)
        assert not correct, "should be wrong for all three for deepseekcoder if deterministic"


# assert result is seeded and reproducible and get the same thing everytime from local model / deterministic
