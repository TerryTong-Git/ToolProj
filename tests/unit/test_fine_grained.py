# import ast
# import pickle as pkl
# from src.exps_performance.llm import run_batch
# from src.exps_performance.problems.nphard.spp import SPP, SPPCodeReasoning

# # TODO: terminal choose based off of fast vs slow

# # def run(messages, default, client):

# # only count instances that were parsed?
# def test_fine_grained(instantiate_llm, default_args):
#     spp = SPP("code")
#     data = spp.load_data("/nlpgpu/data/terry/ToolProj/src/exps_performance/Data_V2/SPP/")

#     # check the first 3 examples and see if they work.
#     examples = [spp.format_one(d) for d in data[:3]]
#     messages = [[{"role": "user", "content": example}] for example in examples]
#     client = instantiate_llm
#     answers = run_batch(messages, default_args, client)
#     parsed_answer = [spp.parse_output(answer) for answer in answers]

#     # test code
#     code_to_run = []
#     for q, p in zip(data[:3], parsed_answer):
#         import pdb; pdb.set_trace() # exclude instruction following parse errors. Although there may be some interactions present there?
#         correct, reason = spp.decision_check(q, p) #integrate err handling here or nah, do like wilcoxon.
#         # assert not correct, "should be wrong for all three for deepseekcoder if deterministic"
#         code = p.imports + "\n" + p.code + "\n" + p.code_answer
#         code_to_run.append(code)

#     with open("/nlpgpu/data/terry/ToolProj/tests/integration/fixtures/saved_code_spp.pkl", 'wb+') as f:
#         pkl.dump(code_to_run, f)
#     # try except runtime error. Then log the runtime error.
#     def extract_locals(code):
#         local = {}
#         try:
#             exec(code, globals(), local)
#             return local['answer']
#         except Exception as e:
#             with open("./log.txt", 'w+') as f:
#                 f.write(f"error {e}")

#     code_ran = [extract_locals(code) for code in code_to_run]  # extract the local variable from prompt.

#     for q, p in zip(data[:3], code_ran):
#         if p == -1:
#             sol = SPPCodeReasoning(
#                 prefix ="",
#                 imports = "",
#                 code = "",
#                 code_answer = "",
#                 simulation = "",
#                 Path="",
#                 TotalDistance="")
#         else:
#             sol =  SPPCodeReasoning(
#                 prefix ="",
#                 imports = "",
#                 code = "",
#                 code_answer = "",
#                 simulation = "",
#                 Path=p[0],
#                 TotalDistance=p[1])
#         correct, reason = spp.decision_check(q, sol)
#         import pdb; pdb.set_trace()

#     # test sim
#     spp_sim = SPP("sim")
#     examples = [spp_sim.format_one(d) for d in code_to_run]
#     messages = [[{"role": "user", "content": example}] for example in examples]
#     answers = run_batch(messages, default_args, client)
#     parsed_answer = [spp_sim.parse_output(answer) for answer in answers]

#     for q, parsed in zip(data[:3], parsed_answer):

#         correct, reason = spp_sim.decision_check(q, parsed)
#         import pdb
#         pdb.set_trace()
#         # assert not correct, "should be wrong for all three for deepseekcoder if deterministic"


# # def test_spp_llm_NL(instantiate_llm, default_args):
# #     spp = SPP("nl")
# #     data = spp.load_data("/nlpgpu/data/terry/ToolProj/src/exps_performance/Data_V2/SPP/")

# #     # check the first 3 examples and see if they work.
# #     examples = [spp.format_one(d) for d in data[:3]]
# #     messages = [[{"role": "user", "content": example}] for example in examples]
# #     client = instantiate_llm
# #     answers = run_batch(messages, default_args, client)
# #     parsed_answer = [spp.parse_output(answer) for answer in answers]

# #     # test code
# #     for q, p in zip(data[:3], parsed_answer):
# #         correct, reason = spp.decision_check(q, p)
# #         assert correct == False, "should be wrong for all three for deepseekcoder if deterministic"

# # test spp llm code executable

# # test spp llm code sim runnable

# # assert result is seeded and reproducible and get the same thing everytime from local model / deterministic
