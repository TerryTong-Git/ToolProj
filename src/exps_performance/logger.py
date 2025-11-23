# define how to serialize here.
import os
import time
from functools import partial
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter

# class AggregateRecord(BaseModel):
#     model: str
#     digits: int
#     kinds: str
#     code_acc: float
#     nl_acc: float
#     sim_acc: float
#     controlled_acc: float
#     seed: int
#     n: int
#     exp_id: str


class Record(BaseModel):
    model: str = ""  # answers depend on this
    seed: int = -1  # answers depend on this
    exp_id: str = ""
    digit: int = -1
    kind: str = ""
    question: str = ""
    answer: str = ""

    nl_question: str = ""
    nl_answer: str = ""
    nl_reasoning: str = ""
    nl_correct: bool = False
    nl_parse_err: bool = False
    nl_err_msg: str = ""  # defaults to "" if not err

    code_question: str = ""
    code_answer: str = ""  # (or err message)
    code_correct: bool = False
    code_parse_err: bool = False
    code_gen_err: str = ""
    code_err_msg: str = ""

    sim_question: str = ""
    sim_reasoning: str = ""  # attempted reasoning
    sim_answer: str = ""
    sim_correct: bool = False
    sim_parse_err: bool = False
    sim_err_msg: str = ""

    controlsim_question: str = ""
    controlsim_reasoning: str = ""
    controlsim_answer: str = ""
    controlsim_correct: bool = False
    controlsim_parse_err: bool = False
    controlsim_err_msg: str = ""


# results/model_seed/exp_id/res.csv
def create_dir(args, base):  # root would be like ./results
    outdir: str = args.model.split("/")[1] + f"_seed{args.seed}"
    abs_outdir = os.path.join(base, "results", outdir)
    os.makedirs(abs_outdir, exist_ok=True)
    exp_id = time.strftime("run_%Y%m%d_%H%M%S")
    actual_logdir = os.path.join(abs_outdir, "tb", exp_id)
    return actual_logdir


def init_tensorboard(args, exp_dir) -> SummaryWriter:  # use the logdir to specify tmp for testing, later switch to results dir
    return SummaryWriter(log_dir=exp_dir)


def tb_text(
    tag: str,
    title: str,
    body: str,
    tb: SummaryWriter,
    args,
    step: int = 0,
):
    if tb is None:
        return
    body = body or ""
    n = args.tb_text_chars
    body_show = body if len(body) <= n else (body[: max(0, n - 3)] + "...")
    tb.add_text(tag, f"**{title}**\n\n```\n{body_show}\n```", global_step=step)


# tb for fine-grained, csv for coarse grained / direct loading into dataframes


def write_text_to_tensorboard(records: List[Record], tb: SummaryWriter, args):
    write_tb = partial(tb_text, tb=tb, args=args)
    for i, record in enumerate(records):  # assumes ordered according to the questions
        base_nl = f"{record.model}/nl/d{record.digit}/{record.kind}/i{i}"
        base_code = f"{record.model}/code/d{record.digit}/{record.kind}/i{i}"
        base_sim = f"{record.model}/sim/d{record.digit}/{record.kind}/i{i}"
        base_controlsim = f"{record.model}/controlsim/d{record.digit}/{record.kind}/i{i}"

        write_tb(f"{base_nl}/question", "Original Question Data", str(record.question))
        write_tb(f"{base_nl}/answer", "Gold Answer", str(record.answer))

        write_tb(f"{base_nl}/nl_question", "NL Prompt", str(record.nl_question))
        write_tb(f"{base_nl}/nl_answer", "Final Answer (NL)", str(record.nl_answer))
        write_tb(f"{base_nl}/nl_correct", "Is the answer correct?", str(record.nl_correct))
        write_tb(f"{base_nl}/nl_parse_err", "Is there an error?", str(record.nl_parse_err))
        write_tb(f"{base_nl}/nl_err_msg", "The err message if there is one", str(record.nl_err_msg))

        write_tb(f"{base_code}/code_question", "code Prompt", str(record.code_question))
        write_tb(f"{base_code}/code_answer", "Final Answer (code)", str(record.code_answer))
        write_tb(f"{base_code}/code_correct", "Is the answer correct?", str(record.code_correct))
        write_tb(f"{base_code}/code_parse_err", "Is there a parse error?", str(record.code_parse_err))
        write_tb(f"{base_code}/code_gen_err", "Is there a generation / formatting error?", str(record.code_parse_err))
        write_tb(f"{base_code}/code_err_msg", "The err message if there is one", str(record.code_err_msg))

        write_tb(f"{base_sim}/sim_question", "sim Prompt", str(record.sim_question))
        write_tb(f"{base_sim}/sim_answer", "Final Answer (sim)", str(record.sim_answer))
        write_tb(f"{base_sim}/sim_correct", "Is the answer correct?", str(record.sim_correct))
        write_tb(f"{base_sim}/sim_parse_err", "Is there an error?", str(record.sim_parse_err))
        write_tb(f"{base_sim}/sim_err_msg", "The err message if there is one", str(record.sim_err_msg))

        write_tb(f"{base_controlsim}/controlsim_question", "controlsim Prompt", str(record.controlsim_question))
        write_tb(f"{base_controlsim}/controlsim_answer", "Final Answer (controlsim)", str(record.controlsim_answer))
        write_tb(f"{base_controlsim}/controlsim_correct", "Is the answer correct?", str(record.controlsim_correct))
        write_tb(f"{base_controlsim}/controlsim_parse_err", "Is there an error?", str(record.controlsim_parse_err))
        write_tb(f"{base_controlsim}/controlsim_err_msg", "The err message if there is one", str(record.controlsim_err_msg))
    tb.flush()
    tb.close()


def write_to_csv(logdir, records: List[Record]):
    result: List[Dict] = []
    for record in records:
        result.append(record.model_dump())
    df = pd.DataFrame(result)
    df.to_csv(logdir)


def read_from_csv(logdir) -> List[Record]:
    result: List[Record] = []
    df = pd.read_csv(logdir)
    dicts = df.to_dict("records")
    for r in dicts:
        result.append(Record(**r))  # type: ignore
    return result


def walk_results_folder(csv_folder):
    csv_files = []
    for dirpath, dirnames, filenames in os.walk(csv_folder):
        for filename in filenames:
            if filename.endswith(".csv"):
                csv_files.append(os.path.join(dirpath, filename))
    return csv_files


def create_big_df(csv_files: List[Path]):
    big_df = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        big_df.append(df)
    return pd.concat(big_df, axis=0, ignore_index=True)
