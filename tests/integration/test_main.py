import json
import re
from pathlib import Path
from typing import Any

import pytest

import src.exps_performance.arms as arms
from src.exps_performance.logger import create_big_df, walk_results_folder
from src.exps_performance.main import Args, run


@pytest.mark.slow
# on add questions of 2 digits, we should have > 80% on all areas. set this result here.
def test_e2e(tmp_path_factory: Any, default_args: Any) -> None:
    base = tmp_path_factory.mktemp("base")
    default_args.root = base
    run(default_args)
    files = walk_results_folder(str(base))
    df = create_big_df(files)
    assert df[["nl_correct"]].sum() >= int(0.8 * default_args.n), "nl surpringly wrong"
    assert df[["sim_correct"]].sum() >= int(0.8 * default_args.n), "sim surpringly wrong"
    assert df[["code_correct"]].sum() >= int(0.8 * default_args.n), "code surpringly wrong"
    assert df[["controlsim_correct"]].sum() >= int(0.1 * default_args.n), "controlsim surpringly wrong"


def test_e2e_rlm_only_mode(tmp_path_factory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Result:
        def __init__(self, response: str, err: str = "ok") -> None:
            self.response = response
            self.err = err

    class _DummyExecutor:
        def __init__(self, args: Any) -> None:
            self.args = args

        def run(self, prompt: str) -> _Result:
            match = re.search(r"Compute:\s*(\d+)\s*([+\-*])\s*(\d+)", prompt)
            assert match is not None, f"unexpected prompt format: {prompt}"
            left, op, right = match.groups()
            a = int(left)
            b = int(right)
            answer = a + b if op == "+" else (a - b if op == "-" else a * b)
            simulation = f"Computed {a} {op} {b} = {answer}."
            if '"code"' in prompt or "The code block that specifies a function 'solution()'" in prompt:
                response = json.dumps(
                    {
                        "Answer": str(answer),
                        "code": f"```python\ndef solution() -> int:\n    return {answer}\n```",
                        "simulation": simulation,
                    }
                )
            else:
                response = json.dumps({"Answer": str(answer), "simulation": simulation})
            return _Result(response)

    monkeypatch.setattr(arms, "RecursiveLMExecutor", _DummyExecutor)

    base = tmp_path_factory.mktemp("rlm_only")
    args = Args(
        root=str(base),
        n=2,
        digits_list=[2],
        kinds=["add"],
        seed=0,
        backend="dummy",
        model="openai/gpt-4o-mini",
        batch_size=1,
        checkpoint_every=1,
        only_rlm=True,
        rlm_nl=True,
        rlm_code=True,
    )

    run(args)

    files = walk_results_folder(str(base))
    assert len(files) == 1
    path = Path(files[0])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    for row in rows:
        assert row["rlmnl_answer"] != ""
        assert row["rlmcode_answer"] != ""
        assert "nl_question" not in row
        assert "sim_question" not in row
        assert "code_question" not in row
        assert "controlsim_question" not in row


def test_e2e_skips_code_and_control_when_disabled(tmp_path_factory: Any) -> None:
    base = tmp_path_factory.mktemp("no_exec")
    args = Args(
        root=str(base),
        n=2,
        digits_list=[2],
        kinds=["add"],
        seed=0,
        backend="dummy",
        model="openai/gpt-4o-mini",
        batch_size=2,
        checkpoint_every=2,
        exec_code=False,
        controlled_sim=False,
    )

    run(args)

    files = walk_results_folder(str(base))
    assert len(files) == 1
    path = Path(files[0])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    for row in rows:
        assert row["sim_question"] != ""
        assert row["nl_question"] != ""
        assert row["code_question"] == ""
        assert row["controlsim_question"] == ""
