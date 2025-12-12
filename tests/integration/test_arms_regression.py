import pytest

from src.exps_performance.arms import Arm1, Arm2, Arm3, Arm4
from src.exps_performance.dataset import make_dataset
from tests.conftest import check


@pytest.mark.slow
def test_nphard_regression_real_run(default_args: object, llm: object) -> None:
    # Keep runtime reasonable but exercise a real end-to-end flow.
    default_args.n = min(getattr(default_args, "n", 10), 3)
    data = list(make_dataset(["spp"], n=default_args.n))
    data_subset = list(data[: default_args.n])
    client = llm

    arm2 = Arm2(data_subset, default_args, client)
    sim_acc, data_subset = arm2.run()
    assert data_subset == arm2.edited_problems
    assert len(arm2.parsed_answer) == len(data_subset)
    check(arm2, data, "code")
    sim_correct = sum(int(d.record.sim_correct) for d in data_subset)
    assert sim_acc == pytest.approx(sim_correct / len(data_subset))

    blanks = sum(1 for p in data_subset if p.code == "")
    assert blanks < len(data_subset), "too many empty code generations"

    arm3 = Arm3(data_subset, default_args, client)
    code_acc, data_subset = arm3.run()
    assert data_subset == arm3.edited_problems
    assert len(arm3.parsed_answer) == len(data_subset)
    assert arm3.parse_fail < len(data_subset)
    code_correct = sum(int(d.record.code_correct) for d in data_subset)
    assert code_acc == pytest.approx(code_correct / len(data_subset))

    arm4 = Arm4(data_subset, default_args, client)
    control_acc, data_subset = arm4.run()
    check(arm4, data, "sim")
    control_correct = sum(int(d.record.controlsim_correct) for d in data_subset)
    assert control_acc == pytest.approx(control_correct / len(data_subset))

    arm1 = Arm1(data_subset, default_args, client)
    nl_acc, data_subset = arm1.run()
    check(arm1, data, "nl")
    nl_correct = sum(int(d.record.nl_correct) for d in data_subset)
    assert nl_acc == pytest.approx(nl_correct / len(data_subset))
