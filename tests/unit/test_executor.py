from src.exps_performance.core.executor import ProgramChatInterface


def test_executor_retries_timeout_then_succeeds():
    code = """
import time

def solution():
    if ATTEMPT == 0:
        time.sleep(2)
    return 42
"""
    itf = ProgramChatInterface(timeout_seconds=1, max_attempts=2)
    out, err = itf.run(code)
    assert err == "ok"
    assert out == 42


def test_executor_times_out_until_limit():
    code = """
import time

def solution():
    time.sleep(2)
    return 1
"""
    itf = ProgramChatInterface(timeout_seconds=1, max_attempts=3)
    out, err = itf.run(code)
    assert out == ""
    assert err == "timeout"


def test_executor_retries_on_crash_and_returns_error():
    code = """

def solution():
    raise ValueError("boom")
"""
    itf = ProgramChatInterface(timeout_seconds=1, max_attempts=2)
    out, err = itf.run(code)
    assert out == ""
    assert "boom" in err
