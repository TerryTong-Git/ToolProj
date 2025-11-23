import copy
import signal
from typing import Any, Dict, List, Tuple


class GenericRuntime:
    GLOBAL_DICT: Dict = {}
    LOCAL_DICT: Dict = {}
    HEADERS: List = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def answer(self):
        return self._global_vars["answer"]


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class ProgramChatInterface:
    def __init__(self, answer_expr: str = "solution()"):
        self.answer_expr = answer_expr
        self.runtime = GenericRuntime()

    def run(self, code: str) -> Tuple[str, str]:
        err = "ok"
        exec_result = ""
        with timeout(1):
            try:
                self.runtime.exec_code(code)
                exec_result = self.runtime.eval_code(self.answer_expr)
            except Exception as e:
                err = str(e)
        return exec_result, err
