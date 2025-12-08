import copy
import logging
import multiprocessing as mp
import resource
import signal
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GenericRuntime:
    GLOBAL_DICT: Dict = {}
    LOCAL_DICT: Dict = {}
    HEADERS: List = []

    def __init__(self) -> None:
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
    def answer(self) -> Any:
        return self._global_vars["answer"]


class ProgramChatInterface:
    def __init__(self, answer_expr: str = "solution()", timeout_seconds: int = 2):
        self.answer_expr = answer_expr
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _child_runner(code: str, answer_expr: str, conn: Any) -> None:
        def _timeout(_signum: int, _frame: Any) -> None:
            raise TimeoutError("code execution timed out")

        # Set CPU and wall limits
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        except Exception:
            pass
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(5)

        runtime = GenericRuntime()
        err = "ok"
        out: Any = ""
        try:
            runtime.exec_code(code)
            out = runtime.eval_code(answer_expr)
        except Exception as e:  # pragma: no cover
            err = str(e)
        finally:
            try:
                conn.send((out, err))
            except Exception:
                pass
            conn.close()
            signal.alarm(0)

    def run(self, code: str) -> Tuple[str, str]:
        """
        Execute code in an isolated subprocess with a hard timeout.
        Returns (result, error_message_or_ok).
        """
        parent_conn, child_conn = mp.Pipe(duplex=False)
        proc = mp.Process(target=self._child_runner, args=(code, self.answer_expr, child_conn))
        proc.start()
        proc.join(self.timeout_seconds)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return "", "timeout"
        if parent_conn.poll():
            out, err = parent_conn.recv()
            return out, err
        return "", "unknown error"
