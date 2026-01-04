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
    def __init__(self, answer_expr: str = "solution()", timeout_seconds: int = 5, max_attempts: int = 2):
        self.answer_expr = answer_expr
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    @staticmethod
    def _child_runner(code: str, answer_expr: str, conn: Any, timeout_seconds: int, attempt_idx: int) -> None:
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
        safe_timeout = max(1, int(timeout_seconds)) if timeout_seconds else 1
        signal.alarm(safe_timeout)

        runtime = GenericRuntime()
        runtime.inject({"ATTEMPT": attempt_idx})
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
        Execute code in an isolated subprocess with a hard timeout, retrying on
        timeout or errors up to max_attempts. Returns (result, error_message_or_ok).
        """
        last_out: Any = ""
        last_err: str = "unknown error"
        join_timeout = self.timeout_seconds

        for attempt_idx in range(self.max_attempts):
            parent_conn, child_conn = mp.Pipe(duplex=False)
            proc = mp.Process(
                target=self._child_runner,
                args=(code, self.answer_expr, child_conn, self.timeout_seconds, attempt_idx),
            )
            proc.start()
            child_conn.close()
            proc.join(join_timeout)

            if proc.is_alive():
                proc.terminate()
                proc.join()
                last_out, last_err = "", "timeout"
                parent_conn.close()
            elif parent_conn.poll():
                try:
                    out, err = parent_conn.recv()
                    last_out, last_err = out, err
                    parent_conn.close()
                    if err == "ok":
                        return out, err
                except EOFError as eof_err:
                    last_out, last_err = "", f"EOFError: {eof_err}"
                    parent_conn.close()
                except Exception as recv_err:
                    last_out, last_err = "", f"recv error: {recv_err}"
                    parent_conn.close()
            else:
                last_out, last_err = "", "unknown error"
                parent_conn.close()

        return last_out, last_err
