# cot_jointprob_exp/prompts.py
SYSTEM = "You are a helpful math tutor. Always show detailed reasoning before the final answer."
STOP_STR = "Final answer:"

def nl_instruction(task) -> str:
    return (
        f"{SYSTEM}\n"
        f"Task: {task.x_nl}\n"
        f"Answer the question. Show your step-by-step reasoning, "
        f"then put \"{STOP_STR}\" on its own line followed by the final number."
    )

def code_instruction(task) -> str:
    # ask for a small code-like spec as z, then final answer
    return (
        f"{SYSTEM}\n"
        f"Task (produce a minimal Python-like plan first):\n"
        f"- Define variables a={task.a}, b={task.b}\n"
        f"- Operation: {task.theta}\n"
        f"- Compute result r\n"
        f"Then show the computation lines explicitly, and finally put "
        f"\"{STOP_STR}\" on its own line followed by the final number."
    )
