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
    # minimal, explicit pseudo-code first to stabilize seeds
    return (
        f"{SYSTEM}\n"
        f"Task (first write a short Python-like plan, then compute):\n"
        f"a = {task.a}\n"
        f"b = {task.b}\n"
        f"operation = \"{task.theta}\"\n"
        f"# compute r deterministically using operation\n"
        f"Then show the computation lines explicitly, and finally put "
        f"\"{STOP_STR}\" on its own line followed by the final number."
    )
