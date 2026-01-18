"""Prompt templates for information additivity experiment.

Conditions:
1. x: Question only
2. x || z_nl: Question + NL reasoning trace
3. x || z_code: Question + code (not translated)
4. x || z_nl || z_code: Question + NL reasoning + code
5. mismatch: Question + mismatched NL reasoning + code (control)
"""

# Base instruction for all conditions
BASE_INSTRUCTION = """You are given an algorithmic problem. Your task is to determine the final answer.

## Problem
{question}

## Expected Answer Format
Respond with ONLY the final numerical answer (integer or float). No explanation needed.
Example: 42
"""

# Condition 1: Question only
CONDITION_X = """You are given an algorithmic problem. Your task is to determine the final answer.

## Problem
{question}

## Expected Answer Format
Respond with ONLY the final numerical answer (integer or float). No explanation needed.
Example: 42

Answer:"""

# Condition 2: Question + NL reasoning
CONDITION_X_NL = """You are given an algorithmic problem along with a natural language reasoning trace. Use the reasoning to determine the final answer.

## Problem
{question}

## Reasoning Trace (Natural Language)
{nl_reasoning}

## Expected Answer Format
Respond with ONLY the final numerical answer (integer or float). No explanation needed.
Example: 42

Answer:"""

# Condition 3: Question + Code (not translated)
CONDITION_X_CODE = """You are given an algorithmic problem along with code that solves it. Use the code to determine the final answer.

## Problem
{question}

## Solution Code
{code}

## Expected Answer Format
Respond with ONLY the final numerical answer (integer or float). No explanation needed.
Example: 42

Answer:"""

# Condition 4: Question + NL reasoning + Code
CONDITION_X_NL_CODE = """You are given an algorithmic problem along with both a natural language reasoning trace and code that solves it. Use both to determine the final answer.

## Problem
{question}

## Reasoning Trace (Natural Language)
{nl_reasoning}

## Solution Code
{code}

## Expected Answer Format
Respond with ONLY the final numerical answer (integer or float). No explanation needed.
Example: 42

Answer:"""

# Condition 5: Mismatch control (NL from different problem)
CONDITION_MISMATCH = """You are given an algorithmic problem along with both a natural language reasoning trace and code that solves it. Use both to determine the final answer.

## Problem
{question}

## Reasoning Trace (Natural Language)
{mismatched_nl_reasoning}

## Solution Code
{code}

## Expected Answer Format
Respond with ONLY the final numerical answer (integer or float). No explanation needed.
Example: 42

Answer:"""

CONDITIONS = {
    "x": CONDITION_X,
    "x_nl": CONDITION_X_NL,
    "x_code": CONDITION_X_CODE,
    "x_nl_code": CONDITION_X_NL_CODE,
    "mismatch": CONDITION_MISMATCH,
}
