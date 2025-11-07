# ------------------------------- Prompts ------------------------------------

JSON_SCHEMA = "Return only JSON with keys 'rationale' and 'answer'. " "'answer' must be a single integer. No extra keys, no text outside JSON."

# IMPORTANT: double braces {{ }} for format literals
NL_PROMPT = """
You are tasked with solving an algorithmic problem by reasoning through it step by step using a chain-of-thought approach expressed 
in clear, natural language. Begin by thoroughly analyzing the problem, breaking it down into manageable parts, and explaining your
thought process in detail. The problem is given after <|Problem|>. You should fill in <|Response|>. You are never allowed to use code. 
After fully reasoning through the problem in natural language, output a JSON dictionary containing two keys:

- "rationale": a comprehensive explanation summarizing your reasoning and approach to the problem.
- "answer": give the final requested answer as an integer.

Ensure your explanation is clear, logically structured, and leads naturally to the final answer provided in the JSON output. 

Examples:

================ ONE-SHOT EXAMPLE ================

<|Problem|>
Compute the GCD of 48 and 18.

<|Response|>
I will use the Euclidean algorithm, which repeatedly replaces a and b with b and a modulo b until the remainder is zero. 
The last nonzero remainder is the greatest common divisor.

Start with forty eight and eighteen. Compute forty eight modulo eighteen which is twelve, so update to eighteen and twelve. 
Next, eighteen modulo twelve is six, so update to twelve and six. Then, twelve modulo six is zero, so update to six and zero.
When the second number becomes zero, the greatest common divisor is the first number, which is six.

rationale is I will use the Euclidean algorithm, repeatedly replacing a and b with b and a modulo b until the remainder is zero. 
Start with forty eight and eighteen. Forty eight modulo eighteen is twelve, so update to eighteen and twelve. Then eighteen modulo 
twelve is six, so update to twelve and six. Then twelve modulo six is zero, so update to six and zero. When the second number becomes zero,
the greatest common divisor is six.

answer is six

{{
\"rationale\": \"I will use the Euclidean algorithm, which repeatedly replaces a and b with b and a modulo b until the remainder is zero. 
The last nonzero remainder is the greatest common divisor.

Start with forty eight and eighteen. Compute forty eight modulo eighteen which is twelve, so update to eighteen and twelve. 
Next, eighteen modulo twelve is six, so update to twelve and six. Then, twelve modulo six is zero, so update to six and zero.
When the second number becomes zero, the greatest common divisor is the first number, which is six.

rationale is I will use the Euclidean algorithm, repeatedly replacing a and b with b and a modulo b until the remainder is zero. 
Start with forty eight and eighteen. Forty eight modulo eighteen is twelve, so update to eighteen and twelve. Then eighteen modulo 
twelve is six, so update to twelve and six. Then twelve modulo six is zero, so update to six and zero. When the second number becomes zero,
the greatest common divisor is six.

answer is six\",
\"answer\": 6
}}

================= YOUR TASK =================
<|Problem|>
{problem}

<|Response|>
"""

CODE_PROMPT = """
You are an expert algorithm problem solver who reasons primarily in Python, but you may mix brief natural language and math. Think step by step.

What to produce in <|Response|> (in this order):
1) A SHORT plan (2–5 bullet points) explaining your approach.
2) A single Python code block that:
   - Defines all variables correctly, indents correctly, and computes the answer
   - Ends by printing the final integer result on the last line via print(...).
   - Uses only deterministic logic (no external I/O or randomness).
   - You MAY use: math, numpy, torch, pulp, scipy, pandas (but prefer pure Python if possible).
3) An **Execution Attempt** section where you mentally simulate the main steps of your program:
   - You should attempt to simulate the execution of the program in natural language. 
4) A JSON object with two keys:
   - "rationale": the complete Python code solution, inside a code block.
   - "answer": the integer result printed by your program.

================ ONE-SHOT EXAMPLE ================

<|Problem|>:
Compute the GCD of 48 and 18.

<|Response|>:
Plan:
- Use Euclid’s algorithm: repeatedly replace (a, b) with (b, a % b) until b == 0.
- Return a.
- Print the result.

```python
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a
res = gcd(48, 18)
print(res)
```

Execution Attempt:
I will use the Euclidean algorithm, which repeatedly replaces (a, b) with (b, a mod b) until the remainder is 0; 
the last nonzero remainder is the GCD.
Start with (48, 18). Compute 48 mod 18 = 12, so update to (18, 12).
Next, 18 mod 12 = 6, so update to (12, 6).
Then, 12 mod 6 = 0, so update to (6, 0).
When the second number becomes 0, the GCD is the first number, which is 6.

{{
"rationale": "```python
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a
res = gcd(48, 18)
print(res)
```",
"answer": 6
}}

================= YOUR TASK =================
<|Problem|>:
{problem}

<|Response|>:
"""

SIM_PROMPT = """ 
Simulate the code. Give your final answer in json format. For example, if the answer is "6", respond like this: {{"answer": 6
}}

Code:
{problem}
"""
