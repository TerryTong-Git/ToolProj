# ------------------------------- Prompts ------------------------------------

# Should have a part designated to code reasoning, versus NL reasoning in the prompt.
# Then a format part, which may differ for each problem.
# Then for each problem they specify their own format etc.
# Store the prompt with the problem actually...


JSON_SCHEMA = (
    "Return only JSON with keys 'rationale' and 'answer'. "
    + "'answer' is the final answer in valid json format, e.g. an int, dictionary etc. No extra keys, no text outside JSON."
)


# IMPORTANT: double braces {{ }} for format literals
NL_PROMPT = """
You are tasked with solving an algorithmic problem by reasoning through it step by step using a chain-of-thought approach expressed 
in clear, natural language. Begin by thoroughly analyzing the problem, breaking it down into manageable parts, and explaining your
thought process in detail. The problem is given after <|Problem|>. You should fill in <|Response|>. You are never allowed to use code. 
After fully reasoning through the problem in natural language, output a JSON dictionary containing two keys:

- "rationale": a comprehensive explanation summarizing your reasoning and approach to the problem.
- "answer":  Follow the same format specified in the question, default is an integer if not specified


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
   - Ends by printing the final result on the last line via print(...).
   - Uses only deterministic logic (no external I/O or randomness).
   - You MAY use: math, numpy, torch, pulp, scipy, pandas (but prefer pure Python if possible).
3) An **Execution Attempt** section where you mentally simulate the main steps of your program:
   - You should attempt to simulate the execution of the program in natural language. 
4) A JSON object with two keys:
   - "rationale": the complete Python code solution, inside a code block.
   - "answer": the expected result printed by your program.
   Note that the rationale printed answer and expected answer should follow the same format 
   specified in the question. Default is an integer if not specified. 

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


# ================= NPHARDEVAL PROMPTS ======================
sppPrompts = {
    "Intro": "The Shortest Path Problem (SPP) involves finding the shortest path between two nodes in a weighted graph.",
    "Initial_question": "You need to find the shortest path between node {start_node} and node {end_node} in a graph."
    + " The graph's edges and their weights are given",
    "Output_format": "where '0->......->3' is an example path, 'i' is the total distance of the path."
    """ "answer": {'Path': '0->......->3', 'TotalDistance': 'j'} """,
}

mfpPrompts = {
    "Intro": "The Maximum Flow Problem (MFP) seeks to find the maximum possible flow from a source node "
    "to a sink node in a flow network, subject to capacity constraints on the edges.",
    "Initial_question": "Determine the maximum flow from the source node {source_node} to the sink node {sink_node} in the "
    "given flow network. The capacities of the edges are provided.",
    "Output_format": """
    where 'i' is the max flow value, and the dictionary is an example of flow for each edge 
    "answer": {'MaxFlow': 'i', 'Flows': {'0->1': 4, ... ... '2->3': 1}}""",
}

bspPrompts = {
    "Intro": "The Binary Search Problem (BSP) deals with finding the position of a target value within a sorted array using"
    " a binary search algorithm, which efficiently narrows down the search range.",
    "Initial_question": "Find the position of the target value {target_value} in the sorted array. The index begins with 0. "
    "The array elements are provided.",
    "Output_format": """ "answer": {'Position': 'TARGET_POSITION'} """,
}

edpPrompts = {
    "Intro": "The Edit Distance Problem (EDP) involves finding the minimum number of operations required to transform one "
    "string into another, where each operation is either an insertion, deletion, or substitution of a single character.",
    "Initial_question": "Find the minimum number of operations required to transform the first string {string_a} "
    "into the second string {string_b}. The operations are insertion, deletion, and substitution of a single character, "
    " each requiring 1 edit operation.",
    "Output_format": """ "answer": {'Operations': 'MINIMUM_NUMBER_OF_OPERATIONS'} """,
}

# NP-complete problems
tsp_dPrompts = {
    "Intro": "The Traveling Salesman Problem (Decision Version, TSP-D) focuses on determining if a salesman can "
    "complete a route, visiting each city at least once, with the total travel distance being less than a specified value.",
    "Initial_question": "Check if it's possible for a salesman to visit each of the {total_cities} cities at"
    " least once and return to the starting city with the total distance less than {distance_limit}. "
    "The distances between each pair of cities are given.",
    "Output_format": """ "answer": {'Feasible': 'YES_OR_NO'} """,
}

gcp_dPrompts = {
    "Intro": "The Graph Coloring Problem (Decision Version, GCP-D) involves determining if it is possible"
    " to color the vertices of a graph using a given number of colors, ensuring no two adjacent vertices have the same color.",
    "Initial_question": "Find out if the vertices of a graph with {total_vertices} vertices can be "
    "colored using only {number_of_colors} colors, such that no adjacent vertices share the same color.",
    "Output_format": """ "answer": {'Feasible': 'YES_OR_NO'} """,
}

kspPrompts = {
    "Intro": "The Knapsack Problem (KSP) asks whether a subset of items, each with a given weight and value, "
    "can be chosen to fit into a knapsack of fixed capacity, maximizing the total value without exceeding the capacity.",
    "Initial_question": "Determine if a subset of items can be selected to fit into a knapsack with a capacity of "
    "{knapsack_capacity}, maximizing value without exceeding the capacity. Item weights and values are provided.",
    "Output_format": """ "answer": {'Feasible': 'YES_OR_NO', 'TotalValue': 'TOTAL_VALUE'} """,
}

# NP-hard problems
tspPrompts = {
    "Intro": "The traveling salesman problem (TSP) is a classic optimization problem that aims to find the shortest possible"
    " route that visits a set of cities, with each city being visited exactly once and the route returning to the original city.",
    "Initial_question": "You must find the shortest path that visits all {total_cities} cities,"
    " labelled from 1 to {total_cities}. The distances between each pair of cities are provided.",
    "Output_content": "Please list each city in the order they are visited. Provide the total distance of the trip."
    " You should also provide very short step by step reasoning. Do not use multiple lines and try your best to save output tokens.",
    "Output_format": """ "answer": {'Path': '0->1->2->...->N->0', 'TotalDistance': 'INT_TOTAL_DISTANCE'}""",
}

gcpPrompts = {
    "Intro": "Graph coloring refers to the problem of coloring vertices of a graph in such a way that no two adjacent"
    " vertices have the same color. ",
    "Initial_question": "There are {max_vertices} vertices 1 to {max_vertices} in a graph. "
    "You may use {max_colors} colors with alphabats from A, B, C,... to color the graph.",
    "Output_format": """ "answer":{0:'COLOR_1', 1:'COLOR_2', ...} """,
}

mspPrompts = {
    "Intro": "The meeting scheduling problem (MSP) is a type of constraint satisfaction problem where the goal is"
    " to find a suitable time slot for a meeting that all participants can attend without conflicts in their schedules.",
    "Initial_question": "There are {total_participants} participants with their available time slots. "
    "There are {total_timeslots} consecutive non-overlapping time slots. Let's assume all meetings has duration of 1.",
    "Output_format": """ "answer": {0:[1,2], 1:[4], ...}""",
}
