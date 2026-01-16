# ICL Prompt: Code → Indistinguishable Natural Language Reasoning

You are transforming code-based solutions into natural language reasoning that sounds **exactly like someone thinking through the problem from scratch** — not someone who has seen code.

## Critical Constraints

**FORBIDDEN patterns (these reveal code origin):**
- "The algorithm initializes..." / "First, we set up..."
- "We iterate through..." / "For each element..."
- "The counter is incremented..." / "The variable tracks..."
- "When the condition is met..." / "If X equals Y..."
- "The function returns..." / "The output is..."
- Step-by-step enumeration that mirrors loop execution
- Technical terms borrowed from code: "nested loop", "sliding window", "index"

**REQUIRED patterns (natural human reasoning):**
- Start with problem intuition: "Looking at this problem, I notice..."
- Use comparative observations: "The pattern contains 3s, but the sequence only has 0s and 1s..."
- Express insights as realizations: "This means there's no way for..."
- Use natural quantifiers: "none of the positions", "at no point", "nowhere in the sequence"
- Conclude with confidence: "So the answer must be..." / "Therefore..."

## Examples

### Example 1: Pattern Matching (Impossible Match)

**Problem:** Find occurrences of pattern [3, 1, 0, 0, 3, 0] in sequence [0, 0, 0, 0, 0, 1].

**Code:**
```python
def solution() -> int:
    string = [0, 0, 0, 0, 0, 1]
    key = [3, 1, 0, 0, 3, 0]
    n, m = len(string), len(key)
    count = 0
    for i in range(n - m + 1):
        if all(string[i+j] == key[j] for j in range(m)):
            count += 1
    return count
```

**❌ BAD (reveals code origin):**
"The algorithm iterates through the string with a sliding window of length 6. At each position i, it compares elements string[i+j] with key[j]. Since string[0]=0 ≠ key[0]=3, the match fails immediately. The counter remains 0."

**✓ GOOD (sounds like natural reasoning):**
"Looking at this, I notice the pattern starts with 3 and 1, but the sequence I'm searching through only contains 0s and a single 1. Since the pattern requires a 3 in the first position, and there's no 3 anywhere in the sequence, there's no possible way to find a match. The answer is 0."

---

### Example 2: Task Scheduling

**Problem:** Given tasks with durations d = [1, 4, 2, 1] and weights w = [0.602, 0.544, 0.423, 0.645], find optimal weighted completion time.

**Code:**
```python
def solution() -> float:
    import numpy as np
    d = np.array([1, 4, 2, 1])
    w = np.array([0.602, 0.544, 0.423, 0.645])
    return round(np.sum(d * w), 3)
```

**❌ BAD:**
"The code computes the element-wise product of arrays d and w, then sums the result. This gives 1×0.602 + 4×0.544 + 2×0.423 + 1×0.645 = 4.269."

**✓ GOOD:**
"For weighted scheduling, I need to think about how each task's duration affects its weighted contribution. The first task takes 1 unit with weight 0.602, the second takes 4 units with weight 0.544, and so on. Multiplying each duration by its weight and adding them up: 0.602 + 2.176 + 0.846 + 0.645 = 4.269."

---

### Example 3: Graph Connectivity

**Problem:** Find articulation points in graph with adjacency matrix [[0,0,0,0], [0,0,0,1], [0,0,0,0], [0,1,0,0]].

**Code:**
```python
def solution() -> int:
    graph = [[0,0,0,0], [0,0,0,1], [0,0,0,0], [0,1,0,0]]
    # ... DFS-based articulation point finding
    return articulation_points
```

**❌ BAD:**
"The algorithm performs DFS from each node, temporarily removing it to check if the graph becomes disconnected. Starting from node 0, we visit reachable nodes..."

**✓ GOOD:**
"Looking at the adjacency matrix, I can see which nodes connect to each other. Nodes 0 and 2 are completely isolated — they have no edges. Only nodes 1 and 3 share an edge between them. This means the graph is already fragmented into three separate components: {0}, {1,3}, and {2}. Since removing any single node can't disconnect something that's already disconnected, there are no true articulation points. But wait — if I remove node 1, then node 3 loses its only connection. So node 1 is an articulation point. The answer is 1."

---

### Example 4: Longest Common Subsequence

**Problem:** Find LCS length between [1, 2, 3, 4] and [2, 4, 1, 3].

**Code:**
```python
def lcs(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]
```

**❌ BAD:**
"We fill a DP table where dp[i][j] represents the LCS length of the first i elements of sequence a and first j elements of sequence b. When elements match, we add 1 to the diagonal..."

**✓ GOOD:**
"I need to find the longest sequence of elements that appears in both lists, keeping the relative order. Let me trace through: from [1,2,3,4], I could pick 2 and 4 — and those appear in order in [2,4,1,3] too. That gives length 2. Could I do better? What about 1 and 3? They're both present, but in [2,4,1,3], the 1 comes after the 3, so that doesn't work. The best I can find is 2."

---

## Your Task

Transform this code into natural language reasoning that sounds like genuine problem-solving thought. Avoid ANY language that reveals you're describing code execution.

**Problem:** [PROBLEM_DESCRIPTION]

**Code:**
```python
[CODE_HERE]
```

**Natural language reasoning:**
