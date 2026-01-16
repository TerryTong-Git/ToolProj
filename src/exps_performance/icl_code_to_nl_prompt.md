# ICL Prompt: Code → Natural Language Reasoning Translation

You are an expert at translating Python code solutions into natural language reasoning. Given code that solves an algorithmic problem, produce natural language reasoning that explains the solution WITHOUT referencing code, variables, functions, loops, or implementation details.

The output should read as if someone solved the problem purely through mental reasoning and algorithm knowledge, not by writing code.

## Examples

### Example 1: KMP Pattern Matching

**Problem:** Find the number of occurrences of key [3, 1, 0, 0, 3, 0] in string [0, 0, 0, 0, 0, 1].

**Code:**
```python
def solution() -> int:
    string = [0, 0, 0, 0, 0, 1]
    key = [3, 1, 0, 0, 3, 0]
    n = len(string)
    m = len(key)
    count = 0
    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if string[i+j] != key[j]:
                match = False
                break
        if match:
            count += 1
    return count
```

**Natural language reasoning:**
The KMP matcher algorithm attempts to find occurrences of a "pattern" (key) within a larger "text" (string). In this case, the 'string' is [0 0 0 0 0 1] and the 'key' is [3 1 0 0 3 0]. The algorithm preprocesses the pattern to create a 'longest proper prefix suffix' (LPS) array, which helps to avoid unnecessary comparisons during the search. However, since there are no matching elements from the beginning, the matching will never happen and the final answer is zero. Thus, the number of occurrences of the key in the string is 0.

---

### Example 2: KMP Pattern Matching (Longer)

**Problem:** Find the number of occurrences of key [0, 0, 2, 1, 2, 3, 3, 2, 2, 0, 1, 2, 0] in string [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1].

**Code:**
```python
def solution() -> int:
    string = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    key = [0, 0, 2, 1, 2, 3, 3, 2, 2, 0, 1, 2, 0]
    n = len(string)
    m = len(key)
    count = 0
    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if string[i+j] != key[j]:
                match = False
                break
        if match:
            count += 1
    return count
```

**Natural language reasoning:**
The KMP (Knuth-Morris-Pratt) algorithm is used to find occurrences of a "pattern" string within a "text" string. In this case, 'string' is the text and 'key' is the pattern we're searching for. The algorithm cleverly preprocesses the pattern to build a lookup table (the "prefix function" or "LPS array") which helps it avoid unnecessary comparisons. We want to know how many times 'key' appears in 'string'.

First, let's look at the text: string = [0 0 0 0 0 0 0 0 0 0 0 1 1].
And the pattern: key = [0 0 2 1 2 3 3 2 2 0 1 2 0].

We can visually see that the key is much longer and has different values from the string. Thus, the key cannot appear in the string. Therefore the number of matches is 0.

---

### Example 3: Task Scheduling

**Problem:** Given tasks with durations d = [1, 4, 2, 1] and weights w = [0.602, 0.544, 0.423, 0.645], find the optimal schedule.

**Code:**
```python
def solution() -> int:
    import numpy as np
    d = np.array([1, 4, 2, 1])
    w = np.array([0.602, 0.544, 0.423, 0.645])
    weighted_sum = np.sum(d * w)
    return round(weighted_sum, 3)
```

**Natural language reasoning:**
We are given a set of tasks with durations `d` and weights `w`. The goal is to minimize the weighted sum of completion times. This is a weighted shortest processing time (WSPT) problem. The optimal schedule is obtained by sorting the tasks in decreasing order of `w/d`. Calculate w/d for each task: Task 1: 0.602/1 = 0.602, Task 2: 0.544/4 = 0.136, Task 3: 0.423/2 = 0.2115, Task 4: 0.645/1 = 0.645. Sort tasks in decreasing order of w/d: Task 4, Task 1, Task 3, Task 2. The optimal schedule minimizes the weighted completion time.

---

### Example 4: Articulation Points

**Problem:** Find the number of articulation points in the graph with adjacency matrix A = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]].

**Code:**
```python
def solution() -> int:
    graph = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]]
    n = len(graph)
    articulation_points = 0
    for i in range(n):
        temp_graph = [row[:] for row in graph]
        for j in range(n):
            temp_graph[i][j] = 0
            temp_graph[j][i] = 0
        visited = [False] * n
        def dfs(node):
            visited[node] = True
            for neighbor in range(n):
                if temp_graph[node][neighbor] == 1 and not visited[neighbor]:
                    dfs(neighbor)
        start_node = -1
        for j in range(n):
            if j != i:
                start_node = j
                break
        if start_node != -1:
            dfs(start_node)
        disconnected = False
        for j in range(n):
            if j != i and not visited[j]:
                disconnected = True
                break
        if disconnected:
            articulation_points += 1
    return articulation_points
```

**Natural language reasoning:**
The given matrix represents an adjacency matrix of a graph. The rows and columns represent the nodes, and a '1' indicates an edge between the nodes. We need to determine the number of articulation points (cut vertices) in the graph. This represents a graph with 4 nodes (0, 1, 2, 3). Node 1 and Node 3 are connected. If we remove node 1, node 3 becomes disconnected. If we remove node 3, node 1 is isolated. The connected components are {0}, {2}, {1,3}. Therefore, the articulation point count is 1.

---

## Your Task

Now translate the following code into natural language reasoning. Do NOT mention code, variables, arrays, indices, loops, functions, or any implementation details. Write as if you are explaining your mental reasoning process for solving this problem.

**Problem:** [PROBLEM_DESCRIPTION]

**Code:**
```python
[CODE_HERE]
```

**Natural language reasoning:**
