# OPTIMIZED ICL PROMPT: Code → “human explanation that matches the code’s vibe”

You will receive a chunk of code. Write a natural-language explanation that reads like it was written by a real person who understood the code and is explaining it for someone else to maintain it.

The key is **tone-matching**: don’t force a single “programmer voice” or a single “math teacher voice”. Let the code’s domain decide the register.

## Tone + style (match what a human would naturally do)
- If the code is **systems/DS-heavy** (lots of arrays, pointers, sets, parent links, etc.), it’s fine to sound like a developer: “it keeps a `parent` array…”, “it skips cases where…”.
- If the code is **basic arithmetic / digit manipulation / manual addition/subtraction**, explain it in arithmetic terms (“carry”, “borrow”, “digit-by-digit”) and **avoid** software-y phrasing like “state machine”, “branching”, “fixed constant”, “implementation detail”.
- If it’s **problem-solve-y math**, keep it plain and practical; don’t turn it into a lecture or proof.

## What to cover (content)
- Start from the *purpose*: what the code is trying to compute or decide.
- Then give the **1–3 key ideas** that make it work (the mechanism).
- Use **2–6 concrete anchors from the code** (variable names, key conditions, sentinel values, etc.) so it feels grounded.
- Don’t narrate every loop index; don’t do a full trace. Mention repeated patterns (“loops over digits”, “tries each split point”, etc.) instead.

## Structure constraints (to avoid “AI tell”)
- **No headings. No bullet points. No numbered steps.**
- **1–2 paragraphs** is preferred (up to 3 if truly needed), usually **90–180 words**.
- Avoid canned openers like “This code implements…” / “The algorithm is…”.
- Avoid overly chatty filler (“Pretty straightforward”, “obviously”, “lol”) and avoid academic proof language (“therefore”, “we define”, “invariant”).

## Terminology guardrails
- Don’t name-drop famous algorithm labels (e.g., Kruskal, Dijkstra, BFS/DFS, DP, greedy, binary search, topological sort). Describe behavior instead.
- Don’t over-define terms (“X is used to…”). Just use names naturally.

## Required ending
End with **one plain sentence** stating what the function returns/prints/produces (no fixed wording, just make it clear).

---

## Example 1 (digit subtraction: stay arithmetic-focused, not “code-y”)

**Code:**
```python
def subtract(a: str, b: str) -> str:
    # assumes a and b are non-negative integers as strings, and a >= b
    a = list(map(int, a[::-1]))
    b = list(map(int, b[::-1]))
    out = []
    borrow = 0

    for i in range(len(a)):
        x = a[i] - borrow
        y = b[i] if i < len(b) else 0
        if x < y:
            x += 10
            borrow = 1
        else:
            borrow = 0
        out.append(x - y)

    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return "".join(map(str, out[::-1]))
```

**Good explanation:**

It subtracts `b` from `a` when both are given as digit strings, assuming `a >= b`. The digits are reversed so the loop can work from the ones place upward, and `borrow` tracks whether the previous place had to borrow 1. For each position it takes `x = a[i] - borrow`, compares it to the corresponding `y` digit from `b` (or 0 past the end), and if `x < y` it adds 10 and carries a borrow into the next position.

After building `out` in reverse, it trims leading zeros from the most significant end (keeping at least one digit), then reverses back into a normal string.

It returns the decimal string for `a - b`.

---

## Example 2 (edge filtering with parent links: allow developer tone, but keep it calm/neutral)

**Code:**
```python
def count_links(n, edges):
    edges.sort(key=lambda e: e[2])
    parent = list(range(n))
    size = [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        return True

    picked = 0
    for u, v, w in edges:
        if union(u, v):
            picked += 1
    return picked
```

**Good explanation:**

It goes through `edges` in increasing `w` order, but it only counts an edge if it actually connects two currently separate groups. The `parent`/`size` arrays are maintaining those groups: `find(x)` walks up to a representative and also compresses the path (`parent[x] = parent[parent[x]]`) so repeated lookups get cheaper. `union(a, b)` checks reps first, returns `False` if they’re already together, otherwise it links the smaller group under the larger one via `size` and returns `True`.

The loop just calls `union(u, v)` for each edge and increments `picked` when it succeeds; weights only matter for the order they’re considered.

It returns how many edges were accepted.

---

## Example 3 (range splitting table: describe the table and recurrence without name-dropping)

**Code:**
```python
def min_cost(dims):
    n = len(dims) - 1
    best = [[0]*n for _ in range(n)]

    for gap in range(1, n):
        for i in range(n-gap):
            j = i + gap
            best[i][j] = 10**18
            for k in range(i, j):
                cost = best[i][k] + best[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
                if cost < best[i][j]:
                    best[i][j] = cost

    return best[0][n-1] if n > 0 else 0
```

**Good explanation:**

It’s computing the cheapest way to combine a chain of items whose dimensions are in `dims`. The `best[i][j]` table stores the minimum cost to handle the subrange from `i` to `j`, and it fills that table from shorter ranges to longer ones using `gap`. For each interval `(i, j)`, it tries every split point `k` between them and takes the left cost `best[i][k]` plus the right cost `best[k+1][j]`, plus the “merge” cost `dims[i] * dims[k+1] * dims[j+1]`. The large `10**18` is just a sentinel to make the first comparison work cleanly.

It returns the minimum total cost for the full range.

---

## Your task
Given new code, produce an explanation in the same style as the examples, **tone-matched to the code’s domain**, with the structure constraints and the required final sentence about the output.