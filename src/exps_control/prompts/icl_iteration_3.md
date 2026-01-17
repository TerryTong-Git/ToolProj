# OPTIMIZED ICL PROMPT (v4): Code → human explanation that *sounds like the original solver*

You will receive a chunk of code. Write a natural-language explanation that reads like a real person’s solution/explanation for that exact code, as if they wrote it right after solving the problem.

Your job is not to “explain the implementation”; it’s to produce the kind of explanation a human would naturally give for the same approach. The discriminator is looking for style mismatches (too code-y vs too math-y, too abstract vs too step-by-step), so you must match the register the code suggests.

## Core rule: match the code’s *native voice*
Decide which bucket the code belongs to, and write like a human in that bucket.

1) **Arithmetic / digit manipulation / hand-calculation style**
- Explain in *manual arithmetic* terms: digits, places, carry/borrow, remainders, running totals.
- It’s OK to be a bit step-by-step and concrete (what quantity is computed first, then next).
- Avoid software-y framing: don’t talk about “storing results”, “data types”, “redundant checks”, “edge cases”, “implementation details”, “loops over indices”. Use the math objects instead.
- Also avoid formal textbook phrasing (“We are asked to compute…”, “Thus, the answer is…”, “therefore”). Keep it plain.

2) **Data-structure / graph / simulation / parsing style**
- It’s OK to sound like a developer: refer to arrays/maps/sets, `parent`, `seen`, queues, etc.
- Describe the mechanism at a medium level: what’s tracked, what gets updated, when you skip something, what condition triggers what.
- Don’t name-drop famous algorithm labels (no BFS/DFS, Dijkstra, DP, greedy, etc.). Describe behavior instead.

3) **Mixed “contest math + implementation”**
- Prefer the math idea first, then lightly anchor it to code names/conditions.
- Don’t drift into Python-specific commentary unless the behavior depends on it.

## What to include (content)
- Start with the *purpose*: what value/decision the code produces.
- Then the **1–3 key ideas** that make it work (how it gets there).
- Use **2–6 anchors from the code** (variable names, key conditions, sentinel values, important constants) so it feels grounded.
- Match granularity:
  - If the code is basically a formula with a few intermediate quantities, explain those quantities and the order they’re computed.
  - If it’s digit-wise arithmetic, describe the place-by-place process.
  - If it’s DS-heavy, describe what structures represent and how they’re updated.
- Don’t narrate every loop variable. Don’t do a full trace.

## Style constraints (anti-“AI tell”)
- No headings. No bullet points. No numbered steps.
- Aim for **1–2 paragraphs**, **90–180 words**. You may use 3 short paragraphs (up to ~220 words) if the code truly needs it.
- Avoid canned openers like “This code implements…” / “The algorithm is…”.
- Avoid proofy tone and avoid overly chatty filler.
- Don’t invent motivations or constraints not visible in the code.

## Required ending
End with **one plain sentence** stating what the function returns/prints/produces.

---

## Example 1 (digit arithmetic: sound like hand subtraction, not like code)

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

It subtracts the number `b` from `a` when both are given as digit strings and `a` is at least as large. The work is done from the ones place upward, keeping a `borrow` flag from the previous place. At each place you start with the current digit of `a` minus any borrow (`x`), compare it with the matching digit of `b` (`y`, or 0 once `b` runs out), and if `x` is smaller you add 10 and carry a borrow into the next place. The digit you write down for that place is `x - y`.

After all places are handled, any extra leading zeros are removed (but it keeps one digit if the result is zero), and the digits are read back in the normal order. It returns the decimal string for `a - b`.

---

## Example 2 (simple arithmetic expression: be concrete, not “program behavior”)

**Code:**
```python
def solve():
    n = int(input())
    s = n * (n + 1) // 2
    print(s * s)
```

**Good explanation:**

It takes an integer `n` and first computes the sum of the first `n` positive integers as `s = n(n+1)/2`, using `// 2` because the product is always even. Then it squares that sum by printing `s * s`, so the final value is \((1 + 2 + \dots + n)^2\). It prints the square of the triangular number for `n`.

---

## Example 3 (DS-heavy: developer tone is fine, but no named algorithms)

**Code:**
```python
def components(n, edges):
    g = [[] for _ in range(n)]
    for a, b in edges:
        g[a].append(b)
        g[b].append(a)

    seen = [False] * n
    sizes = []

    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        cnt = 0
        while stack:
            v = stack.pop()
            cnt += 1
            for to in g[v]:
                if not seen[to]:
                    seen[to] = True
                    stack.append(to)
        sizes.append(cnt)

    return sorted(sizes)
```

**Good explanation:**

It builds an adjacency list `g` for an undirected graph on `n` nodes, then walks through the nodes to find groups of vertices connected to each other. The `seen` array marks which nodes have already been visited, and whenever it finds an unseen node `i` it starts a new traversal from there using a `stack`. Each time it pops a vertex `v`, it counts it in `cnt` and pushes any unvisited neighbors from `g[v]`, marking them `seen` immediately to avoid revisiting.

When the stack empties, `cnt` is the size of that connected group, and it adds it to `sizes`. At the end it sorts the list of sizes. It returns the sorted sizes of all connected components.

---

## Example 4 (mixed math + conditions: plain math voice, lightly anchored to code)

**Code:**
```python
def count_pairs(a):
    a.sort()
    ans = 0
    j = 0
    for i in range(len(a)):
        while j < len(a) and a[j] - a[i] <= 2:
            j += 1
        ans += (j - i - 1)
    return ans
```

**Good explanation:**

It counts how many index pairs `(i, k)` have values within 2 of each other. After sorting `a`, all values close to `a[i]` sit in one block to its right, so for each `i` it moves `j` forward as long as `a[j] - a[i] <= 2`. At that point, the valid partners for `i` are exactly the elements in positions `i+1` up to `j-1`, which is `j - i - 1` of them, and that amount is added to `ans`. The pointer `j` never moves backward, so the scan stays linear after sorting. It returns the number of pairs whose difference is at most 2.

---

## Now do the same
Given the next code chunk, write a human explanation that matches the code’s natural voice, avoids the style mismatches above, and ends with one plain sentence saying what it returns/prints/produces.