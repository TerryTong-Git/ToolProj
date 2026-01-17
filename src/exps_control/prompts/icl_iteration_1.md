# NEW OPTIMIZED ICL PROMPT: Code → “teammate explaining it in a PR comment”

You get a chunk of code. Write a natural-language explanation that sounds like a real developer talking to another developer: clear, slightly informal, practical. Not a lecture, not a meme, not a math proof.

## Target vibe (aim for “human and believable”)
- Neutral-to-casual tone (like Slack/PR), **not** overly chatty (“lol”, “yeah”, “wait—actually…”) and **not** academic (“therefore”, “we define”, “invariant”).
- Explain the **point** of the code and the **one or two key ideas** that make it work.
- Use **2–5 concrete anchors** from the code: variable names (`parent`, `best[i][j]`), key conditions (`if find(u) != find(v)`), constants (`10**18`), etc.
- It’s fine to be slightly non-exhaustive (“it’s basically tracking…”, “it’s choosing the cheapest…”), but don’t be wrong about the big behavior.

## Structure constraints (to avoid “AI pattern”)
- **No headings, no bullet lists, no numbered steps.**
- **1–2 short paragraphs** (usually 90–170 words). If code is huge, you can go up to ~220, but still keep it tight.
- Don’t narrate every loop index. Don’t do a full trace.
- Avoid “This code implements …” and other textbook opener phrases.
- Avoid overly formal math walkthroughs (e.g., manual arithmetic step-by-step). Mention the mechanism, not a classroom derivation.

## Terminology guardrails
- Don’t name-drop famous algorithms/data-structures as labels (avoid: Kruskal, Prim, Dijkstra, BFS/DFS, “dynamic programming”, “greedy”, “binary search”, “topological sort”, etc.).  
  You can describe what it does (“keeps groups”, “tries all split points”) without the label.
- Don’t over-define terms (“X is used to…”). Just use them naturally.

## Required ending
End with **one plain sentence** stating what the function returns/prints/produces.  
No fixed phrase; just a simple final line like: “It returns …” / “The output is …”.

---

## Example 1 (accepting only “useful” connections)

**Code:**
```python
def solution():
    edges = [(0,7,0.546), (2,7,0.247), (6,7,0.033), (8,9,0.138)]
    edges.sort(key=lambda x: x[2])
    parent = list(range(10))

    def find(x):
        return x if parent[x] == x else find(parent[x])

    def union(a, b):
        parent[find(a)] = find(b)

    count = 0
    for u, v, w in edges:
        if find(u) != find(v):
            union(u, v)
            count += 1
    return count
```

**Good explanation:**

It’s scanning through `edges` from smallest weight to largest (`edges.sort(... x[2])`), but it only “accepts” an edge if it actually connects two previously separate groups. The `parent = list(range(10))` plus `find(x)` is the tell: it keeps chasing `parent[x]` until it hits a representative, so `find(u) != find(v)` is basically “are these endpoints already in the same bucket?”.

When they’re different, `union(u, v)` links the reps and it increments `count`. Notice it never sums `w`, so the weights only affect the order it considers edges, not the final number.

It returns how many edges were accepted.

---

## Example 2 (best cost across a range)

**Code:**
```python
def solve(dims):
    n = len(dims) - 1
    best = [[0]*n for _ in range(n)]
    split = [[-1]*n for _ in range(n)]

    for gap in range(1, n):
        for i in range(n-gap):
            j = i + gap
            best[i][j] = 10**18
            for k in range(i, j):
                cost = best[i][k] + best[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
                if cost < best[i][j]:
                    best[i][j] = cost
                    split[i][j] = k
    return best[0][n-1], split
```

**Good explanation:**

`best[i][j]` is treating the slice from `i` to `j` as a subproblem and trying to find the cheapest way to combine it. The code walks by increasing `gap`, so when it’s filling `best[i][j]` it already has values for smaller ranges like `best[i][k]` and `best[k+1][j]`.

For each possible split point `k`, it computes `cost = best[i][k] + best[k+1][j] + dims[i]*dims[k+1]*dims[j+1]`, keeps the minimum in `best[i][j]`, and records the winning `k` in `split[i][j]`. The `10**18` is just a big starter value so the first real cost wins.

It returns the minimum total cost for the full range plus the split table.

---

## Example 3 (adding big numbers stored as strings)

**Code:**
```python
def add_strings(a, b):
    i, j = len(a) - 1, len(b) - 1
    carry = 0
    out = []

    while i >= 0 or j >= 0 or carry:
        da = ord(a[i]) - 48 if i >= 0 else 0
        db = ord(b[j]) - 48 if j >= 0 else 0
        s = da + db + carry
        out.append(chr(s % 10 + 48))
        carry = s // 10
        i -= 1
        j -= 1

    return ''.join(reversed(out))
```

**Good explanation:**

This is doing addition on two digit strings without converting the whole thing to an int. It starts from the end with `i` and `j`, pulls the current digits (`ord(...)-48`), and keeps a `carry` across iterations. Each loop computes `s = da + db + carry`, pushes the last digit of that sum into `out` via `s % 10`, and updates `carry = s // 10`.

The `while i >= 0 or j >= 0 or carry` condition is what makes it handle different-length inputs and also any leftover carry at the end. Since digits are appended least-significant first, it does `reversed(out)` before joining.

It returns the decimal sum of `a` and `b` as a string.

---

## Now do the same for the next code block you receive.