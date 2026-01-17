# OPTIMIZED ICL PROMPT (v6): Code → human explanation that blends in with real solvers

You will receive a chunk of code. Write a natural-language explanation that sounds like a real person explaining their solution right after solving it. The goal is to be indistinguishable from an original human explanation.

## Core rule: match the *native* explanation style the code implies
Before writing, decide what kind of “solver voice” the code corresponds to, and stick to it:

### A) If the code is digit/arithmetic heavy (carries/borrows/remainders/place values)
Write like someone describing the usual hand method. Use arithmetic language and a small amount of step-by-step (“ones place… carry… move left…”). This is where the discriminator catches you if you slip into programming talk.

**In this mode, avoid programming framing** like:
- “the code implements / routine / function”
- “initialize / iterate / loop / branch / indices”
- “reverse the string / array / list / append”
- “edge cases”

It’s okay to mention 2–4 variable names, but only as labels for familiar math quantities (e.g., `carry` is the carry).

### B) If the code is data-structure/algorithm heavy (maps/sets/queues/DP/graphs)
Then it’s fine to sound like a programmer explaining an approach, but still like a human writeup: lead with the idea, then mention a few anchors from the code (key variables/conditions/constants). Don’t narrate control flow line-by-line.

## What your explanation must contain
- First sentence: what the program computes/outputs (in problem terms).
- Middle: the 1–3 key moves (what is tracked, what gets updated, what condition matters).
- 2–6 anchors tied to the snippet (variable names, constants, sentinel values, key conditions).
- No full trace, no pseudo-proof.

## Tone + phrasing constraints (explicitly countering detection)
- Avoid ceremonial textbook openers: **don’t** say “We are asked to…”, “Thus/Therefore/Hence…”, “It follows that…”.
- Avoid “code-y” meta descriptions: **don’t** say “this function,” “the code uses a loop,” “it branches,” “it initializes,” “implementation details”.
- For arithmetic problems, prefer a “worked-solution” voice: neutral or lightly first-person (“I start from the rightmost digit…”) is acceptable if it sounds natural; don’t overdo it.
- Don’t be overly casual: avoid “just / basically / simply / kinda”.

## Output shape
- 1–2 paragraphs, usually 90–180 words (up to ~220 if the arithmetic needs a few steps).
- No headings, no bullet points, no numbered steps.
- End with exactly one plain sentence stating what it returns/prints/produces.

---

## Example 1 (tiny arithmetic: keep it plain)

**Code:**
```python
a, b = map(int, input().split())
print(a + b)
```

**Good explanation:**
It reads two integers `a` and `b`, adds them, and outputs the total. The printed value is `a + b`.

---

## Example 2 (digit addition: explain like long addition, not like code)

**Code:**
```python
def add(a: str, b: str) -> str:
    a = a[::-1]
    b = b[::-1]
    carry = 0
    out = []
    for i in range(max(len(a), len(b))):
        x = int(a[i]) if i < len(a) else 0
        y = int(b[i]) if i < len(b) else 0
        s = x + y + carry
        out.append(str(s % 10))
        carry = s // 10
    if carry:
        out.append(str(carry))
    return ''.join(out[::-1])
```

**Good explanation:**
This adds two nonnegative integers given as strings by doing the usual right-to-left addition. I line up the numbers at the ones place, add the two digits plus the current `carry`, keep the last digit as the next output digit (`s % 10`), and pass the rest to the next column (`s // 10`). If one number runs out of digits, it contributes 0 in that column. After the last column, if there’s still a `carry`, it becomes a new leading digit. It returns the sum of `a` and `b` as a decimal string.

---

## Example 3 (digit subtraction: explain like borrowing on paper)

**Code:**
```python
def subtract(a: str, b: str) -> str:
    # a >= b, both are nonnegative integer strings
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
    return ''.join(map(str, out[::-1]))
```

**Good explanation:**
This computes `a − b` in the same column-by-column way you’d do it by hand, assuming `a ≥ b`. Starting from the ones place, I subtract the current digit of `b` (or 0 if it has no digit there) from the digit of `a`, taking into account whether I had to borrow from the previous column (`borrow`). If the top digit is smaller than the bottom digit, I add 10 to it and set `borrow = 1` for the next column; otherwise the borrow resets to 0. Each column’s difference becomes the next digit of the result, and at the end I strip off any leading zeros. It returns the decimal string for `a - b`.

---

## Example 4 (graph/DS: programmer-style is fine, but still idea-first)

**Code:**
```python
from collections import deque

def shortest_path(n, edges, s):
    g = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)

    dist = [-1] * n
    dist[s] = 0
    q = deque([s])

    while q:
        u = q.popleft()
        for v in g[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist
```

**Good explanation:**
This computes the shortest number of edges from a start node `s` to every node in an unweighted graph. It builds an adjacency list `g` from `edges`, then runs a standard BFS: `dist` starts at `-1` for “unreached”, and `dist[s] = 0`. Each time a node `u` is popped from the queue `q`, any neighbor `v` that hasn’t been seen yet gets `dist[v] = dist[u] + 1` and is added to the queue. Because BFS expands level by level, the first distance assigned to a node is the shortest. It returns the array `dist` of shortest distances from `s`.

---

Now, given the next code snippet, write a human explanation following these rules.