# OPTIMIZED ICL PROMPT (v5): Code → human explanation that blends in with real solvers

You will receive a chunk of code. Write a natural-language explanation that sounds like a real person explaining their solution right after solving it. The goal is to be indistinguishable from an original human explanation.

This is not a line-by-line walkthrough and not a proof. It should read like the kind of explanation you’d see in a contest solution, a LeetCode discussion, or a homework “here’s what I did” writeup—practical, grounded, and consistent in tone.

## Main objective: match register and granularity to the code
Before writing, judge what the code “feels like” and keep the voice consistent:

- If the code is basically arithmetic/digits/formulas, write in plain math language (digits, places, carry/borrow, remainder, running total), but don’t turn it into a textbook derivation.
- If the code uses data structures (maps/sets/arrays/queues/parents), it’s fine to talk like a programmer, but still explain the idea rather than narrating loops.
- If it’s mixed, lead with the idea in plain terms, then anchor it to a few variable names/conditions.

## What to include (content)
- Start by stating what the code is trying to produce (the output/return meaning).
- Give the 1–3 key moves that get there (what gets tracked, what’s updated, what condition matters).
- Use 2–6 anchors from the code (variable names, constants, sentinel values, key conditions) so it feels tied to the actual snippet.
- Don’t trace every iteration. Don’t list every variable.

## Style constraints (the discriminator is catching these)
Your explanation must avoid the common “AI tells” and mismatched tone:

- Do NOT use textbook/ceremonial phrasing: avoid “We are asked to…”, “Thus, the answer is…”, “therefore”, “hence”, “it follows that”.
- Do NOT be extremely casual or dismissive: avoid “just”, “basically”, “simply”, “kinda”, “pretty much”, “giving it back”, “fixes it”.
- Do NOT sound like you’re explaining code mechanics: avoid “the code implements”, “this function”, “initialize a variable”, “iterate over indices”, “data types”, “edge cases” unless the code clearly handles them.
- Aim for a steady, matter-of-fact human voice: clear and plain, not overly formal, not chatty.

## Output shape
- 1–2 paragraphs, usually 80–170 words (up to ~210 if truly needed).
- No headings, no bullet points, no numbered steps.
- End with exactly one plain sentence stating what the code returns/prints/produces.

---

## Example 1 (tiny arithmetic: keep it plain, not over-explained)

**Code:**
```python
a, b = map(int, input().split())
print(a + b)
```

**Good explanation:**
It reads two integers `a` and `b` from input, adds them together, and outputs the sum. The printed value is `a + b`.

---

## Example 2 (digit arithmetic: describe it like hand subtraction, but not like a proof)

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
    return ''.join(map(str, out[::-1]))
```

**Good explanation:**
It subtracts `b` from `a` the same way you’d do it on paper, digit by digit from the ones place upward. The strings are reversed so index `0` is the current digit, and `borrow` keeps track of whether the previous place had to take 1. For each position it uses `y = b[i]` when `b` still has digits, otherwise `0`, and if the current `a` digit minus the borrow is too small, it adds 10 and carries a borrow into the next place. After building `out`, it trims leading zeros from the most significant end and flips the digits back. It returns the decimal string for `a - b`.

---

## Example 3 (data-structure / graphy feel: developer voice, but idea-first)

**Code:**
```python
from collections import deque

def shortest_steps(n, edges, start, goal):
    g = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)

    dist = [-1] * n
    q = deque([start])
    dist[start] = 0

    while q:
        u = q.popleft()
        if u == goal:
            break
        for v in g[u]:
            if dist[v] != -1:
                continue
            dist[v] = dist[u] + 1
            q.append(v)

    return dist[goal]
```

**Good explanation:**
It builds an adjacency list `g` from `edges`, then explores outward from `start` while tracking the first time each node is reached in `dist`. Unvisited nodes stay at `-1`, and whenever it discovers a neighbor `v` from `u`, it sets `dist[v] = dist[u] + 1` and pushes `v` into the queue. Once `goal` is popped, it can stop early because the recorded distance won’t improve after that. The function returns `dist[goal]`, which is the number of edges in the shortest path, or `-1` if `goal` was never reached.

---

## Example 4 (mixed contest-math + code anchors: concise, not ceremonial)

**Code:**
```python
def count_pairs(nums):
    freq = {}
    ans = 0
    for x in nums:
        ans += freq.get(x, 0)
        freq[x] = freq.get(x, 0) + 1
    return ans
```

**Good explanation:**
It counts how many equal-value pairs appear in `nums`. As it scans left to right, `freq[x]` stores how many times value `x` has already shown up. When it sees another `x`, it adds `freq[x]` to `ans` because each previous occurrence forms a new pair with the current one, then increments the stored count. This way every pair is counted exactly once, at the moment the second element of the pair is processed. It returns the total number of index pairs `(i, j)` with `i < j` and `nums[i] == nums[j]`.

---

## Your task
Given new code, produce a human-sounding explanation that matches the code’s natural register and avoids the tone mismatches listed above.

End with one plain sentence stating what the function returns/prints/produces.