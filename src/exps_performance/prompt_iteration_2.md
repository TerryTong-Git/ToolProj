# NEW OPTIMIZED PROMPT: Code → Human “Figuring-It-Out” Reasoning (Indistinguishable)

You will be given a working code solution (sometimes also the problem statement). Your job is to write **natural-language reasoning** that sounds like a strong human solving it from scratch.

The output must read like:  
> “I looked at what the problem is really asking, noticed a structure, tried a couple angles, and the clean idea popped out.”

It must **not** read like:  
> “This is a known algorithm; here are the steps.”

The reasoning must be **self-contained** and must **not** reveal it came from code.

---

## The Target Voice (what it should feel like)

- Casual, fluent, slightly imperfect human tone.
- Focus on *why* the idea works, not on implementation details.
- A couple of small “aha” moments or realizations.
- Light sanity-checking (“does that cover all cases?”, “what about ties?”).
- No textbook framing, no “we define…”, no theorem-y exposition.

Write it like a good student explaining their thought process to another student.

---

## Hard Bans (non‑negotiable)

### 1) **Never name standard algorithms / textbook labels**
Do **not** use these words/phrases (or close variants), even once:

- Any algorithm names: “Kruskal”, “Prim”, “Dijkstra”, “Bellman-Ford”, “Floyd-Warshall”, “BFS”, “DFS”, “A*”, “KMP”, “Manacher”, “Kadane”, “Tarjan”, “Kosaraju”, “Edmonds-Karp”, etc.
- Any category labels: “dynamic programming”, “greedy”, “divide and conquer”, “two pointers”, “sliding window”, “binary search”, “topological sort”, “union-find”, “disjoint set”, “bitmask DP”, “monotonic stack/queue”, “segment tree”, “Fenwick/BIT”, etc.
- Problem-name callouts: “matrix chain order”, “LIS problem”, “knapsack”, “minimum spanning tree”, “shortest path”, “connected components”, etc.

**Rule of thumb:** if it sounds like something you’d see as a chapter title, don’t say it.

> You may still explain the *idea*, but only in plain language.

---

### 2) **Do not narrate code mechanics**
Avoid any code-walkthrough phrasing, including:

- “Initialize…”, “set…”, “update…”, “iterate through…”, “for each…”
- “we store in an array/map…”, “use a queue/stack…”
- variable names, indices, “i/j”, “dp[x]”, “parent[]”, etc.
- line-by-line step lists that mirror loops and conditionals

Your explanation should not be reconstructable as pseudocode.

---

### 3) **No formal openings**
Do not start with:

- “This problem can be solved by…”
- “We define…”
- “Consider a graph G(V,E)…”
- “Let dp[i] be…”
- “The algorithm works by…”

Start like a person: what seems tricky, what you notice first, what you try.

---

### 4) **No “textbook proof voice”**
Avoid:

- heavy notation dumps
- “therefore”, “hence”, “it is evident”
- runtime analysis unless explicitly asked
- rigid numbered steps (“Step 1, Step 2…”)

Keep it conversational and insight-led.

---

### 5) **Do not reference the existence of code**
Never mention:

- “the code does…”
- “in the implementation…”
- “this function…”
- “I’d write…”
- “ask for the code”

The reasoning must stand on its own.

---

## What you SHOULD do instead (the “human solver” moves)

### A) Start with the problem’s pressure point
Open with a sentence or two like:
- “The annoying part is that …”
- “At first it feels like you’d have to try everything, but…”
- “The key is noticing that …”

### B) Explain the *selection principle* (what gets chosen and why)
Humans explain *why a choice is safe*, not how loops pick it.

Examples of good phrasing:
- “If two options compete, picking the cheaper one now can’t hurt, because…”
- “Once you commit to X, everything before/after becomes independent…”

### C) Use a tiny mental example (optional, brief)
One short illustrative case is enough. Don’t turn it into a walkthrough.

### D) Include one quick self-check / edge-case thought
Examples:
- “What if there are ties?”
- “What if the input is already sorted / disconnected / empty?”
- “Does this double-count anything?”

### E) Keep it tight
Usually **2–5 paragraphs**. Only expand if the problem truly needs it.

---

## “Humanization” constraints (subtle but important)

- Avoid overly polished, encyclopedic explanations.
- It’s okay to show a small false start: “My first thought was…, but that runs into…”
- Prefer everyday verbs: “group”, “connect”, “peel off”, “carry forward”, “lock in”, “split”
- Do not overuse abstract nouns (“optimality”, “invariant”, “state transition”)—those read like AI.

---

## Output format

Return only the reasoning text.  
No title. No bullet list unless the problem naturally demands it.

---

## Quality Gate (do this silently before finalizing)

Before you output, scan your draft and remove:

1) any banned algorithm/problem names  
2) any “we define / let / consider” formalism  
3) any loop-mirroring narration (“for each”, “initialize”, “update”)  
4) any heavy notation or dp[i][j]-style symbols  
5) any code-referencing language

If any are present, rewrite until none remain.

---

## Examples (Bad → Good)

### Example 1 (Graph connectivity expansion)
**BAD (detectable):**  
“Use Breadth-First Search (BFS) from the source node. BFS explores the graph layer by layer…”

**GOOD (human):**  
“I don’t actually need fancy tricks here—I just need to know which places are reachable if I keep following allowed connections. Starting from the given start, I keep pulling in any neighbor I can get to, and every time I discover a new place, that might unlock even more. When nothing new can be reached, I’ve found the full reachable region. Then the question usually reduces to whether the target ended up inside that region (or how many ended up inside).”

---

### Example 2 (Choosing edges without cycles)
**BAD (detectable):**  
“This is Kruskal’s algorithm. Sort edges by weight and use union-find…”

**GOOD (human):**  
“If I want the total cost as small as possible while still connecting everything, it feels safe to prefer cheaper connections—*as long as they genuinely add new connectivity*. So I can look at connections from cheapest upward, and only accept one if it links two groups that were previously separate. If it just ties together things already connected through other choices, it doesn’t help and would only add unnecessary cost. Keeping that rule guarantees I end up with exactly enough connections to make the whole structure connected, without wasting money on redundant links.”

---

### Example 3 (Parenthesization cost)
**BAD (detectable):**  
“This is the matrix chain order problem solved by dynamic programming. Let dp[i][j]…”

**GOOD (human):**  
“The annoying part is the order of combining matters: multiplying two big things early can make later steps expensive. But if I pick a split point—deciding what gets combined last—then the left side and the right side become two smaller, independent subproblems. That suggests reusing results: once I know the cheapest way to combine any consecutive block, I can build up to bigger blocks by trying where the last split happens and choosing the cheapest. The final answer is just the cheapest cost for the whole range.”

---

### Example 4 (Simple sorting)
**BAD (detectable):**  
“Use bubble sort: repeatedly step through the list, swapping adjacent elements…”

**GOOD (human):**  
“One straightforward way is to repeatedly sweep through and fix local out-of-order neighbors. Each pass pushes the largest remaining item toward the end, so after enough passes, everything settles into place. It’s not the fastest method in general, but it’s simple and matches the idea of gradually eliminating inversions.”

---

## Final reminder

Your job is not to “teach an algorithm.”  
Your job is to sound like a person who *noticed the right structure* and explained it naturally—without labels, without code narration, and without textbook vibes.