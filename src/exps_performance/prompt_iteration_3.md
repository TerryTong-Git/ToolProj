# NEW OPTIMIZED PROMPT: Code → “Human Figuring-It-Out” Reasoning (Hard to Detect)

You will be given a correct code solution (sometimes also the problem statement). Your job is to write the kind of reasoning a strong human would say **while arriving at the solution**, not a lecture about a known technique and not a narration of the code.

**Goal:** produce natural reasoning that’s *indistinguishable* from a real person thinking it through.

**Crucial:** The output must be **self-contained** and must **not** mention the code, the prompt, “the solution”, “the implementation”, or anything meta like “I’m missing input” or “to humanize this”.

---

## What the reasoning should feel like

Write like a smart student explaining to another student:

- Starts from what the problem is asking in plain words.
- Notices a couple of constraints/patterns and uses them to narrow down the plan.
- Includes small “wait, what about this case?” sanity checks.
- Sounds fluent and human, with mild imperfection, but **confident** (no asking the user for missing info).
- Focuses on **why the idea works**, not on procedure.

It should read like: “At first I thought X, but that’s too slow because…, so I looked for Y. The key observation is…, then everything falls out.”

---

## Absolute bans (these are what discriminators latch onto)

### A) Never use textbook labels or chapter-title language
Do **not** name standard algorithms, paradigms, or famous problem types. Avoid *even indirect* references like:

- Algorithm names (e.g., Kruskal/Prim/Dijkstra/BFS/DFS/KMP/etc.)
- Paradigms (e.g., “dynamic programming”, “greedy”, “divide and conquer”, “two pointers”, “binary search”, “topological sort”, “union-find”, “segment tree”, etc.)
- Canonical problem labels (e.g., “matrix chain order”, “minimum spanning tree”, “shortest path”, “connected components”, “knapsack”, etc.)

**Rule:** If it could appear as a heading in a CS textbook, don’t say it.

### B) No “definition-first” formalism
Avoid phrases like:
- “We define…”, “Let X be…”, “Consider the recurrence…”, “Base case…”
- “adjacency matrix”, “parenthesization”, “layer by layer”, “invariant”, “optimal substructure”
- Heavy symbol/notation dumps

You can still express the idea, just in everyday language.

### C) Don’t narrate mechanics like a tutorial
No step-by-step procedural walkthroughs like:
- “Compare adjacent elements and swap…”
- “Push into a queue, pop, mark visited…”
- “Sort edges then iterate, union sets…”
- “Fill a table from length 2 to n…”

Instead, explain **the intention** behind those actions.

### D) No numbered steps, no rigid structure
Avoid “First…, Second…, Third…” and bullet-pointed algorithm recipes. Use flowing prose with natural transitions.

### E) No meta or “assistant tells”
Do not say:
- “Given the code…”
- “The program does…”
- “We can implement…”
- “I don’t have the input”
- “To humanize…”
- “This is a known approach”

Also avoid sounding like you’re trying to be human (no wink-wink self-references).

---

## What to do instead (the “human” recipe)

### 1) Start with the *pressure* in the problem
- What makes the naive way too slow or too messy?
- What structure is being exploited (ordering, local choice, re-use of partial results, etc.)—but describe it without labels.

### 2) Explain the key insight in plain language
- “If I commit to X, then Y becomes forced.”
- “The only thing that matters about the past is …”
- “I can treat these as independent chunks once …”

### 3) Keep details selectively fuzzy
Mention only the minimum detail needed to make the logic believable. Don’t enumerate every loop-like action.

### 4) Add quick sanity checks
One or two lines like:
- “Does this break on ties/duplicates?”
- “What if everything is already sorted?”
- “What about disconnected pieces?”
- “Edge case: empty/single item.”

### 5) End with why it’s efficient (casually)
A single sentence is enough: “This avoids redoing work and stays manageable even for large inputs.”

---

## Output format requirements

- 2–6 short paragraphs (typically 120–250 words unless the problem truly needs more).
- No headings, no numbered lists, no bullet points.
- Use everyday words. Prefer “keep track of”, “connect”, “group”, “cheapest so far”, “best way so far”, “re-use” over formal terms.
- No variable names or code-level artifacts (no `i`, `dp`, `adj[i][j]`, etc.).

---

## Mini “detector traps” to avoid (common failure modes)

- Opening with: “This problem can be solved using …”
- Dropping a famous name indirectly: “classic”, “well-known”, “standard technique”
- Over-explaining mechanics (especially sorting-then-choosing, queue exploration, or pairwise swapping)
- Sounding like a lecture: crisp theorem-y statements, recurrences, base cases, and formal definitions
- Fake-human meta: “I’m missing the input”, “I can’t see the constraints”, “Let me humanize it”

---

## Examples (BAD vs GOOD)

### Example 1 (connecting points cheaply)
**BAD:** “We use Kruskal’s algorithm to find the minimum spanning tree. Sort the edges, then use union-find to avoid cycles…”

**GOOD:** “I want to connect everything while paying as little total cost as possible, but the tricky part is avoiding spending money on a connection that ends up redundant. The nice thing is: if I look at the cheapest connections first, any time a connection links two areas that were already connected through other choices, it can’t be helping—I’d be paying extra just to create a loop. So I keep taking the next cheapest connection that actually links two previously separate groups, and I skip the ones that don’t change anything. Once everything ends up in one connected group, I’m done. Ties don’t really matter; picking either cheap option first leads to the same total idea: never pay for a connection that doesn’t reduce the number of separate groups.”

---

### Example 2 (choosing how to group multiplications)
**BAD:** “This is the matrix chain order problem solved with a dynamic programming recurrence over parenthesization…”

**GOOD:** “Even though multiplying everything gives the same final result, the order can wildly change how much work I do, because intermediate results can get big. So instead of committing to a full order immediately, I think in terms of smaller chunks: if I already know the cheapest way to compute a left chunk and a right chunk, then combining them has a clear extra cost based on the sizes at the boundary. The key is that once I pick where the split is, the two sides don’t interfere anymore—I can optimize them independently. So I try all split points for every interval and keep the cheapest total. It’s basically ‘best cost for this segment’ built from smaller segments, and it avoids recalculating the same sub-choices over and over.”

---

### Example 3 (exploring a graph from a start)
**BAD:** “Use BFS with an adjacency matrix; BFS explores layer by layer with a queue…”

**GOOD:** “I need the fewest hops from the start to everything else, so it helps to think in waves: first everything reachable in 1 step, then 2 steps, and so on. If I process nodes in the same order I discover them, I naturally finish all 1-step options before I ever move on to 2 steps, which is exactly what ‘fewest hops’ means. I also keep a simple record of whether I’ve already discovered a node so I don’t bounce around in circles. By the time I reach a node the first time, that route has to be the shortest one, because any later route would have at least as many steps.”

---

## Your task each time
Given the code (and optional statement), produce a single natural reasoning passage that follows all rules above, matches the solution’s real logic, and avoids every banned signal.