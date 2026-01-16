# Prompt: Code → Human-Style “From Scratch” Reasoning (Indistinguishable)

You will be given a code solution (and sometimes the original problem). Your job is to write a **natural-language reasoning** that sounds like a human figuring it out on their own.

The output must feel like: *“I stared at the problem, noticed a few things, tried an angle, realized what must be true, and that led me to the answer.”*  
It must **not** feel like: *“Here is the algorithm; now I’ll describe its steps.”*

---

## Core Goal

Produce reasoning that a strong student might write while solving, with:
- intuitive observations,
- a couple of key “aha” moments,
- only the essential logic (no loop-by-loop narration),
- casual, flowing prose.

**The reasoning must not reveal that it came from code.**

---

## Absolute “Do Not Do” List (Hard Bans)

### 1) Never name standard algorithms or textbook labels
**Do not say**:  
“Kruskal,” “Prim,” “BFS,” “DFS,” “Dijkstra,” “dynamic programming,” “topological sort,” “binary search,” “two pointers,” “sliding window,” “union-find,” “disjoint set,” “stack/queue approach,” “greedy algorithm,” “DAG,” “adjacency matrix,” “connected component,” “parenthesization,” “scalar multiplications,” etc.

If the code uses one of these, you must **explain the idea without the label**.

### 2) Don’t describe the code’s mechanics
Avoid phrases like:
- “Initialize low/high…”
- “Set mid…”
- “We iterate through…”
- “For each i…”
- “If condition then…”
- “Update the variable…”
- “Return the value…”

No variable names, no indices, no line-by-line walkthrough, no step numbering that mirrors loops.

### 3) Don’t start formally
**No**: “This problem can be solved using …”  
**No**: “We define …”  
**No**: “Consider a graph G(V, E) …”  
**No**: “Let dp[i] be …”

Start like a person: what you notice first, what seems hard, what structure jumps out.

### 4) Don’t overuse formal structures and symbols
Avoid heavy notation dumps like `A[i][j] = 1`, `dp[i][j]`, `O(n log n)`, etc.  
(Complexity can be mentioned lightly at the end only if it fits naturally.)

### 5) Don’t exhaustively enumerate cases unless a human would
Don’t list every iteration or every subcase. Humans summarize patterns and only zoom into one small example when needed.

---

## What Good Reasoning Must Do Instead

### Sound like genuine thought, not a lecture
Use:
- “Looking at it, I notice…”
- “The key thing is…”
- “At first I thought… but then…”
- “This suggests…”
- “That means we can’t afford to…”
- “So what I really need is…”

### Use selective detail
- Mention only the **deciding insight**.
- If you need an example, use **one** small illustrative one, then generalize.

### Replace formal terms with everyday equivalents
Examples:
- Instead of “connected component,” say “a group of points that can all reach each other.”
- Instead of “adjacency matrix,” say “a table that tells me whether two points are directly linked.”
- Instead of “binary search,” say “keep narrowing the range by checking the middle.”
- Instead of “dynamic programming,” say “reuse results from smaller pieces instead of redoing work.”

### Keep a smooth narrative flow
No numbered steps. No “first/second/third.”  
Use paragraphs and natural transitions.

### End with a confident wrap-up
Conclude with something like:
- “So that’s why this works.”
- “That’s enough to pin down the answer.”
- “From there, the result follows.”

---

## Conversion Procedure (Internal—Don’t Mention in Output)

1. Read the code and identify the **real idea** (what property is exploited).
2. Identify the **one or two key turning points** (monotonicity, reuse of subresults, picking smallest edges, exploring outward by distance, etc.).
3. Write reasoning as if you did not see the code:
   - start from observations,
   - introduce the insight naturally,
   - explain why it guarantees correctness,
   - keep it concise and human.

---

## Examples

### Example A — Pattern Matching (Impossible Match)

**Problem:** Find occurrences of pattern `[3, 1, 0, 0, 3, 0]` in sequence `[0, 0, 0, 0, 0, 1]`.

**❌ BAD (code-flavored):**  
“We slide a window of length 6 and compare each index. Since string[0] ≠ key[0], we never increment the counter.”

**✅ GOOD (human):**  
“Right away there’s a mismatch in what values are even available. The pattern needs a 3 (in fact it starts with 3), but the sequence only contains 0s and a single 1. No matter where I try to line the pattern up, I’ll never be able to match that first 3, so there can’t be any full matches. That forces the count to be 0.”

---

### Example B — Building the Cheapest Network (MST-like code) Without Naming It

**❌ BAD (too textbook):**  
“Kruskal’s algorithm sorts edges and uses union-find to avoid cycles, forming a minimum spanning tree.”

**✅ GOOD (human):**  
“I want the total cost to be as small as possible, but I also can’t create ‘wasted’ links that just loop back without connecting anything new. So the natural strategy is: keep favoring the cheapest available connections, but only when they actually help merge two previously separate groups. If a cheap link connects two points that are already connected indirectly, adding it would just create a loop and wouldn’t help us reach any new place, so it’s safe to skip. Repeating that idea—always taking the cheapest helpful link—keeps the cost minimal while still ending up with everything connected.”

---

### Example C — Shortest Steps in an Unweighted Map (BFS-like) Without Naming It

**❌ BAD (formal/algorithm name + matrix talk):**  
“We run BFS on the graph using the adjacency matrix A[i][j]. The queue ensures shortest distance from the source node.”

**✅ GOOD (human):**  
“Since every move counts the same (one step), the clean way to get the shortest number of steps is to think in ‘rings’: first everything you can reach in 1 move, then everything you can reach in 2 moves, and so on. The first time you reach a location is automatically the best possible, because any later route would have to be at least as many moves. So if I expand outward level by level and record when I first see each place, the recorded step count is the shortest.”

---

### Example D — “Narrowing Down” Search (Binary-search-like) Without Step Script

**❌ BAD (loop narration):**  
“Initialize low and high. While low ≤ high compute mid… update bounds…”

**✅ GOOD (human):**  
“The key is that the answer changes in only one direction: once a candidate value works, anything bigger (or smaller—depending on the problem) also works. That means I don’t need to test everything. I can keep a current range of plausible answers, check something in the middle, and then throw away the half that can’t possibly contain the boundary where it flips from ‘doesn’t work’ to ‘works.’ Repeating that quickly pins down the exact cutoff.”

---

### Example E — Reusing Results From Smaller Pieces (DP-like) Without Saying “DP”

**❌ BAD (textbook labels):**  
“We use a dynamic programming approach with dp[i][j] for optimal parenthesization and scalar multiplications…”

**✅ GOOD (human):**  
“The tricky part is that a choice I make early affects what’s possible later, but the same sub-question shows up again and again: ‘What’s the best way to handle this smaller segment?’ Instead of re-solving that smaller segment every time it appears, I can settle its best cost once and then reuse it. Then for a bigger segment, I try the possible split points and combine the already-known best results from the left and right parts. Building up from smaller chunks to larger ones keeps everything consistent and avoids repeated work.”

---

## Output Requirements

- Output only the reasoning (no meta commentary like “the code does…”).
- No algorithm names, no formal definitions, no variable/index narration.
- Use natural language, smooth flow, and focus on the core insight.
- Keep it believable: a human wouldn’t sound like a textbook.

---

## Final Checklist (Before You Respond)

- Did I name an algorithm or a formal structure? If yes, remove it.
- Did I describe code mechanics (initialize, iterate, update, return)? If yes, rewrite as intent/insight.
- Does the first sentence sound like a person noticing something, not a lecture?
- Did I avoid adjacency-matrix / DAG / connected-component style vocabulary?
- Did I summarize rather than enumerate?

If all checks pass, produce the reasoning.