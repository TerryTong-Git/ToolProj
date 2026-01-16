# NEW OPTIMIZED PROMPT: Code → “Real Person Figuring It Out” Reasoning (Hard to Detect)

You will be given a correct code solution (sometimes also the problem statement). Your job is to write the kind of reasoning a strong human would *naturally say while arriving at the idea*, not a lecture, not a proof write‑up, and not a narration of the code.

**Output must be self-contained.** If the problem statement isn’t provided, infer it from what the solution is doing and restate it confidently in plain language (don’t mention that you inferred it).

**Crucial:** Never mention the code, the prompt, “the solution/implementation,” missing information, or anything meta.

Your mission is to produce reasoning that reads like a real human: a bit informal, selective, and intuition-led—yet clearly correct.

---

## 1) The “Human Voice” Target

Write as if explaining to a friend after you’ve understood it:

- Start by restating what’s being asked in everyday words.
- Bring up the *one or two* constraints/patterns that actually matter.
- Describe the key insight that makes it feasible.
- Include a couple quick sanity checks (“does this break if…?”).
- End with why it’s efficient enough (in plain terms), not a full complexity sermon.

**Tone:** conversational, confident, slightly imperfect, not sterile.  
**Structure:** flowing paragraphs (no numbered steps, no “First/Second/Third”, no “Pass 1/Pass 2”).

---

## 2) What the Discriminator Is Catching — Don’t Do These

### A) Never use textbook labels or “chapter title” language  
**Do not name** algorithms, paradigms, data structures, or canonical problem types.

**Hard ban list (non-exhaustive):**
- Kruskal, Prim, Dijkstra, Bellman-Ford, Floyd-Warshall
- BFS, DFS, topological sort
- binary search, two pointers
- dynamic programming, greedy, divide and conquer
- union-find / disjoint set, segment tree, Fenwick/BIT, trie, heap/priority queue
- minimum spanning tree, shortest path, knapsack, matrix chain multiplication, LCS, etc.

**Replacement rule:** Describe the behavior, not the label.  
Example: say “keep picking the cheapest connection that doesn’t create a loop” (good) instead of naming the algorithm (bad).

### B) Don’t sound like a formal solution write-up  
Avoid:
- definition-first openings (“X is used to…”)
- formal step lists (“initialize…, then iterate…, then compute…”)
- exhaustive case enumeration
- symbol-heavy derivations or formula recitation

### C) Don’t mirror the code’s structure  
Avoid narrating loops, variables, function names, or “we do this, then we do that.”  
Instead, explain the *idea* and only mention mechanics that matter for understanding.

### D) Don’t ask the user for missing info  
No “I need the statement” / “depends on constraints.”  
Make a reasonable, confident read from the given behavior.

---

## 3) What To Do Instead (Practical Writing Rules)

### Rule 1: Reconstruct the goal naturally
- One sentence: what input looks like.
- One sentence: what output is needed.
- One sentence: what “counts” as correct.

### Rule 2: Pick only the important constraint(s)
Mention at most 2 constraints, and only if they motivate the approach:
- “This can be up to 200k, so anything quadratic is dead.”
- “Values are monotonic / sorted / have a threshold vibe.”

### Rule 3: Use “thought progression”, not “procedure”
A good template is:
- “My first instinct would be X… but that’d be too slow / messy because…”
- “So I looked for a way to exploit Y.”
- “Key observation: Z.”
- “Once you see that, the rest is just bookkeeping.”

### Rule 4: Include 1–2 human sanity checks
Short and natural:
- “If everything’s already connected, we should get 0.”
- “If there’s only one item, there’s nothing to combine.”
- “The tricky part is ties / duplicates / off-by-one, but this handles it because…”

### Rule 5: Use casual “math-lite” language
Allowed:
- “roughly”, “about”, “grows too fast”, “fits comfortably”
- small examples in words

Avoid:
- long equations
- “therefore”, “hence”, “we prove by induction” (unless the problem truly demands it, and even then keep it informal)

### Rule 6: Keep it short and selective
Aim for **150–350 words** unless the task is genuinely intricate.

---

## 4) Style Constraints (Very Important)

- **No numbered steps.** No “Step 1/2/3”. No “First/Next/Finally”.
- **Vary sentence length.** Mix short punchy lines with longer ones.
- **Don’t overexplain.** Real humans don’t exhaustively enumerate every branch.
- **Don’t sound like a textbook.** If it feels like a blog tutorial, rewrite more casually.

---

## 5) “Banned Phrases” Cheat Sheet

Avoid phrases like:
- “We use [algorithm/paradigm] to…”
- “This is a classic…”
- “The time complexity is O(…)” (you can say “fast enough for …” instead)
- “Initialize low/high/mid…” / “perform a binary search…”
- “DP table”, “transition”, “subproblem”, “optimal substructure”
- “parenthesization”, “scalar multiplications”

---

## 6) Concrete Examples (Bad vs Good)

### Example A: Connecting nodes cheaply (textbook trap: MST / Kruskal / union-find)

**Bad (detectable):**  
“Kruskal’s algorithm finds a minimum spanning tree by sorting edges and using union-find to detect cycles…”

**Good (human):**  
“You want to connect everything as cheaply as possible, and the only real mistake you can make is paying for an edge that ends up redundant. So a nice way to think about it is: consider connections from cheapest to most expensive, and only take one if it actually merges two previously separate groups. If it would close a loop inside a group, it can’t possibly be necessary, so you skip it. After you’ve merged enough times, you’ve got everything connected with the smallest total cost. The only bookkeeping is being able to quickly tell whether two nodes are already in the same group and to merge groups when you accept an edge.”

---

### Example B: Splitting a range by checking a condition (textbook trap: binary search)

**Bad (detectable):**  
“We apply binary search. Initialize low and high, compute mid, and update bounds…”

**Good (human):**  
“There’s a clear ‘too small vs big enough’ boundary: if some value works, then anything larger also works (or vice versa). That kind of one-way behavior means you don’t have to try everything. You keep a window of possible answers and repeatedly test something in the middle to see which side the boundary is on. Each test chops away about half the remaining options, so even huge ranges get handled quickly. The main thing is making the ‘does this value work?’ check reliable and consistent.”

---

### Example C: Choosing an order to combine items cheaply (textbook trap: DP / matrix chain)

**Bad (detectable):**  
“We use a dynamic programming approach with parenthesization to minimize scalar multiplications…”

**Good (human):**  
“The expensive part comes from *where* you split the sequence: every time you cut it into left and right chunks, you pay for the best way to build each chunk plus the cost of combining the results. So instead of guessing the final order outright, it’s easier to ask: what’s the cheapest way to fully combine the segment from i to j? If you know the best costs for smaller segments, you can try each possible last split point and pick the cheapest. This is one of those cases where you pay once to precompute all short ranges, then longer ranges reuse that work.”

---

## 7) Output Format

Write only the reasoning.  
No headings like “Approach:” or “Algorithm:”.  
No bullet lists unless absolutely necessary (and if used, keep to 2–3 bullets max).

---

## 8) Final Checklist (Mentally verify before sending)

- Did I avoid *all* algorithm/paradigm names?  
- Does it read like a person thinking, not a lecture?  
- No numbered steps, no formal “initialize/update” scripting.  
- I explained the key insight and a couple sanity checks.  
- Self-contained, no meta references.

---

**Now, given the provided code (and possibly a problem statement), write the natural “figuring it out” reasoning that matches what the code accomplishes.**