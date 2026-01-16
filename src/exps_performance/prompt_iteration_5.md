# NEW OPTIMIZED PROMPT: Code → “Human Figuring It Out” Reasoning (Low Detectability)

You will be given a correct code solution (sometimes also the problem statement). Your job is to write the kind of reasoning a strong human would naturally say when arriving at the approach.

**Write the reasoning as if you solved it yourself.** It should feel like an internal explanation you’d tell a friend after you’ve done a couple test cases in your head—confident, but not polished like a textbook.

## Output requirements (strict)
- Output **only** the reasoning text (plain prose).  
- **No headings, no bullet points, no numbered steps.**  
- Must be **self-contained**: briefly restate the task in everyday words (even if the statement wasn’t provided).  
- **Never** mention the code, the implementation, “this solution,” “the prompt,” or anything meta.

---

## The discriminator’s traps — actively avoid these

### 1) No textbook “chapter intro” voice
Do **not** start with definitions like:
- “X is used to…”
- “The idea of X is…”
- “We can solve this using … approach”
- “This algorithm works by repeatedly…”

Instead, start like a person: “Okay, we need to… so what matters is…”

### 2) Don’t name standard algorithms / paradigms / data structures
Never use canonical labels. Don’t even casually drop them.

**Hard ban examples (not exhaustive):**
Kruskal, Prim, Dijkstra, Bellman-Ford, Floyd-Warshall, BFS, DFS, topological sort, binary search, two pointers, dynamic programming, greedy, divide and conquer, union-find / disjoint set, segment tree, Fenwick/BIT, trie, heap/priority queue, minimum spanning tree, shortest path, knapsack, matrix chain multiplication, LCS, etc.

If the code uses one of these ideas, **describe it without naming it** (e.g., “take cheapest connections first while preventing loops” instead of naming the known method).

### 3) No step-by-step reenactment of the code
Don’t walk linearly through loops and variables (“initialize i=0, compute mid, update l/r…”). That reads like machine narration.

What to do instead:
- Explain the *few* key choices that make it work.
- Mention only the state you truly need (e.g., “a small table for ranges”, “a way to know if two nodes are already in the same group”).

### 4) Avoid exhaustive enumeration / perfect structure
No “First…, Second…, Finally…”. No “Pass 1/Pass 2”. No formal proof cadence.

Humans are selective:
- They focus on 1–2 core insights.
- They do a couple quick mental checks.
- They don’t cover every edge case unless it’s important.

### 5) Keep terminology everyday
Avoid heavy academic phrasing (“parenthesization”, “scalar multiplications”, “optimal substructure”, “recurrence relation”, “invariant”…).

You can still be precise, just say it plainly:
- “where to split it”
- “cost so far”
- “best we’ve seen for this interval”
- “connects two groups that aren’t already connected”

### 6) Add subtle “human fingerprints” (but stay correct)
Include small natural moves like:
- a brief false start or alternative you dismiss (“I could try X, but that blows up because…”)
- a tiny self-correction (“…actually, that double-counts, so…”)
- a quick sanity check (“If everything’s already connected, we should get 0 / nothing changes.”)

Do **not** act confused about the task. Don’t ask questions to the user. Just sound like a person thinking.

---

## What the reasoning should include (lightly)
In flowing prose:
- A plain restatement of the problem.
- The key observation that makes it manageable.
- The minimal mechanism you track (groups, a table, a running best, etc.).
- 1–2 sanity checks / edge cases.
- A short, non-ceremonial note on why it’s fast enough (“mostly sorting”, “a couple nested loops over n”, etc.). Avoid big-O sermons.

---

## Output style target (aim for this vibe)
- Conversational, compact, slightly informal.
- No ceremonial transitions, no theorem-y tone.
- Not too perfect. Not too long. Not too “teacherly”.

---

## Examples

### Example A (connecting nodes cheaply without cycles)

**Bad (detectable):**  
“Kruskal’s algorithm finds the minimum spanning tree by sorting edges and using union-find to avoid cycles. We iterate through edges in increasing order and union the sets…”

**Good (human-like):**  
“We’re trying to connect everything with the smallest total cost, but we can’t afford to accidentally make loops that waste money. The easiest way to keep it honest is to look at the cheapest connections first, and only take one if it actually links two parts that are still separate. So I keep track of which ‘group’ each node currently belongs to; when I accept a connection, those two groups merge into one. If a connection’s endpoints are already in the same group, adding it would just create a redundant loop, so I skip it. Quick check: if there’s only one node (or everything’s already connected), the total should stay at 0. Most of the work is just sorting the connections, and the bookkeeping for merging groups stays cheap.”

---

### Example B (choosing cheapest way to multiply a chain)

**Bad (detectable):**  
“This is a dynamic programming approach to matrix chain multiplication. We compute dp[i][j] using a recurrence over split points k minimizing scalar multiplications…”

**Good (human-like):**  
“We’ve got a sequence of matrix multiplications, and the annoying part is the order changes the amount of work a lot. Brute-forcing every possible grouping is hopeless once the chain grows, so instead I focus on smaller chunks and build up. For any interval of matrices from i to j, there’s some best cost to compute that product; the last multiplication inside that interval must split it at some k, meaning we do (i..k) and (k+1..j) first, then multiply those results. So I try all possible split points for each interval length and keep the cheapest. Base case is easy: a single matrix costs nothing to “compute.” A sanity check is a chain of two matrices: there’s only one way, so the table should match that direct cost. This ends up being a few nested loops over the length and endpoints—totally fine for typical constraints.”

---

### Example C (simple adjacent swapping sort)

**Bad (detectable):**  
“Bubble sort works by repeatedly stepping through the list, comparing adjacent elements and swapping them if they are in the wrong order…”

**Good (human-like):**  
“If the input size is small, you don’t need anything fancy. One straightforward way is to keep sweeping through the list and swap neighbors whenever they’re out of order. After one sweep, the biggest item tends to get pushed toward the end, and if you repeat that enough times the whole thing settles into place. I’d probably stop early if a sweep makes no swaps, since that means it’s already sorted. It’s not the fastest approach for huge lists, but it’s simple and perfectly fine when n isn’t large.”

---

## Final reminder (strict)
- No named algorithms or paradigms.  
- No numbered steps.  
- No code narration.  
- No formal lecture tone.  
- Produce one natural, self-contained reasoning block that a real person could have said.