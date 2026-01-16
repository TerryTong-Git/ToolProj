NEW OPTIMIZED PROMPT: Code → “Human Figuring It Out” Reasoning (Ultra Low Detectability)

You’re given a correct piece of code (sometimes also a problem statement). Your job is to write what it would sound like if a smart person figured this out on their own and is casually explaining it to a friend right after solving it.

Write natural “thinking-out-loud” reasoning, not a lesson. The goal is: it should read like someone remembering the key insight(s), not narrating a procedure.

STRICT OUTPUT RULES
- Output ONLY the reasoning text as plain prose.
- No headings, no bullet points, no numbered lists.
- No formulas/notation dumps (like f(i,j)=… or m[i][j]=…).
- Must be self-contained: briefly restate the task in everyday words even if the statement wasn’t provided.
- Never mention the code, the implementation, “this solution,” “the prompt,” or anything meta like “the program does…”.
- Keep it selective: emphasize 1–2 key ideas and maybe a quick sanity check. Do not walk through every iteration.

TONE + SHAPE (IMPORTANT)
- Start mid-thought, like: “So basically…”, “What matters is…”, “The trick is…”, “At first I thought…, but…”.
- Allow tiny human artifacts: a quick correction (“wait—actually…”), a small hesitation (“kinda”, “more or less”), a brief alternative you discarded.
- Use everyday language. If you need a structure, describe it vaguely: “a little table”, “a running best”, “groups of connected things”, “a to-do list”, “a stack-ish thing” (but don’t get cute).
- Do not over-explain obvious stuff. Assume the listener knows basics.

HARD AVOIDS (DISCRIMINATOR TRAPS)
1) Never name standard algorithms/paradigms/data structures.
Do not say or imply canonical labels. Hard-ban includes (not exhaustive): Kruskal, Prim, Dijkstra, Bellman-Ford, Floyd-Warshall, BFS, DFS, topological sort, binary search, two pointers, dynamic programming, greedy, divide and conquer, union-find/disjoint set, heap/priority queue, segment tree, Fenwick/BIT, trie, minimum spanning tree, shortest path, knapsack, matrix chain multiplication, LCS, etc.
If the idea matches one of these, describe the behavior in plain terms without naming it.

2) No textbook intro voice.
Avoid openings like:
- “X is used to…”
- “We can solve this using…”
- “The algorithm works by…”
- “The approach is to…”

3) No procedural reenactment.
Do not narrate loops, indices, or “first/then/finally” sequences. No “initialize”, “iterate”, “update l/r”, “for each i”.
Instead: mention only the key state and the key decision rule.

4) No exhaustive enumeration.
Don’t simulate every pass, every comparison, every sub-case. One tiny example or sanity check is fine, but keep it short.

5) Avoid academic jargon.
Don’t use: recurrence, invariant, optimal substructure, parenthesization, scalar multiplications, amortized, etc.
Use: “best so far”, “split point”, “cost”, “already connected”, “cheapest option that doesn’t mess things up”.

WHAT TO DO INSTEAD (THE “HUMAN” RECIPE)
- Restate the task casually in one sentence.
- Immediately jump to the crux: what you track and why it’s enough.
- Mention one subtle pitfall the method avoids (“prevents loops”, “doesn’t double-count”, “doesn’t redo work”).
- Add a quick check with a small imagined input, but do not fully trace it.

LENGTH
Usually 6–14 sentences. Shorter is fine if the task is simple. Longer only if truly needed.

EXAMPLES (CRITICAL STYLE TRAINING)

Example 1 (connecting nodes cheaply without forming loops)

Input code (example only):
```python
def solve(n, edges):
    edges.sort(key=lambda x: x[2])
    parent=list(range(n))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=parent[a]
        return a
    cost=0
    for u,v,w in edges:
        ru,rv=find(u),find(v)
        if ru!=rv:
            parent[ru]=rv
            cost+=w
    return cost
```

Good reasoning output:
“So we’re trying to connect all the points with the cheapest total cost, but we’re not allowed to create pointless loops while doing it. The way I think about it is: always grab the cheapest available connection, and only keep it if it actually merges two currently separate groups. So I need some lightweight way to know ‘are these two endpoints already in the same connected blob?’—if yes, that edge doesn’t help, it just makes a cycle, so skip it. If no, take it and merge the two blobs. After you’ve merged enough times, everything ends up in one group and the sum you kept is as small as it can reasonably be. A quick gut-check: if you already can get from A to B through other chosen links, paying extra to connect A-B directly is never going to reduce the total.”

Bad (too textbook / detectable):
“Kruskal’s algorithm finds a minimum spanning tree by sorting edges and using union-find to avoid cycles. First, sort the edges… Next, iterate…”

Example 2 (range cost minimized by trying split points, using a table)

Input code (example only):
```python
def solve(a):
    n=len(a)
    pref=[0]
    for x in a: pref.append(pref[-1]+x)
    best=[[0]*n for _ in range(n)]
    for length in range(2,n+1):
        for i in range(n-length+1):
            j=i+length-1
            s=pref[j+1]-pref[i]
            ans=10**18
            for k in range(i,j):
                ans=min(ans, best[i][k]+best[k+1][j]+s)
            best[i][j]=ans
    return best[0][n-1]
```

Good reasoning output:
“Here the task is basically: you keep combining neighboring chunks, and each combine costs the sum of what’s inside that chunk, so the order you combine in really matters. If you do it naively, you end up rethinking the same sub-ranges over and over, which is the real time-waster. The trick is to keep a little grid of ‘what’s the cheapest cost to fully combine everything from i to j’, and build up from small ranges to bigger ones. For any range, you’re just choosing where the last split was: left part gets combined somehow, right part gets combined somehow, and then you pay the total sum of the whole range once at the end. I also keep quick prefix sums so I can grab “sum from i to j” instantly instead of recounting it. If you test a tiny case like [1,2,3], you can feel why it works: you only really have two meaningful last-split options, and you pick the cheaper one.”

Bad (too formal / detectable):
“This is a dynamic programming approach with recurrence m[i][j] = min(…). We iterate over subproblem lengths…”

Example 3 (searching a sorted list for a boundary without scanning)

Input code (example only):
```python
def solve(arr, x):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo+hi)//2
        if arr[mid] < x:
            lo = mid+1
        else:
            hi = mid
    return lo
```

Good reasoning output:
“We’ve got a sorted list and we want to find where x would ‘fit’—like the first spot where values stop being smaller than x. Scanning from the front works but it’s wasteful because the sorted order is telling you more than that. So I keep a shrinking window of where the answer could still be, and each time I peek in the middle: if the middle is still too small, the answer must be to the right; otherwise it’s at the middle or to the left. The only thing I’m careful about is that it returns the leftmost valid spot, so equality should push the right side inward, not skip past it. At the end the window collapses to one position, and that index is the insertion point.”

Bad (detectable):
“We use binary search. Initialize low/high. Step 1… Step 2…”

Example 4 (simple sorting by repeated adjacent swaps — but explained like a human, not a textbook)

Input code (example only):
```python
def solve(a):
    n=len(a)
    changed=True
    while changed:
        changed=False
        for i in range(n-1):
            if a[i]>a[i+1]:
                a[i],a[i+1]=a[i+1],a[i]
                changed=True
    return a
```

Good reasoning output:
“The list is out of order, and the vibe here is just: keep fixing obvious neighboring inversions until there aren’t any left. Whenever two neighbors are the wrong way around, you swap them, and that pushes bigger values to the right over time. You repeat because one swap can reveal a new problem just before it. It’s not the fanciest way to sort, but it’s straightforward, and the stopping condition is super natural: do one full sweep with zero swaps and you’re done.”

Bad (detectable):
“Bubble sort works by repeatedly stepping through the list, comparing adjacent elements…”

FINAL REMINDER BEFORE YOU WRITE
If your reasoning sounds like a chapter from an algorithms book, it will be flagged. Make it sound like you personally reasoned it out, focusing on the core idea and one quick check, with slightly informal wording, and absolutely no canonical names.