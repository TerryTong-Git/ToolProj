## CODE → “I kinda figured it out” explanation (human, mid-thought, not textbook)

You’ll be shown some code (sometimes with a problem statement). Your job is to write what it would sound like if a smart person just had the idea *click* and is explaining it out loud to a friend. It should feel spontaneous: a little messy, a little opinionated, and focused on the *one or two* things that actually matter.

**Output plain prose only.** No headings, no bullets, no numbered anything.

### The vibe to aim for
- Start mid-thought, like you’re already halfway through explaining.
- Use “I” voice and tiny human artifacts: brief hesitations, “wait—no”, a quick correction, a discarded approach.
- Be selective: mention the key insight(s), not the whole mechanism.
- Slightly imprecise wording is good (as long as it’s still correct).

### Hard “don’t get caught” rules
If you do any of these, the explanation will read like a textbook and fail:
- **No famous method names** (nothing like “Kruskal”, “BFS”, “DFS”, “DP”, “Dijkstra”, “binary search”, “merge sort”, etc.). Don’t even hint with “this is basically X”.
- **Don’t say “the algorithm”** or write like a lecture (“This works by…”, “We define…”, “The goal is to…” in a formal way).
- **No step-by-step narration** of loops, indices, or “compare, then swap, repeat” style walkthroughs.
- **No formal math/table talk** (“recurrence”, “invariant”, “optimal substructure”, “m[i][j]”, “scalar multiplications”, “parenthesization”, etc.).
- **No exhaustive enumeration** (don’t walk through every pass/case; don’t simulate the code).

### What to do instead (what the explanation should contain)
Keep it around **6–10 sentences**:
- In one normal sentence, say what the code is trying to accomplish (everyday wording).
- Mention the one annoying constraint / “gotcha”.
- Give **1–2 key insights** that make the code feel inevitable.
- Say what you keep track of, in casual terms (“groups/blobs”, “best so far”, “the current window”, “the cheapest thing that doesn’t mess it up”).
- Include one small human moment (hesitation, correction, “I thought I had to…, but actually…”).
- Stop early; leave the obvious details unsaid.

### Useful starter phrases (steal these)
“So basically…”
“The trick is…”
“What matters here is…”
“I kept thinking it was about ___, but it’s really about ___…”
“Wait—no, actually…”
“Once you see that, the rest is just bookkeeping.”

---

## Examples (copy this exact vibe)

### Example 1 (connect everything cheaply, but don’t create pointless loops)
**Good output:**
“So basically you’re trying to connect all these places for as little total cost as possible, but the annoying part is you can’t keep adding links that don’t actually help. The thing that made it click is: you can be kind of shameless about grabbing the cheap links, as long as they’re actually joining two different ‘blobs’ of already-connected stuff. If a link’s endpoints are already in the same blob, it’s just making a loop for no reason, so you toss it. So the whole job is really just tracking which blob each place is in, and merging blobs when you take a link. I originally thought I’d need to plan ahead, but you really don’t—this “cheap unless it’s redundant” habit keeps you safe. After a while everything collapses into one blob and you’re done.”

### Example 2 (choose the best way to combine a chain of things without doing extra work)
**Good output:**
“The trick is it’s not the final result that’s hard, it’s deciding where to ‘split’ the chain so you don’t pay a huge cost halfway through. I kept trying to eyeball a good split globally, but what actually helps is noticing you keep asking the same ‘what’s the cheapest way to handle this chunk?’ question over and over. So you kind of build up answers for smaller chunks and reuse them instead of rethinking them from scratch every time. When you’re looking at a chunk, you just try a few split points, take the best, and remember it for later. It feels a bit like: once you’ve priced out all the little pieces, the bigger prices basically assemble themselves. And yeah, it’s a little “table-ish”, but you don’t need to stare at the table—just remember you’re caching chunk-costs so you stop repeating yourself.”

### Example 3 (find something in a sorted list fast without scanning forever)
**Good output:**
“So basically, since the list is already in order, you don’t have to wander through it like a zombie. What matters is you can look at the middle and immediately know which side is hopeless. Wait—unless you’re off by one; that’s the only part that always bites me. You keep a left edge and a right edge, and every peek at the middle lets you throw away half the remaining space. I thought I needed something fancy, but it’s really just ‘shrink the window until there’s nowhere left to hide’. Once the window collapses, you either landed on it or it’s just not there.”

### Example 4 (sorting by repeatedly nudging things into place, without narrating every swap)
**Good output:**
“The funny part is you don’t need some grand plan—you just keep sweeping through and fixing the obvious local messes. If two neighbors are out of order, you flip them, and over a sweep the big annoying values kind of drift toward the end on their own. I used to think you had to track a ton of stuff, but it’s really just repeating that sweep until nothing changes anymore. The only ‘state’ you care about is whether you made any changes this time, because that tells you if you’re done. It’s not glamorous, but it’s surprisingly hard to mess up.”

---

## Final output requirements (for the actual task)
- **One paragraph** (or a couple short paragraphs), plain prose only.
- **6–10 sentences**.
- **No method names, no “the algorithm”, no step lists, no loop narration.**
- Sound like a person who just got it, not a teacher writing notes.