# ULTRA-UNPOLISHED ICL PROMPT: Code → “me kinda mumbling what I think it does”

You’ll get a chunk of code. Write what it does the way I’d explain it to a friend after staring at it for a minute and only *mostly* getting it. The goal is to sound like a real person being a bit sloppy, not like a neat textbook summary.

## Style rules (more important than being correct/complete)
- **Casual + unsure:** add **3–6** little hedges like “I think…”, “kinda”, “basically”, “pretty sure”, “from what I can tell”, “idk”, “maybe”.
- **Imperfect flow:** short sentences. Fragments. A tiny run-on is fine. Include **one** quick self-correction (e.g., “wait—no, actually…”). Optional small aside in parentheses.
- **Don’t tie it up nicely:** avoid a clean conclusion or “this is optimal” vibe. Leave a little fuzziness. It should feel like you’re recalling it, not presenting it.
- **Selective:** only **1–2 main takeaways**. Do **not** walk through every loop/branch. No step-by-step. No mini-derivations.
- **Use 2–4 anchors from the code** by name (like `dp[i][j]`, `parent`, `find(x)`, `edges.sort(...)`, `memo`, `while`, etc.). Mention them naturally, not as a list.
- **Avoid precision:** don’t include exact numeric results, worked examples, or “if dims are X then Y” type calculations. No specific computed totals.

## What to avoid (big giveaways)
- **No famous technique names** (no “Dijkstra”, “BFS”, “DP”, “greedy”, “topological”, etc.).
- No lecture voice: don’t say “This code implements…”, don’t do “First/Second/Finally”, no bullet points.
- No formal definitions (“X is used to…” / “in order to…” / “therefore…”).
- Don’t over-justify. Don’t sound confident.

## Length / formatting
- **90–170 words** total.
- **1–2 short paragraphs max.**
- Readable, but slightly messy is good.

## Output format requirement
- Write the explanation text only.
- End with **exactly one plain closing line** that starts with: **“So in the end it returns …”** (finish the sentence).