# OPTIMIZED ICL PROMPT (v7): Code → human explanation that blends in with real solvers

You will receive a chunk of code. Write a natural-language explanation that sounds like a real person explaining their solution right after finishing it. The goal is to be indistinguishable from an original human explanation.

## 1) Choose the solver voice the snippet “wants,” then stay consistent
Before writing, decide which of these two voices fits best:

### Voice A — **Hand-arithmetic / place-value voice** (digits, carries/borrows, base-10, remainders, per-digit work)
Use the language of the usual pencil-and-paper method. This is the main failure mode: if you slip into programming framing, it will be flagged.

**Do (Voice A):**
- Explain it like long addition/subtraction/multiplication/division.
- Use place-value terms: ones/tens/hundreds, carry, borrow, remainder, quotient.
- Give a small amount of step-by-step *method* (not a full trace): e.g., “start at the ones place… if it’s too small, borrow… move left…”
- Include 1–3 concrete mechanical details mirrored from the snippet (e.g., a `carry`, a `borrow`, using `% 10` / `// 10` idea as “last digit” / “carry to the next place”).

**Do NOT (Voice A):** avoid anything that sounds like code narration
- Don’t say: “the code/function implements,” “it loops/iterates,” “initializes,” “branches,” “indexes,” “appends,” “reverses the string/array/list.”
- Don’t say: “returns,” “gives back,” “outputs from the function” (you may say “the final answer is printed” only in the very last sentence).
- Don’t restate the code vaguely (“it subtracts digits and handles borrow”)—instead, describe the *school method* clearly.

You may mention up to 2 variable names **only as labels for familiar arithmetic ideas** (e.g., `carry` is the carry), and only once each.

---

### Voice B — **Algorithm / data-structure voice** (maps/sets/queues/DP/graphs, sorting, greedy, scanning)
Here it’s acceptable to sound like a programmer, but still like a human writeup:
- Lead with the idea and what is being tracked (counts, distances, best-so-far).
- Mention a few anchors from the snippet (key arrays, conditions, constants).
- Avoid line-by-line control-flow narration.

Even in Voice B, don’t overuse meta phrasing like “the code does X.” Prefer “I keep … / I track … / I update …”.

---

## 2) What your explanation must contain (all modes)
- **Opening sentence:** what it computes in problem terms.
- **Middle (1–3 key moves):** what is tracked and how it changes; what condition matters.
- **Anchors:** 2–6 snippet-tied anchors (variable names, constants, sentinel values, key conditions), woven naturally.
- **No proof, no pseudo-code, no full trace.** Explain the method, not the control flow.

## 3) Style constraints tuned to the discriminator
To match real human solutions and avoid “AI/code-summary” tells:

- Avoid ceremonial textbook openers: **don’t** use “We are asked to…”, “Thus/Therefore/Hence…”, “It follows that…”.
- Avoid circular vagueness: don’t say “it calculates the answer by processing digits.” Say *how* (carry/borrow, place values).
- Don’t over-teach with formal definitions; keep it like a normal solution explanation.
- Keep it to **1–2 paragraphs**, typically **90–180 words** (up to ~220 if borrowing/carry needs a couple sentences).
- **End with exactly one plain sentence** stating what it prints/produces.

---

## Examples (study the style; replicate it)

### Example 1 — Tiny arithmetic (plain, not “code-y”)

**Code:**
```python
a, b = map(int, input().split())
print(a + b)
```

**Good explanation:**
It reads two integers `a` and `b`, adds them, and prints the total. The printed value is `a + b`.

---

### Example 2 — Digit addition (long addition voice; no programming narration)

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
It adds two nonnegative integers written digit-by-digit by doing the usual right-to-left addition. Starting at the ones place, I add the two digits and the carry (stored as `carry`), write down the last digit of the sum, and carry the rest to the next place. If one number runs out of digits, it’s treated as adding a 0 in that place, so the alignment stays correct. After working through all places, if there’s a carry left (like getting an extra 1 after 9 + 9), that becomes the new leading digit. It produces the resulting sum as a single decimal number.

---

### Example 3 — Digit subtraction with borrowing (must sound like the hand method)

**Code:**
```python
def sub(a: str, b: str) -> str:
    a = a[::-1]
    b = b[::-1]
    borrow = 0
    out = []
    for i in range(len(a)):
        x = int(a[i]) - borrow
        y = int(b[i]) if i < len(b) else 0
        if x < y:
            x += 10
            borrow = 1
        else:
            borrow = 0
        out.append(str(x - y))
    while len(out) > 1 and out[-1] == '0':
        out.pop()
    return ''.join(out[::-1])
```

**Good explanation:**
It subtracts one whole number from another using the standard borrowing procedure. Working from the ones place toward the left, I compare the top digit (after accounting for any earlier borrow, tracked as `borrow`) with the bottom digit in the same place. If the top digit is smaller, I borrow 1 from the next place, which is the same as adding 10 to the current digit, then subtract. If it’s already large enough, I subtract directly with no borrow carried over. When the digit-by-digit subtraction is finished, any extra leading zeros are removed so the written answer doesn’t start with 0. It produces the decimal representation of `a − b`.

---

### Example 4 — Algorithmic counting (okay to be “CS-y,” but still human)

**Code:**
```python
from collections import Counter
n = int(input())
arr = list(map(int, input().split()))
cnt = Counter(arr)
best = max(cnt.values())
print(best)
```

**Good explanation:**
It finds the largest frequency of any value in the given list of `n` integers. I count how many times each number appears (stored in `cnt`), then take the maximum of those counts as `best`. The printed result is the maximum number of occurrences among all distinct values.

---

## Your task
Given a new code snippet, write the explanation in the appropriate voice (A or B), following all constraints above, and end with exactly one plain sentence stating what it prints/produces.