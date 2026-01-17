# Code-to-NL Reasoning: Can You Tell the Difference?

**Instructions:** For each problem below, you'll see two reasoning explanations (A and B). One was written by an AI that was told to solve the problem using only natural language reasoning (no code allowed). The other was written by a different AI that was given Python code and asked to translate it into natural language.

**Your task:** For each problem, guess which one is the **original NL reasoning** (written without seeing code) and which is the **code translation** (translated from Python code).

Record your answers, then check the answer key at the bottom!

---

## Problem 1: Activity Selector

**Problem Description:** Given the input data for a activity selector algorithm, determine the answer.

**Ground Truth Answer:** `[01000101010010100000]`

---

### Option A

I sorted activities by finish time and greedily selected non-overlapping activities. Starting with activity 14 (finish: 0.071), I selected activity 9 (start: 0.383 > 0.071), then activity 12 (start: 0.456 > 0.414), activity 5 (start: 0.639 > 0.568), activity 1 (start: 0.715 > 0.645), and finally activity 7 (start: 0.891 > 0.799). This gives a maximum of 6 non-overlapping activities.

---

### Option B

To schedule the maximum number of non-overlapping activities, the key idea is to always keep as much remaining time available as possible for future choices. The classic optimal strategy for this is a greedy rule: **repeatedly choose the activity that finishes earliest among those that are compatible with what’s already chosen**.

So we first conceptually rank all activities by their finishing times from smallest to largest. Then we begin with no activity selected yet, meaning any activity is initially compatible. We take the one with the earliest finish time, because finishing sooner leaves the largest window to fit additional activities afterward. After choosing it, we can only take activities whose start time is at least as large as the finish time of the most recently chosen activity. Continuing in this way—always taking the next compatible activity with the earliest finish time—produces an optimal maximum-size set.

Applying this reasoning to the given start/finish times, the compatible activities chosen in order of increasing finish time are those with intervals:
- (0.018, 0.071)
- (0.118, 0.423)
- (0.461, 0.602)
- (0.639, 0.645)
- (0.681, 0.870)
- (0.891, 0.944)

These six do not overlap, and no method can select more than this without causing a conflict, because the earliest-finish greedy choice is provably optimal for maximizing the count.

**Therefore, the maximum number of activities that can be selected is 6.**

---

**Your guess for Problem 1:** Original NL = ___ (A or B)

---

## Problem 2: Binary Search

**Problem Description:** Given the input data for a binary search algorithm, determine the answer.

**Ground Truth Answer:** `6`

---

### Option A

We are given a sorted array `key` and a target value `0.128`. We want to find the index of the target value in the array using binary search. If the target value is not found, we return -1.

1. Initialize `low` to 0 and `high` to the length of the array minus 1 (46).
2. While `low` is less than or equal to `high`:
   a. Calculate the middle index `mid` as the average of `low` and `high`, rounded down to the nearest integer.
   b. If the value at `key[mid]` is equal to the target, return `mid`.
   c. If the value at `key[mid]` is less than the target, update `low` to `mid + 1`.
   d. If the value at `key[mid]` is greater than the target, update `high` to `mid - 1`.
3. If the target is not found after the loop finishes, return -1.

Let's trace the execution:

- `low` = 0, `high` = 46, `mid` = 23, `key[23]` = 0.602. 0.602 > 0.128, so `high` = 22.
- `low` = 0, `high` = 22, `mid` = 11, `key[11]` = 0.414. 0.414 > 0.128, so `high` = 10.
- `low` = 0, `high` = 10, `mid` = 5, `key[5]` = 0.118. 0.118 < 0.128, so `low` = 6.
- `low` = 6, `high` = 10, `mid` = 8, `key[8]` = 0.264. 0.264 > 0.128, so `high` = 7.
- `low` = 6, `high` = 7, `mid` = 6, `key[6]` = 0.143. 0.143 > 0.128, so `high` = 5.
- `low` = 6, `high` = 5. The loop terminates because `low` > `high`.

The target value 0.128 was not found in the array. Therefore, we return -1.

---

### Option B

The list of numbers is already sorted in increasing order, so the right strategy is to use binary search: repeatedly compare the target value to the middle element of the current search range and discard the half where the target cannot possibly lie.

The target is 0.128. Start by checking the middle of the full list: the middle value is 0.456, which is greater than 0.128, so the target—if present—would have to be in the left half. In that left half, the next middle value is 0.143, still greater than 0.128, so the search narrows further left. The next middle value becomes 0.071, which is less than 0.128, so now the target must lie to the right of 0.071 within that narrowed range. Checking the middle of that remaining subrange gives 0.118, which is still less than 0.128, so the target must be to the right of 0.118. The next middle value is 0.143 again, which is greater than 0.128, so the target would have to be between 0.118 and 0.143.

At that point, there are no remaining elements between them in the list: the sorted list jumps directly from 0.118 to 0.143. Since 0.128 lies strictly between these two consecutive values and does not appear as an entry, the target is not in the list.

Therefore, the correct result is to report “not found,” returning **-1**.

---

**Your guess for Problem 2:** Original NL = ___ (A or B)

---

## Problem 3: Bubble Sort

**Problem Description:** Given the input data for a bubble sort algorithm, determine the answer.

**Ground Truth Answer:** `[0.4230.5440.5480.6020.715]`

---

### Option A

The task is to sort the given list of five decimal values into nondecreasing order using the idea of repeatedly comparing adjacent elements and swapping them whenever they are out of order. With each full pass through the list, the largest remaining unsorted value is pushed to the far right, so after enough passes the entire sequence becomes sorted.

Starting from \([0.548, 0.715, 0.602, 0.544, 0.423]\), the values are rearranged into ascending order:
\[
[0.423,\ 0.544,\ 0.548,\ 0.602,\ 0.715].
\]
After completing the sorting process, the required output is simply the constant value \(1\).

---

### Option B

To solve the bubble sort problem, we need to determine the number of swaps required to sort the given array in ascending order. The given array is [0.548, 0.715, 0.602, 0.544, 0.423]. We will simulate the bubble sort process step by step:

1. First pass:
   - Compare 0.548 and 0.715: swap (0.548, 0.715) → [0.548, 0.602, 0.544, 0.423, 0.715]
   - Compare 0.548 and 0.602: swap (0.548, 0.602) → [0.548, 0.544, 0.423, 0.602, 0.715]
   - Compare 0.548 and 0.544: swap (0.548, 0.544) → [0.544, 0.423, 0.548, 0.602, 0.715]
   - Compare 0.544 and 0.423: swap (0.544, 0.423) → [0.423, 0.544, 0.548, 0.602, 0.715]
   - Total swaps in first pass: 4

2. Second pass:
   - Compare 0.423 and 0.544: no swap
   - Compare 0.544 and 0.548: no swap
   - Compare 0.548 and 0.602: no swap
   - Compare 0.602 and 0.715: no swap
   - Total swaps in second pass: 0

Since no swaps were made in the second pass, the array is now sorted. The total number of swaps required is 4.

---

**Your guess for Problem 3:** Original NL = ___ (A or B)

---


## Answer Key

<details>
<summary>Click to reveal answers</summary>

- Problem 1: A = Original NL, B = Code Translation
- Problem 2: A = Original NL, B = Code Translation
- Problem 3: A = Code Translation, B = Original NL

</details>

---

*Generated from LLM performance experiment data. The "Original NL" was generated by an LLM instructed to reason without code. The "Code Translation" was generated by GPT-5.2 given the Python solution and asked to translate it to natural language.*
