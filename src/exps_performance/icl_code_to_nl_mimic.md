# ICL Prompt: Code → Structured Natural Language (Mimic Original Style)

You are translating code into natural language reasoning. Your output must be **INDISTINGUISHABLE** from a human's detailed problem-solving explanation.

## CRITICAL: Match the Target Style

The target style is **detailed, methodical explanation** - NOT casual chat. Your output must:
- Be **300-600 words** (match the length of detailed explanations)
- Use **complete sentences and paragraphs** (not bullet points)
- Include **step-by-step reasoning** with actual values
- Show **intermediate calculations** and checks
- Sound like someone explaining their work on a homework problem

## FORBIDDEN (instant detection)

**NEVER use these algorithm/technique names:**
- Kruskal, Prim, Dijkstra, Bellman-Ford, Floyd-Warshall
- BFS, DFS, topological sort, binary search
- Dynamic programming, DP, memoization, tabulation
- Greedy, divide and conquer, two pointers
- Union-find, disjoint set, minimum spanning tree
- Matrix chain multiplication, LCS, knapsack
- Bubble sort, merge sort, quick sort (or any named sort)
- Heap, priority queue, segment tree, trie

## REQUIRED Structure

Your explanation should follow this pattern:

1. **Problem understanding** (1-2 sentences): Restate what we're trying to find
2. **Key observation** (1-2 sentences): What makes this problem tractable
3. **Approach description** (2-3 sentences): How to solve it WITHOUT naming the technique
4. **Walkthrough with values** (main body): Show actual computation steps with numbers
5. **Verification** (1-2 sentences): Quick sanity check
6. **Final answer** (1 sentence): State the result

## Examples

### Example 1: Connecting Nodes Cheaply

**Code:**
```python
def solution():
    # Kruskal's MST implementation
    edges = [(0,7,0.546), (2,7,0.247), (6,7,0.033), (8,9,0.138)]
    edges.sort(key=lambda x: x[2])
    parent = list(range(10))
    def find(x): return x if parent[x]==x else find(parent[x])
    def union(a,b): parent[find(a)] = find(b)
    mst_count = 0
    for u,v,w in edges:
        if find(u) != find(v):
            union(u,v)
            mst_count += 1
    return mst_count
```

**Reasoning (mimic this length and detail level):**

We need to find how many connections are required to link all reachable nodes together at minimum total cost, without creating any redundant paths.

Looking at the available connections, I can see we have: node 0 to node 7 with cost 0.546, node 2 to node 7 with cost 0.247, node 6 to node 7 with cost 0.033, and node 8 to node 9 with cost 0.138.

The key insight is that we should consider cheaper connections first, and only add a connection if it actually joins two separate groups of nodes. If two nodes are already reachable from each other through previous connections, adding another link between them would be wasteful.

Let me work through this systematically. Sorting by cost: (6,7) at 0.033 is cheapest, then (8,9) at 0.138, then (2,7) at 0.247, finally (0,7) at 0.546.

Starting fresh, all nodes are in their own separate groups.

First, connection (6,7) with cost 0.033: nodes 6 and 7 are currently separate, so I add this connection. Now nodes 6 and 7 are in the same group. Connection count: 1.

Next, connection (8,9) with cost 0.138: nodes 8 and 9 are separate from everything else, so I add this. Now 8 and 9 form their own group. Connection count: 2.

Next, connection (2,7) with cost 0.247: node 2 is alone, node 7 is grouped with 6. These are different groups, so I add this connection. Now nodes 2, 6, and 7 are all connected. Connection count: 3.

Finally, connection (0,7) with cost 0.546: node 0 is alone, node 7 is in the group {2,6,7}. Different groups, so I add this. Now {0,2,6,7} are connected. Connection count: 4.

Quick check: nodes 1, 3, 4, 5 have no connections in the input, so they stay isolated. We successfully connected all nodes that could be connected.

The answer is 4 connections in the minimum-cost spanning structure.

---

### Example 2: Sorting by Adjacent Swaps

**Code:**
```python
def solution():
    arr = [0.548, 0.715, 0.602, 0.544, 0.423]
    n = len(arr)
    swaps = 0
    for i in range(n):
        for j in range(n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swaps += 1
    return swaps
```

**Reasoning:**

We need to count how many adjacent swaps are needed to sort this array in ascending order, where we can only swap neighboring elements.

The array is [0.548, 0.715, 0.602, 0.544, 0.423].

The approach is to repeatedly scan through the array, and whenever we find two adjacent elements in the wrong order (left bigger than right), we swap them. Each complete scan pushes the largest unsorted element toward the end. We continue until no more swaps are needed.

First pass through the array:
- Compare 0.548 and 0.715: 0.548 < 0.715, no swap needed.
- Compare 0.715 and 0.602: 0.715 > 0.602, swap! Array becomes [0.548, 0.602, 0.715, 0.544, 0.423]. Swaps: 1.
- Compare 0.715 and 0.544: 0.715 > 0.544, swap! Array becomes [0.548, 0.602, 0.544, 0.715, 0.423]. Swaps: 2.
- Compare 0.715 and 0.423: 0.715 > 0.423, swap! Array becomes [0.548, 0.602, 0.544, 0.423, 0.715]. Swaps: 3.

After first pass, 0.715 is in its final position. Continue with remaining elements...

Second pass:
- 0.548 < 0.602, no swap.
- 0.602 > 0.544, swap! Array: [0.548, 0.544, 0.602, 0.423, 0.715]. Swaps: 4.
- 0.602 > 0.423, swap! Array: [0.548, 0.544, 0.423, 0.602, 0.715]. Swaps: 5.

Third pass:
- 0.548 > 0.544, swap! Array: [0.544, 0.548, 0.423, 0.602, 0.715]. Swaps: 6.
- 0.548 > 0.423, swap! Array: [0.544, 0.423, 0.548, 0.602, 0.715]. Swaps: 7.

Fourth pass:
- 0.544 > 0.423, swap! Array: [0.423, 0.544, 0.548, 0.602, 0.715]. Swaps: 8.

Final array is sorted: [0.423, 0.544, 0.548, 0.602, 0.715].

The total number of swaps needed is 8.

---

## Your Task

Translate the following code into detailed natural language reasoning. Match the length (300-600 words), structure, and detail level shown in the examples. DO NOT name any algorithms.

**Code:**
```python
[CODE_HERE]
```

**Reasoning:**
