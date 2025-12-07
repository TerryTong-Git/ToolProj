from src.exps_performance.utils import clean_code_llm


def test_clean_code():
    sample = """
    ```python abc```
    """
    assert clean_code_llm(sample) == " abc", "parsing wrong"
    sample2 = "```python\n# Import necessary libraries\n\n# Define a function to find the shortest path\ndef GETPATH(graph, start, end):\n    # Your code to find the shortest path using Dijkstra's algorithm or any suitable method\n    return path, total_distance\n\n# Define the graph as an adjacency list\ngraph = {\n    0: {3: 4},\n    1: {0: 1, 3: 5, 2: 1},\n    2: {3: 1},\n    3: {}\n}\n\n# Find the shortest path from node 0 to node 3\npath, total_distance = GETPATH(graph, 0, 3)\n\n# Store the result\nanswer = {'Path': path, 'TotalDistance': total_distance}\n\n```"

    answer = "\n# Import necessary libraries\n\n# Define a function to find the shortest path\ndef GETPATH(graph, start, end):\n    # Your code to find the shortest path using Dijkstra's algorithm or any suitable method\n    return path, total_distance\n\n# Define the graph as an adjacency list\ngraph = {\n    0: {3: 4},\n    1: {0: 1, 3: 5, 2: 1},\n    2: {3: 1},\n    3: {}\n}\n\n# Find the shortest path from node 0 to node 3\npath, total_distance = GETPATH(graph, 0, 3)\n\n# Store the result\nanswer = {'Path': path, 'TotalDistance': total_distance}\n\n"
    assert clean_code_llm(sample2) == answer, "parsing wrong"

    sample3 = "```python\ndef GETPATH(edges):\n    graph = {i: {j: edges[i][j] for j in edges[i] if i!=j} for i in range(len(edges))}\n    path = [0]\n    visited = set([0])\n    while path[-1] != 3:\n        neighbors = graph[path[-1]]\n        next_node = min(neighbors, key=lambda x: neighbors[x])\n        path.append(next_node)\n        visited.add(next_node)\n    return path, sum([edges[path[i]][path[i+1]] for i in range(len(path)-1)])\n\nanswer = GETPATH([[0, 5, 1], [4, 0, 0], [4, 0, 0], [0, 1, 0]])\nprint(answer[0])\nprint(answer[1])\n```"
    answer3 = "\ndef GETPATH(edges):\n    graph = {i: {j: edges[i][j] for j in edges[i] if i!=j} for i in range(len(edges))}\n    path = [0]\n    visited = set([0])\n    while path[-1] != 3:\n        neighbors = graph[path[-1]]\n        next_node = min(neighbors, key=lambda x: neighbors[x])\n        path.append(next_node)\n        visited.add(next_node)\n    return path, sum([edges[path[i]][path[i+1]] for i in range(len(path)-1)])\n\nanswer = GETPATH([[0, 5, 1], [4, 0, 0], [4, 0, 0], [0, 1, 0]])\nprint(answer[0])\nprint(answer[1])\n"
    assert clean_code_llm(sample3) == answer3, "parsing wrong"

    sample4 = "```python\nimport numpy as np\ndef GETPATH(a):\n    out = np.min(a, axis=0) \n    return a[np.argmin(a, axis=0)], out\nanswer = GETPATH([6, 5, 3, 2, 3])\n```"
    answer4 = "\nimport numpy as np\ndef GETPATH(a):\n    out = np.min(a, axis=0) \n    return a[np.argmin(a, axis=0)], out\nanswer = GETPATH([6, 5, 3, 2, 3])\n"
    assert clean_code_llm(sample4) == answer4, "parsing wrong"
