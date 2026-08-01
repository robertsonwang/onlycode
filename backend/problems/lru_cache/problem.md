# LRU Cache Simulation
Difficulty: Medium

Simulate an LRU (Least Recently Used) cache. You are given a capacity and a list of operations to perform.

Each operation is one of:
- `["get", key]` — return the value of the key if it exists, otherwise `-1`
- `["put", key, value]` — insert or update the key-value pair; if the cache exceeds capacity, evict the least recently used item

Return the list of results for all `get` operations (in order).

Implement:

```python
def solve(input_data):
    # input_data = {"capacity": int, "operations": [["get"|"put", ...], ...]}
    # return [results of get operations]
```

Example:
- Input: `{"capacity": 2, "operations": [["put", 1, 1], ["put", 2, 2], ["get", 1], ["put", 3, 3], ["get", 2], ["put", 4, 4], ["get", 1], ["get", 3], ["get", 4]]}`
- Output: `[1, -1, -1, 3, 4]`
