# Group Anagrams
Difficulty: Medium

Given an array of strings `strs`, group the anagrams together.

For deterministic grading, return your result in this canonical format:
- Each group sorted lexicographically.
- The list of groups sorted by the first string of each group.

Implement:

```python
def solve(input_data):
    # input_data = {"strs": [...]} 
    # return list[list[str]]
```

Example:
- Input: `{"strs": ["eat", "tea", "tan", "ate", "nat", "bat"]}`
- Output: `[["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]`
