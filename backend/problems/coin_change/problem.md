# Coin Change
Difficulty: Medium

You are given an integer array `coins` representing coin denominations and an integer `amount` representing a total amount of money.

Return the fewest number of coins needed to make up that amount. If that amount cannot be made up by any combination of the coins, return `-1`.

You may assume you have an infinite number of each coin.

Implement:

```python
def solve(input_data):
    # input_data = {"coins": [...], "amount": int}
    # return int
```

Example:
- Input: `{"coins": [1, 5, 10, 25], "amount": 30}`
- Output: `2` (one 5-cent and one 25-cent)
