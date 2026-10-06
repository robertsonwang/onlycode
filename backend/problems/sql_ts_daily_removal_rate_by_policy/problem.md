# SQL: Daily Removal Rate by Policy
Difficulty: Easy

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB. A `SAFE_DIVIDE` helper is pre-registered.

One table, `moderation_actions`:

| column | type |
| --- | --- |
| action_id | INTEGER |
| content_id | INTEGER |
| reviewer_id | INTEGER |
| decision | VARCHAR (`'remove'` or `'no_action'`) |
| policy | VARCHAR |
| action_ts | TIMESTAMP |

This is the classic "aggregate by two keys" shape: Trust & Safety wants moderation volume and removal rate broken out by **both** `policy` and day, not just one or the other.

Write a query that returns, for every `(policy, day)` combination that has at least one action:
- `policy`
- `action_date`
- `total_actions` — count of actions that day under that policy
- `removed_count` — count with `decision = 'remove'`, via `SUM(IF(decision = 'remove', 1, 0))`
- `removal_rate` — `removed_count / total_actions`, rounded to 2 decimal places

Sort by `policy` ascending, then `action_date` ascending.

## Traps to avoid
- `GROUP BY policy, DATE(action_ts)` — grouping by only one of the two keys collapses rows that should stay separate (e.g. combining all policies' actions on a given day, or combining a policy's actions across every day).
- `removal_rate` is a ratio of counts for that specific `(policy, day)` bucket — don't accidentally compute it against a grand total.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: aggregate by group, two keys (`policy` + day) + pivot outcomes into columns + rates.
