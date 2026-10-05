# SQL: Appeal Overturn Rate
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB. A `SAFE_DIVIDE` helper is pre-registered so divide-by-zero returns `NULL` instead of erroring, exactly like BigQuery.

Two tables:

#### `moderation_actions`
| column | type |
| --- | --- |
| action_id | INTEGER |
| content_id | INTEGER |
| reviewer_id | INTEGER |
| decision | VARCHAR (`'remove'` or `'no_action'`) |
| policy | VARCHAR |
| action_ts | TIMESTAMP |

#### `appeals`
| column | type |
| --- | --- |
| appeal_id | INTEGER |
| action_id | INTEGER |
| appeal_ts | TIMESTAMP |
| outcome | VARCHAR (`'upheld'` or `'overturned'`) |

When a creator appeals a moderation decision, the appeal is either `'upheld'` (the original decision stands) or `'overturned'` (the reviewer got it wrong). Trust & Safety tracks **overturn rate** per policy to see which policies produce the shakiest decisions.

Write a query that returns, for each `policy` that has at least one appeal:
- `policy`
- `total_appeals` — count of appeals against actions under that policy
- `overturned_count` — count of those appeals with `outcome = 'overturned'`
- `overturn_rate` — `overturned_count / total_appeals`, rounded to 2 decimal places

Sort by `policy` ascending.

## Traps to avoid
- Actions with zero appeals shouldn't appear at all — this is an inner join between `appeals` and `moderation_actions` via `action_id`, not a `LEFT JOIN`.
- `overturn_rate` is a ratio of counts; use a safe division so a policy can't crash the query (it won't in this schema since every included policy has ≥1 appeal, but get in the habit anyway).
- Don't confuse `action_id` (joins to `moderation_actions`) with `content_id` — an appeal is against a specific *decision*, not directly against a piece of content.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: join events to entities + pivot outcomes into columns. Metric: appeal overturn rate by policy.
