# SQL: Action Rate by Policy
Difficulty: Hard

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB, which supports the GoogleSQL constructs this problem needs natively: `QUALIFY`, `COUNTIF`, window functions, and `ROUND`. A `SAFE_DIVIDE` helper is pre-registered so divide-by-zero behaves exactly like BigQuery (returns `NULL`, never an error). Don't reach for Postgres-only syntax like `::` casts, `FILTER (WHERE ...)`, or bare `INTERVAL` arithmetic — none of that is valid BigQuery either.

One table, `moderation_actions`:

| column | type |
| --- | --- |
| action_id | INTEGER |
| content_id | INTEGER |
| reviewer_id | INTEGER |
| decision | VARCHAR (`'remove'` or `'no_action'`) |
| policy | VARCHAR |
| action_ts | TIMESTAMP |

Content is sometimes re-reviewed — a reviewer's first pass might say `'no_action'`, then a later re-review under appeal or escalation flips it to `'remove'` (or vice versa), sometimes even under a different `policy`. Only the **final** decision for a piece of content should count toward your metrics; a content item's final policy is whichever policy its most recent action was taken under. If two actions for the same content share the exact same `action_ts`, the one with the higher `action_id` is the later one.

Write a query that computes, for each `policy` (based on each content item's final action only):
- `reviewed_count` — number of distinct content items whose final action falls under this policy
- `removed_count` — number of those whose final decision is `'remove'`
- `action_rate` — `removed_count / reviewed_count`, rounded to 2 decimal places. Use a safe division so this never errors when `reviewed_count` is 0.

Return columns, in this order: `policy`, `reviewed_count`, `removed_count`, `action_rate`.

Sort by `policy` ascending.

## Traps to avoid
- Grouping and counting over **every** action row instead of each content item's final decision will overcount re-reviewed content and can attribute it to the wrong policy.
- `action_rate` is a ratio of counts (sums), not an average of per-row flags — don't `AVG()` a boolean.
- A plain `/` divides by zero and errors out; use a safe division.
- Picking "the" final action with `ORDER BY action_ts DESC LIMIT 1` per group silently breaks when timestamps tie — use `ROW_NUMBER()` with an explicit tiebreaker (`action_id DESC`) and `QUALIFY`, so ties are resolved deterministically rather than dropped or duplicated.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: rank/pick a row within a group (`QUALIFY` + `ROW_NUMBER`) combined with pivoting outcomes into columns (`COUNTIF`). Metric: action rate using each item's final decision.
