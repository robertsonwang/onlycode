# SQL: Top Reported Content Per Day
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB, which supports `QUALIFY` and `RANK()` natively.

One table, `reports`:

| column | type |
| --- | --- |
| report_id | INTEGER |
| content_id | INTEGER |
| reporter_id | INTEGER |
| reason | VARCHAR |
| reported_ts | TIMESTAMP |

For each day, find the piece of content that received the **most** reports that day.

Write a query that returns:
- `report_date`
- `content_id`
- `report_count`

**If two or more pieces of content tie for the most reports on a given day, keep all of them** — don't arbitrarily pick one. Sort by `report_date` ascending, then `content_id` ascending.

## Traps to avoid
- `ORDER BY report_count DESC LIMIT 1` per day silently drops ties, keeping only one arbitrary winner. Use `RANK()` (not `ROW_NUMBER()`, which would also break ties arbitrarily) filtered with `QUALIFY RANK() OVER (...) = 1` so every tied content item is kept.
- Make sure the window is partitioned by day, not computed globally — "most reported content" is a per-day question.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: time bucketing + rank/pick rows within a group, keeping ties (`RANK` + `QUALIFY`, not `LIMIT 1`).
