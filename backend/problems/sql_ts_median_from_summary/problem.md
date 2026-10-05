# SQL: Median From a Summary Table
Difficulty: Hard

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB. Avoid Postgres-only syntax like `::` casts or `FILTER (WHERE ...)`.

One table, `reporter_summary` — a **pre-aggregated frequency table**, not a row-per-user log:

| column | type |
| --- | --- |
| reports_filed | INTEGER |
| num_users | INTEGER |

Each row means "`num_users` distinct users each filed exactly `reports_filed` reports." For example, a row `(2, 30)` means 30 users each filed exactly 2 reports. **You do not have row-level user data and cannot get it** — only this summary. Expanding it into one row per user (e.g. with `UNNEST(GENERATE_ARRAY(1, num_users))`) would work but is wasteful once `num_users` gets large; this problem wants the running-total approach instead.

Write a query that returns a single row with one column, `median_reports_filed`: the median number of reports filed per user, computed **without expanding the table**, rounded to 2 decimal places.

To do this without expanding rows:
1. Compute `total_users = SUM(num_users)` across the whole table.
2. Compute a running total of `num_users` ordered by `reports_filed` ascending — this tells you, for each bucket, the range of "user ranks" it covers (e.g. if the running total is 50 after the first bucket, that bucket covers ranks 1 through 50).
3. The median position(s) are rank `⌊(total_users + 1) / 2⌋` and rank `⌈(total_users + 1) / 2⌉` (the same rank, twice, when `total_users` is odd). Find the bucket(s) whose range contains each of these ranks, and average their `reports_filed` values.

## Traps to avoid
- If `total_users` is even, the two median ranks can fall in **different** buckets — don't assume the median is always a single bucket's value.
- A bucket's range is `(running_total - num_users, running_total]` (1-indexed) — off-by-one errors here are easy. Double check the boundary with a small hand-traced example before trusting the query on the full table.
- This is a different situation than ranking raw rows with `ROW_NUMBER()` — here, one row represents many users at once.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: rates/distributions from a pre-aggregated table. Metric: median without row expansion.
