# SQL: Share of Reports by Reason
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB. A `SAFE_DIVIDE` helper is pre-registered so divide-by-zero returns `NULL` instead of erroring.

One table, `reports`:

| column | type |
| --- | --- |
| report_id | INTEGER |
| content_id | INTEGER |
| reporter_id | INTEGER |
| reason | VARCHAR |
| reported_ts | TIMESTAMP |

Trust & Safety wants a breakdown of which report `reason`s make up the queue, along with a running total so they know how many of the top reasons they'd need to tackle to cover, say, 80% of volume.

Write a query that returns, for each distinct `reason`:
- `reason`
- `report_count` — number of reports with that reason
- `share_pct` — `report_count` as a percentage of all reports, rounded to 2 decimal places
- `cumulative_count` — the running total of `report_count`, accumulated in the same order as the final output (largest `report_count` first)

Sort by `report_count` descending, then `reason` ascending (and accumulate `cumulative_count` in that same order).

## Traps to avoid
- `share_pct` is `report_count / SUM(report_count) OVER ()` — a window function over the whole result, not a second pass over `reports`.
- `cumulative_count` must accumulate in the *same order* as the final sort (ties broken by `reason` ascending) — a plain `ORDER BY report_count DESC` without the tiebreaker would make the running total order-dependent and non-deterministic for tied reasons.
- Don't use `SUM(report_count)` without a window frame bound if you want a strictly cumulative value — make sure the frame is `ORDER BY ... ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` (or the window's default framing, which already means this for an `ORDER BY`-only window).

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: aggregate by group + pivot/share-of-total with a running total (`SUM(COUNT(*)) OVER (...)`).
