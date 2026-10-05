# SQL: Report Spikes
Difficulty: Hard

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB, which supports the window functions this problem needs natively (`LAG`, `AVG ... OVER`, `ROWS BETWEEN`). A `SAFE_DIVIDE` helper is pre-registered so a day with no trailing history doesn't error.

One table, `reports`:

| column | type |
| --- | --- |
| report_id | INTEGER |
| content_id | INTEGER |
| reporter_id | INTEGER |
| reason | VARCHAR |
| reported_ts | TIMESTAMP |

Trust & Safety wants to catch report **spikes** — days where volume jumps well above the recent baseline, which often signals a coordinated abuse campaign or a viral piece of violating content.

Write a query that, for each day with at least one report, returns:
- `report_date`
- `report_count` — number of reports that day
- `day_over_day_change` — `report_count` minus the previous day's `report_count` (`NULL` for the first day in the data)
- `spike_ratio` — `report_count` divided by the **trailing 7-day average** `report_count` (the 7 days strictly before this one, not including today), rounded to 2 decimal places. Use a safe division so a day with no prior history doesn't error.

Sort by `report_date` ascending.

## Traps to avoid
- The trailing average must exclude the current day — don't let today's count leak into its own baseline.
- "Trailing 7 days" means a window frame, not a self-join or a correlated subquery recomputing the same thing per row.
- Every test in this problem has at least one report on every day in range, so you don't need to invent missing days — just aggregate what's there, bucketed by `DATE(reported_ts)`.
- `day_over_day_change` is a difference (`LAG`), not a ratio — don't conflate it with `spike_ratio`.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: time windows/bucketing + rank/window functions. Metric: report spikes (day-over-day change and ratio to trailing average).
