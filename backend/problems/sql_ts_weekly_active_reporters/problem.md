# SQL: Weekly Active Reporters
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB. `DATE_TRUNC(d, WEEK)` in BigQuery takes the date first, the part second; DuckDB's native `date_trunc` takes the part first (`date_trunc('week', d)`) — the opposite order. Write it DuckDB's way here; just know the argument order flips if you paste this into the real BigQuery console.

One table, `reports`:

| column | type |
| --- | --- |
| report_id | INTEGER |
| content_id | INTEGER |
| reporter_id | INTEGER |
| reason | VARCHAR |
| reported_ts | TIMESTAMP |

Write a query that buckets reports into weeks (Monday-start) and returns, for each week that has at least one report:
- `week_start` — the Monday that starts that week, as a `DATE`
- `active_reporters` — count of **distinct** `reporter_id`s who filed a report that week
- `wow_change` — `active_reporters` minus the previous week's `active_reporters` (`NULL` for the first week present in the data)

Sort by `week_start` ascending.

## Traps to avoid
- `COUNT(reporter_id)` counts report rows, not people — the same reporter filing 5 reports in a week is still 1 active reporter. Use `COUNT(DISTINCT reporter_id)`.
- `date_trunc('week', ts)` returns a `TIMESTAMP`, not a `DATE` — cast it explicitly, or your output type won't match what's expected.
- `wow_change` compares consecutive **weeks present in the data**, not literally "7 days ago" — if a week has zero reports, it simply doesn't produce a row, and `LAG` skips straight to the previous row that exists, not the calendar-adjacent week. (None of the test cases here have a gap week, but keep this in mind: `LAG` operates on your result set's row order, not wall-clock time.)

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: time windows/bucketing (weekly, not daily/monthly) + `LAG` for week-over-week change.
