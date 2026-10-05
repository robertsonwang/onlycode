# SQL: Monthly Prevalence
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB. A `SAFE_DIVIDE` helper is pre-registered so divide-by-zero returns `NULL` instead of erroring, exactly like BigQuery. Avoid Postgres-only syntax like `::` casts or `FILTER (WHERE ...)`.

One table, `daily_views`:

| column | type |
| --- | --- |
| date | DATE |
| views | INTEGER |
| violating_views | INTEGER |

**`date` is a reserved keyword**, so you'll need to quote it wherever it's used as a column name. In the real BigQuery console you'd write `` `date` `` with backticks; this DuckDB sandbox uses standard SQL double-quoted identifiers instead — write `"date"`.

**Prevalence** is the fraction of views that were of violating content: `SUM(violating_views) / SUM(views)`. It's always a ratio of sums over some time period — never an average of each day's individual ratio.

Write a query that returns, for each calendar month present in `daily_views`:
- `month` — formatted as `'YYYY-MM'`
- `total_views` — `SUM(views)` for that month
- `total_violating_views` — `SUM(violating_views)` for that month
- `prevalence` — `total_violating_views / total_views`, rounded to 4 decimal places. Use a safe division so a month with zero total views returns `NULL` instead of erroring.

Sort by `month` ascending.

## Traps to avoid
- Don't average each day's `violating_views / views` ratio — a handful of low-traffic days with high ratios would skew the monthly number. Sum first, then divide.
- Days with `views = 0` must not cause a divide-by-zero error if you compute a per-day ratio anywhere; more importantly, a whole month with zero total views should produce `NULL`, not an error and not `0`.
- `date` needs quoting every time it's referenced, not just in `SELECT`.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: aggregate by group + rates/distributions. Metric: prevalence (ratio of sums).
