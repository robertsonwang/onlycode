# SQL: Most-Harassment-Reported Content Type
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB, which supports `QUALIFY` and `RANK()` natively.

Two tables:

#### `content`
| column | type |
| --- | --- |
| content_id | INTEGER |
| author_id | INTEGER |
| content_type | VARCHAR |
| created_ts | TIMESTAMP |

#### `reports`
| column | type |
| --- | --- |
| report_id | INTEGER |
| content_id | INTEGER |
| reporter_id | INTEGER |
| reason | VARCHAR |
| reported_ts | TIMESTAMP |

Trust & Safety wants to know which `content_type` (e.g. `'post'`, `'comment'`, `'video'`) attracts the most reports tagged with reason `'harassment'`, so they can decide where to invest in better detection tooling.

Write a query that returns the `content_type`(s) with the highest count of `'harassment'` reports:
- `content_type`
- `harassment_count`

**If two or more content types tie for the highest count, return all of them**, sorted by `content_type` ascending.

## Traps to avoid
- Filter `reason = 'harassment'` before counting — don't count all reports and then try to filter the aggregate.
- `ORDER BY harassment_count DESC LIMIT 1` silently drops a tie. Use `RANK()` filtered with `QUALIFY RANK() OVER (ORDER BY harassment_count DESC) = 1` so every tied content type survives.
- A `content_type` with zero `'harassment'` reports (even if it has other reports) shouldn't appear at all — the `WHERE reason = 'harassment'` filter needs to happen before the join's results are grouped, not after.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: join events to entities + rank with ties kept (`RANK` + `QUALIFY`, not `LIMIT 1`).
