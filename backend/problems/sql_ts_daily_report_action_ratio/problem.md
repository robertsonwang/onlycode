# SQL: Daily Report-to-Action Ratio
Difficulty: Hard

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB, which supports `FULL OUTER JOIN` the same way BigQuery does. A `SAFE_DIVIDE` helper is pre-registered.

Two tables:

#### `reports`
| column | type |
| --- | --- |
| report_id | INTEGER |
| content_id | INTEGER |
| reporter_id | INTEGER |
| reason | VARCHAR |
| reported_ts | TIMESTAMP |

#### `moderation_actions`
| column | type |
| --- | --- |
| action_id | INTEGER |
| content_id | INTEGER |
| reviewer_id | INTEGER |
| decision | VARCHAR |
| policy | VARCHAR |
| action_ts | TIMESTAMP |

Trust & Safety wants a daily operational view comparing how many reports came in versus how many moderation actions the team completed — two independent counts, bucketed by day, that don't necessarily line up 1:1 (a day's actions might resolve reports filed days earlier, or the queue might be quiet on reports but busy clearing a backlog).

Write a query that returns one row for **every date that appears in either table** (a date with reports but no actions that day, or vice versa, must still appear):
- `report_date`
- `report_count` — reports filed that day (`0` if none)
- `action_count` — actions taken that day (`0` if none)
- `action_to_report_ratio` — `action_count / report_count`, rounded to 2 decimal places. Use a safe division so a day with actions but zero reports doesn't error.

Sort by `report_date` ascending.

## Traps to avoid
- An `INNER JOIN` (or a plain `LEFT JOIN` in either direction) between the two daily rollups would drop dates that only appear on one side. Use a `FULL OUTER JOIN` on the date, and `COALESCE` both count columns to `0` for the side that's missing.
- The join key is the **date**, not `content_id` — a report and an action on the same day aren't necessarily related to each other at all; this is a daily volume comparison, not a per-report lookup.
- Build each side's daily counts independently (one `GROUP BY DATE(...)` per table) before joining — don't try to join the raw `reports` and `moderation_actions` tables directly on date, which would cross-multiply every report against every action that day.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: time windows/bucketing + rates, using `FULL OUTER JOIN` to reconcile two independent daily rollups.
