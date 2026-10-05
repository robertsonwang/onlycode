# SQL: Time-to-Action SLA
Difficulty: Hard

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB. A `SAFE_DIVIDE` helper and a `TIMESTAMP_DIFF(ts1, ts2, unit)` helper are pre-registered to match BigQuery's behavior (`TIMESTAMP_DIFF` isn't built into DuckDB natively). One real difference: BigQuery computes percentiles with `PERCENTILE_CONT(x, 0.5) OVER ()`; this DuckDB sandbox doesn't implement the analytic-function form, so use the standard-SQL equivalent `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)` instead — same result, different call syntax.

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

For every piece of content that was reported **and has since been actioned**, compute its time-to-action: the number of hours between its *earliest* report and the *first action that happened at or after that report* (an action that happened before the content was ever reported doesn't count — it wasn't a response to this report).

Using the time-to-action value for every qualifying content item, return a single row with:
- `median_hours` — the median time-to-action, rounded to 2 decimal places
- `p90_hours` — the 90th percentile time-to-action, rounded to 2 decimal places
- `pct_within_24h` — the percentage of qualifying content items actioned within 24 hours, rounded to 2 decimal places

Report the **median and p90**, not the mean — a few very slow re-reviews shouldn't be allowed to hide in an average.

## Traps to avoid
- Joining actions to reports without requiring `action_ts >= reported_ts` lets an unrelated action from before the report count as the "response," understating time-to-action.
- Use the *first* qualifying action (`MIN(action_ts)` among actions at/after the report), not just any action or the most recent one.
- Content that was reported but never actioned at all shouldn't contribute a row — it has no time-to-action yet (that's a backlog question, not an SLA question).
- Report median/p90, never the mean — a handful of slow outliers skew averages badly.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: join events to entities + time windows/bucketing + rates/distributions. Metric: time-to-action SLA (median, p90, % within 24h).
- If no content qualifies, the single output row will have `NULL` in all three columns — that's expected, not an error.
