# SQL: Trust & Safety Review Backlog
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB, which is close enough to GoogleSQL for every function used here (`COUNT(DISTINCT ...)`, `NOT EXISTS`, window functions, `QUALIFY`, `COUNTIF`, `SAFE_DIVIDE`). Don't rely on Postgres-only syntax like `::` casts or `FILTER (WHERE ...)` — it won't be accepted in a real BigQuery interview either.

You're a data analyst on a Trust & Safety team. Two tables:

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
| decision | VARCHAR (`'remove'` or `'no_action'`) |
| policy | VARCHAR |
| action_ts | TIMESTAMP |

Content can be reported multiple times by different people before anyone reviews it, and some reported content is never reviewed at all — that's the **review backlog**.

Write a query that returns one row per piece of content that has at least one report in `reports` but **zero** rows in `moderation_actions` (i.e. it has never been actioned, regardless of when it was reported).

Return columns, in this order:
- `content_id`
- `report_count` — the number of **distinct** reports filed against that content

Rules:
- Only include content with zero moderation actions ever recorded against it. Content with any action row — even an old one, even a `'no_action'` decision — is **not** backlog; it has been reviewed.
- Sort by `report_count` descending, then `content_id` ascending.
- If nothing is in the backlog, return no rows.

## Traps to avoid
- An `INNER JOIN` between `reports` and `moderation_actions` silently drops every backlog row — you need a `LEFT JOIN ... WHERE ... IS NULL` or `NOT EXISTS` anti-join instead.
- Counting `report_id` rows is fine here since each row is a distinct report, but get in the habit of asking "are we counting reports or unique content?" — this schema repeats that trap elsewhere (e.g. with `COUNT(DISTINCT content_id)`).
- Don't filter on any `moderation_actions` column in a `WHERE` clause after a `LEFT JOIN` — that turns it back into an inner join and breaks the anti-join.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: join events to entities / anti-join. Metric: review backlog.
