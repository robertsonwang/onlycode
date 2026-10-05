# SQL: Repeat Offenders
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB. Don't reach for Postgres-only syntax like `::` casts, `FILTER (WHERE ...)`, or bare `INTERVAL` arithmetic — none of that is valid BigQuery either.

Two tables:

#### `content`
| column | type |
| --- | --- |
| content_id | INTEGER |
| author_id | INTEGER |
| content_type | VARCHAR |
| created_ts | TIMESTAMP |

#### `moderation_actions`
| column | type |
| --- | --- |
| action_id | INTEGER |
| content_id | INTEGER |
| reviewer_id | INTEGER |
| decision | VARCHAR (`'remove'` or `'no_action'`) |
| policy | VARCHAR |
| action_ts | TIMESTAMP |

Trust & Safety wants to flag **repeat offenders**: authors who have had several distinct pieces of content removed in a single audit window.

Write a query that returns every `author_id` with **3 or more distinct pieces of content removed** (`decision = 'remove'`) where `action_ts` falls in March 2024 — that is, `action_ts >= '2024-03-01'` and `action_ts < '2024-04-01'` (half-open range; don't use `BETWEEN` with an inclusive upper bound, since that would wrongly include the first instant of April).

Return columns, in this order:
- `author_id`
- `removal_count` — the number of **distinct** `content_id`s removed by that author in the window

Sort by `removal_count` descending, then `author_id` ascending.

## Traps to avoid
- If a piece of content is re-reviewed and re-confirmed as `'remove'` (two action rows, same `content_id`), that's still only **one** removed item — count `DISTINCT content_id`, not action rows.
- A removal that happened in February doesn't count, even if the same author has removals in March — filter on `action_ts`, not on the author's overall history.
- Use a half-open range (`>= start AND < end`) for the date filter, not `BETWEEN`, to avoid accidentally including the first moment of the next month.
- `HAVING` filters on the aggregated count; don't try to do this with `WHERE`.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: aggregate by group + join events to entities + time window. Metric: repeat offenders (`HAVING COUNT(DISTINCT ...) >= 3`).
