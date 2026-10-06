# SQL: Odd & Even Flag Scores
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB. BigQuery has no `%` modulo operator and no `FILTER (WHERE ...)` clause (both are Postgres-only) — use the `MOD(x, y)` function and conditional aggregation (`SUM(IF(...))` or `CASE WHEN`) instead.

One table, `classifier_flags`:

| column | type |
| --- | --- |
| content_id | INTEGER |
| model_version | VARCHAR |
| score | DOUBLE |
| flagged_ts | TIMESTAMP |

The same content can be flagged multiple times over its lifetime. For an audit, Trust & Safety wants to split each content item's flag scores into two buckets based on **sequence position** (1st, 2nd, 3rd flag... ordered by `flagged_ts`), not by any property of the flag itself: odd-numbered flags vs. even-numbered flags.

Write a query that returns, for each `content_id`:
- `content_id`
- `odd_sum` — sum of `score` from flags numbered 1st, 3rd, 5th, ... (ordered by `flagged_ts` ascending)
- `even_sum` — sum of `score` from flags numbered 2nd, 4th, 6th, ...

If a content item has only one flag, `even_sum` is `0`, not `NULL`.

Sort by `content_id` ascending.

## Traps to avoid
- This is a two-step problem: first number the flags per content with `ROW_NUMBER() OVER (PARTITION BY content_id ORDER BY flagged_ts ASC)`, *then* aggregate by parity of that row number. You can't determine "odd or even" from any column in the raw data — it only exists after ordering.
- Don't reach for `rn % 2` — BigQuery doesn't have a `%` operator. Use `MOD(rn, 2)`.
- Don't reach for `SUM(score) FILTER (WHERE MOD(rn, 2) = 1)` — `FILTER (WHERE ...)` is Postgres/standard-SQL syntax that BigQuery doesn't support. Use `SUM(IF(MOD(rn, 2) = 1, score, 0))` instead.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: rank/pick a row within a group, then pivot a computed sequence property (parity) into columns.
