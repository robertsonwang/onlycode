# SQL: Top Classifier Flag Per Content
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB. One real syntax difference: BigQuery's idiom for "pick one value from the row with the highest/lowest something" is `ARRAY_AGG(x ORDER BY y DESC LIMIT 1)[OFFSET(0)]` — an aggregate with an `ORDER BY ... LIMIT 1` *inside* `ARRAY_AGG`, then pulled out with zero-indexed `OFFSET(0)`. DuckDB's `ARRAY_AGG` doesn't support a `LIMIT` clause inside the aggregate, and its arrays are **1-indexed**, not 0-indexed. In this sandbox, write `ARRAY_AGG(x ORDER BY y DESC)[1]` instead — collect everything in sorted order, then take the first (1-indexed) element.

One table, `classifier_flags`:

| column | type |
| --- | --- |
| content_id | INTEGER |
| model_version | VARCHAR |
| score | DOUBLE |
| flagged_ts | TIMESTAMP |

The same piece of content can be flagged multiple times (by different model versions, or the same one re-scoring it later). Write a query that returns, for each `content_id`, the flag with the **highest `score`**:
- `content_id`
- `top_model_version` — the `model_version` of the highest-scoring flag
- `top_score` — that flag's `score`

If two flags on the same content tie for the highest score, prefer the one with the **earlier** `flagged_ts`.

Sort by `content_id` ascending.

## Traps to avoid
- `GROUP BY content_id` with a bare `MAX(score)` only gets you the score, not the `model_version` that produced it — you need to pick the whole "winning" row, not aggregate each column independently (a naive `MAX(score)` alongside `MAX(model_version)` could mix fields from two different rows if they don't share a max).
- Don't forget the tiebreaker (`flagged_ts` ascending) — without it, which row "wins" a tie is undefined and non-deterministic.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: rank/pick a row within a group, using `ARRAY_AGG(... ORDER BY ...)` instead of `ROW_NUMBER()`/`QUALIFY`.
