# SQL: Reported vs. Flagged Authors
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB, which supports `UNION ALL`, `EXCEPT DISTINCT`, and `INTERSECT DISTINCT` the same way BigQuery does. BigQuery requires the explicit `DISTINCT`/`ALL` keyword on every set operator — a bare `UNION`/`EXCEPT` is a syntax error there, unlike Postgres.

Three tables:

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

#### `classifier_flags`
| column | type |
| --- | --- |
| content_id | INTEGER |
| model_version | VARCHAR |
| score | DOUBLE |
| flagged_ts | TIMESTAMP |

Trust & Safety wants to compare two detection surfaces at the **author** level: human reports vs. the automated classifier. Some authors only ever get reported, some only ever get flagged by the model, and some show up on both.

Write a query that returns one row per author with columns:
- `segment` — one of `'reported_only'`, `'flagged_only'`, or `'both'`
- `author_id`

An author belongs in a segment based on whether **any** of their content has ever been reported and/or **any** of their content has ever been flagged — it doesn't need to be the *same piece of content* on both sides. An author with one post reported and a completely different post flagged still belongs in `'both'`.

Sort by `segment` ascending, then `author_id` ascending.

## Traps to avoid
- This is an author-level comparison, not a content-level one — join `reports`/`classifier_flags` to `content` first to get `author_id`, and de-duplicate per author (`DISTINCT`) before applying set operators. Matching only when the exact same `content_id` is both reported and flagged would wrongly exclude authors who clearly belong in `'both'`.
- Use `EXCEPT DISTINCT` / `INTERSECT DISTINCT` (BigQuery requires the keyword), not a `LEFT JOIN ... WHERE ... IS NULL` pattern — the point of this problem is practicing set operators directly.
- An author must land in exactly one segment — don't let someone in `'both'` also leak into `'reported_only'`.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: joins and sets (`UNION ALL`, `EXCEPT DISTINCT`, `INTERSECT DISTINCT`).
