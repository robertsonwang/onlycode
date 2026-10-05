# SQL: Authors Without Removals
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB.

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
| content_id | INTEGER (nullable — a small number of legacy rows have no linked content) |
| reviewer_id | INTEGER |
| decision | VARCHAR (`'remove'` or `'no_action'`) |
| policy | VARCHAR |
| action_ts | TIMESTAMP |

Write a query that returns the `author_id` of every author who has published content, **none of which has ever been removed**. Return one column, `author_id`, sorted ascending, one row per qualifying author.

## Traps to avoid
- This is the classic `NOT IN` vs `NOT EXISTS` gotcha. `moderation_actions.content_id` can be `NULL` (a handful of legacy rows aren't linked to any content). If you write:
  ```sql
  SELECT DISTINCT author_id FROM content
  WHERE content_id NOT IN (SELECT content_id FROM moderation_actions WHERE decision = 'remove')
  ```
  and that subquery's result set contains even one `NULL`, **the entire `NOT IN` comparison silently evaluates to unknown for every row**, and your query returns an empty result — even for authors who obviously qualify. SQL's three-valued logic means `x NOT IN (3, NULL)` is `NULL` (not `TRUE`) unless `x` happens to equal `3`.
- Use `NOT EXISTS` (or a `LEFT JOIN ... WHERE ... IS NULL`) instead — both are immune to `NULL`s in the subquery because they test row existence, not list membership.
- If you must use `NOT IN`, you'd need to explicitly filter out `NULL`s from the subquery first (`WHERE content_id IS NOT NULL`) — but `NOT EXISTS` sidesteps the whole problem.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: join events to entities (anti-join). Trap: `NOT IN` with a subquery that can return `NULL`s.
