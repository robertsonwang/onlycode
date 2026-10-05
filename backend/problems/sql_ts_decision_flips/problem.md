# SQL: Decision Flips
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB, which supports `LAG` natively.

One table, `moderation_actions`:

| column | type |
| --- | --- |
| action_id | INTEGER |
| content_id | INTEGER |
| reviewer_id | INTEGER |
| decision | VARCHAR (`'remove'` or `'no_action'`) |
| policy | VARCHAR |
| action_ts | TIMESTAMP |

When content is re-reviewed (appealed, escalated, or just double-checked), the new decision sometimes disagrees with the previous one — a **flip**. Trust & Safety wants to audit every flip, in either direction.

Write a query that returns one row for every action that **changed the decision from the immediately preceding action on the same content**:
- `content_id`
- `action_id` — the id of the action that caused the flip (the *new* decision, not the old one)
- `prev_decision`
- `new_decision`
- `action_ts`

A content item's very first action is never a "flip" (there's nothing before it to disagree with). Two consecutive actions with the *same* decision are not a flip either.

Sort by `content_id` ascending, then `action_ts` ascending.

## Traps to avoid
- This needs `LAG(decision) OVER (PARTITION BY content_id ORDER BY action_ts ASC, action_id ASC)` compared against the current row's `decision` — a self-join on "previous action" without a tiebreaker on `action_id` would be ambiguous if two actions share a timestamp.
- The first action per content has no previous decision (`LAG` returns `NULL`) — exclude those rows rather than treating `NULL != decision` as a flip (depending on your SQL engine, a `NULL` comparison is itself a trap: `NULL != 'remove'` is `NULL`, not `TRUE`, so it won't accidentally pass a naive `WHERE` filter — but don't rely on that; filter `prev_decision IS NOT NULL` explicitly so the intent is clear).

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: rank/pick a row within a group, using `LAG` to detect a sequential state change rather than a day-over-day aggregate change.
