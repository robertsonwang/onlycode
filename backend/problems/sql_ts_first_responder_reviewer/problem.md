# SQL: First Responder Reviewer
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB, which supports `QUALIFY` and `ROW_NUMBER()` natively.

One table, `moderation_actions`:

| column | type |
| --- | --- |
| action_id | INTEGER |
| content_id | INTEGER |
| reviewer_id | INTEGER |
| decision | VARCHAR |
| policy | VARCHAR |
| action_ts | TIMESTAMP |

Content is sometimes re-reviewed by a different reviewer later, but Trust & Safety wants to know who's actually triaging the queue — the reviewer who takes the **first** action on each piece of content, before anyone else touches it.

Write a query that returns, for each `reviewer_id`, how many times they were the **first responder** (took the earliest action, by `action_ts` then `action_id` as a tiebreaker, on a given `content_id`):
- `reviewer_id`
- `first_response_count`

Sort by `first_response_count` descending, then `reviewer_id` ascending.

## Traps to avoid
- This is a two-step problem: first pick the first action per `content_id` (`QUALIFY ROW_NUMBER() OVER (PARTITION BY content_id ORDER BY action_ts ASC, action_id ASC) = 1`), *then* aggregate by `reviewer_id` over that filtered set. Aggregating `moderation_actions` directly by `reviewer_id` without first isolating the first action per content would count every action a reviewer ever took, not just the ones where they triaged first.
- Don't confuse this with "most actions taken overall" — a reviewer who does lots of re-reviews but rarely gets there first should have a low `first_response_count`.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: rank/pick a row within a group, then aggregate by a different key than the one you ranked within.
