# SQL: Removals by Content Type
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB, which supports `IF(condition, true_value, false_value)` the same way BigQuery does.

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

Content can be re-reviewed, so a single `content_id` may have multiple `moderation_actions` rows — only the most recent one (by `action_ts`, breaking ties by the higher `action_id`) reflects the current decision.

Write a query that returns, for each `content_type` that has at least one actioned item:
- `content_type`
- `total_actioned` — count of distinct content items of that type that have a final decision
- `removed_count` — how many of those have a final decision of `'remove'`, computed with `SUM(IF(decision = 'remove', 1, 0))`
- `no_action_count` — how many have a final decision of `'no_action'`, computed with `SUM(IF(decision = 'no_action', 1, 0))`

Sort by `content_type` ascending.

## Traps to avoid
- Use each content item's **final** decision (most recent `action_ts`, `action_id` as tiebreaker) — counting every action row double-counts re-reviewed content and can inflate both `removed_count` and `no_action_count` for the same item.
- Content with zero actions shouldn't appear in `total_actioned` at all — that's a backlog question, not this one.
- `SUM(IF(...))` is one of several equivalent ways to pivot a condition into a count (`COUNTIF` is another, used elsewhere) — get comfortable with both spellings since interviewers may ask for either.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: rank/pick a row within a group (final decision) + pivot outcomes into columns with `SUM(IF(...))`.
