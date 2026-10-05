# SQL: Reviewer Team Rollup
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB. A `SAFE_DIVIDE` helper is pre-registered.

Two tables:

#### `reviewers`
| column | type |
| --- | --- |
| reviewer_id | INTEGER |
| name | VARCHAR |
| lead_id | INTEGER (self-referencing — the `reviewer_id` of this reviewer's team lead, `NULL` for reviewers with no lead on record) |

#### `moderation_actions`
| column | type |
| --- | --- |
| action_id | INTEGER |
| content_id | INTEGER |
| reviewer_id | INTEGER |
| decision | VARCHAR |
| policy | VARCHAR |
| action_ts | TIMESTAMP |

`reviewers` is a classic self-referencing table: every reviewer optionally has a `lead_id` pointing to another row in the same table.

Write a query that, for every reviewer who **is** someone else's lead (i.e. has at least one direct report), returns:
- `lead_name` — the lead's `name`
- `team_size` — number of distinct direct reports
- `total_actions` — total moderation actions taken by those direct reports (not counting any actions the lead personally took)
- `removal_rate` — fraction of the team's actions that were `'remove'`, rounded to 2 decimal places

Sort by `lead_name` ascending.

## Traps to avoid
- This needs a **self-join**: `reviewers` joined to `reviewers` again on `lead_id = reviewer_id`.
- Only count actions taken by *direct reports*, not the lead's own actions, and not reports-of-reports (this schema is only one level deep in every test case here, but don't accidentally sum the lead's personal action rows into their own team's total).
- A reviewer with no direct reports shouldn't appear in the output at all.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: self-join + aggregate by group + pivot outcomes into columns.
