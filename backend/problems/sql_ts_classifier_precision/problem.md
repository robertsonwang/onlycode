# SQL: Classifier Precision
Difficulty: Hard

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB, which supports `QUALIFY` and `COUNTIF` natively. A `SAFE_DIVIDE` helper is pre-registered.

Three tables:

#### `classifier_flags`
| column | type |
| --- | --- |
| content_id | INTEGER |
| model_version | VARCHAR |
| score | DOUBLE |
| flagged_ts | TIMESTAMP |

#### `moderation_actions`
| column | type |
| --- | --- |
| action_id | INTEGER |
| content_id | INTEGER |
| reviewer_id | INTEGER |
| decision | VARCHAR |
| policy | VARCHAR |
| action_ts | TIMESTAMP |

A flag is **reviewed** if there's at least one moderation action for its `content_id` at or after the `flagged_ts` (an action from before the flag existed isn't a review of it). A reviewed flag is **confirmed violating** if that content's *final* decision is `'remove'`.

**Classifier precision** = confirmed-violating flags ÷ reviewed flags. Flags that were never reviewed are excluded from precision entirely — they're neither confirmed nor refuted, we simply don't know. Report their share separately; don't fold them into the denominator as if they were "not violating."

Write a query that returns, for each `model_version`:
- `model_version`
- `total_flags` — total flags raised by that model version
- `reviewed_flags` — how many of those were reviewed
- `confirmed_violating` — how many reviewed flags led to a final `'remove'` decision
- `precision` — `confirmed_violating / reviewed_flags`, rounded to 2 decimal places
- `unreviewed_share` — `(total_flags - reviewed_flags) / total_flags`, rounded to 2 decimal places

Sort by `model_version` ascending.

## Traps to avoid
- This schema can't tell you **recall** (violating content the classifier never flagged at all) — you'd need a labeled sample of unflagged content you don't have here. Don't try to compute it; precision and the unreviewed share are the only metrics this data supports.
- An action that happened *before* the flag doesn't count as a review of that flag — always compare `action_ts >= flagged_ts`.
- Use each flagged content item's *final* decision if it was re-reviewed, not just any action row.
- Don't drop unreviewed flags from `total_flags` — they still count toward the denominator of `unreviewed_share`, just not toward `precision`.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Pattern: join events to entities + rank/pick a row within a group (final decision) + rates. Metric: classifier precision, with "spot the missing data" baked in (no recall).
