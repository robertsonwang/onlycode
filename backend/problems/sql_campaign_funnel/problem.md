# SQL: Ad Campaign Funnel Rates
Difficulty: Medium

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB; a `SAFE_DIVIDE` helper is pre-registered so divide-by-zero returns `NULL` instead of erroring, exactly like BigQuery. Avoid Postgres-only syntax like `::` casts or `FILTER (WHERE ...)`.

You work on an ads analytics team. Every campaign generates `impressions`, some of those impressions turn into `clicks`, and some clicks turn into `signups`. You're given three tables:

#### `impressions`
| column | type |
| --- | --- |
| impression_id | INTEGER |
| campaign_id | INTEGER |
| user_id | INTEGER |

#### `clicks`
| column | type |
| --- | --- |
| click_id | INTEGER |
| campaign_id | INTEGER |
| user_id | INTEGER |

#### `signups`
| column | type |
| --- | --- |
| signup_id | INTEGER |
| campaign_id | INTEGER |
| user_id | INTEGER |

Write a SQL query that, for every campaign that appears in `impressions`, reports:

- `campaign_id`
- `total_impressions` — count of rows in `impressions` for that campaign
- `total_clicks` — count of *distinct users* in `clicks` for that campaign
- `total_signups` — count of *distinct users* in `signups` for that campaign
- `click_through_rate` — `total_clicks / total_impressions * 100`, rounded to 2 decimal places
- `signup_rate` — `total_signups / total_clicks * 100`, rounded to 2 decimal places, or `0.0` if `total_clicks` is 0

Return one row per campaign, columns in the order listed above, sorted by `campaign_id` ascending.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- Use `LEFT JOIN` so campaigns with zero clicks or zero signups still show up with `0` counts/rates.
- Use `SAFE_DIVIDE` instead of a bare `/` so a campaign with zero clicks doesn't error out; `IFNULL`/`COALESCE` the result to `0` before multiplying by 100.
- The expected output is a list of rows, where each row is itself a list of column values in `SELECT` order.

## Example
`impressions` has 3 rows for `campaign_id = 1`, `clicks` has 2 distinct users for `campaign_id = 1`, `signups` has 1 distinct user for `campaign_id = 1`.

Expected row: `[1, 3, 2, 1, 66.67, 50.0]`
