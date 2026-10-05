# SQL: Top Product by Month
Difficulty: Hard

Dialect: **Google BigQuery (GoogleSQL)**. This sandbox runs your query against DuckDB, which supports the window functions this problem needs natively (`RANK`, `ROW_NUMBER`, `QUALIFY`). Avoid Postgres-only syntax like `::` casts or `FILTER (WHERE ...)`.

You're a data analyst for an online storefront. You have one table, `orders`:

#### `orders`
| column | type |
| --- | --- |
| order_id | INTEGER |
| product_name | TEXT |
| order_month | TEXT (e.g. `"2024-06"`) |
| quantity | INTEGER |

Write a SQL query that finds the best-selling product (by total quantity sold) for each `order_month`.

Return one row per month with columns, in this order:

- `order_month`
- `product_name`
- `total_quantity` — sum of `quantity` for that product in that month

Rules:
- If two or more products tie for the highest total quantity in a month, pick the one that is first alphabetically by `product_name`.
- Only include a month if it has at least one order.
- Sort the final output by `order_month` ascending.

## Notes
- The editor only takes a raw SQL query. It's run against a throwaway DuckDB database built from the test's table data, and the result rows are compared against the expected output.
- A window function such as `RANK()` or `ROW_NUMBER()` partitioned by `order_month` and ordered by summed quantity (with `product_name` as a tiebreaker), filtered with `QUALIFY`, is the cleanest way to solve this.
- The expected output is a list of rows, where each row is itself a list of column values in `SELECT` order.

## Example
In `2024-06`, "Widget" sells 12 units total and "Gadget" sells 9 units total.

Expected row for that month: `["2024-06", "Widget", 12]`
