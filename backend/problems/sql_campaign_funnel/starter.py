from datetime import date, datetime
from decimal import Decimal
from typing import Any

import duckdb


# ---------------------------------------------------------------------------
# Test harness: builds an in-memory DuckDB (GoogleSQL-compatible) db from the
# input tables and registers a couple of BigQuery-only helper functions that
# DuckDB doesn't ship natively (DO NOT MODIFY)
# ---------------------------------------------------------------------------
def _build_db(tables: dict[str, Any]) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE MACRO safe_divide(a, b) AS CASE WHEN b = 0 THEN NULL ELSE a::DOUBLE / b::DOUBLE END")
    conn.execute("CREATE MACRO timestamp_diff(ts1, ts2, unit) AS date_diff(unit, ts2, ts1)")
    conn.execute("CREATE MACRO regexp_contains(s, pattern) AS regexp_matches(s, pattern)")
    for table_name, spec in tables.items():
        conn.execute(f"CREATE TABLE {table_name} ({', '.join(spec['columns'])})")
        rows = spec["rows"]
        if rows:
            placeholders = ", ".join(["?"] * len(rows[0]))
            conn.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", rows)
    return conn


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Decimal):
        return float(value)
    return value


def run_query(tables: dict[str, Any]) -> list[list]:
    conn = _build_db(tables)
    try:
        rows = conn.execute(QUERY).fetchall()
        return [[_serialize(v) for v in row] for row in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The editor only shows a raw SQL box; your query is spliced in here at submit time.
# ---------------------------------------------------------------------------
QUERY = "__SQL_QUERY_PLACEHOLDER__"


def solve(input_data: dict[str, Any]) -> list[list]:
    return run_query(input_data["tables"])
