"""SQL safety guard for agent-generated queries.

Only read-only SELECT/WITH statements are allowed, and a LIMIT is added when
the generated query does not already include one.
"""

import sqlparse
from sqlparse.sql import TokenList


FORBIDDEN_KEYWORDS = {
    "ALTER",
    "ATTACH",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "INSERT",
    "PRAGMA",
    "REINDEX",
    "REPLACE",
    "TRUNCATE",
    "UPDATE",
    "VACUUM",
}


class UnsafeSQLError(ValueError):
    """Raised when generated SQL is not safe to execute."""


# Flattens parsed SQL tokens so keyword checks can inspect the whole statement.
def _flatten_token_values(statement: TokenList) -> list[str]:
    return [
        token.value.upper()
        for token in statement.flatten()
        if token.value and not token.is_whitespace
    ]


# Validates that SQL is a single read-only SELECT/WITH query.
def validate_select_only(sql: str) -> str:
    cleaned_sql = sql.strip().rstrip(";").strip()
    if not cleaned_sql:
        raise UnsafeSQLError("SQL query is empty.")

    statements = sqlparse.split(cleaned_sql)
    if len(statements) != 1:
        raise UnsafeSQLError("Only one SQL statement is allowed.")

    parsed = sqlparse.parse(cleaned_sql)
    if not parsed:
        raise UnsafeSQLError("SQL query could not be parsed.")

    statement = parsed[0]
    statement_type = statement.get_type().upper()
    if statement_type not in {"SELECT", "UNKNOWN"}:
        raise UnsafeSQLError("Only SELECT queries are allowed.")

    token_values = _flatten_token_values(statement)
    first_keyword = next((value for value in token_values if value not in {"(", ")"}), "")
    if first_keyword not in {"SELECT", "WITH"}:
        raise UnsafeSQLError("Only SELECT queries are allowed.")

    forbidden = FORBIDDEN_KEYWORDS.intersection(token_values)
    if forbidden:
        blocked = ", ".join(sorted(forbidden))
        raise UnsafeSQLError(f"Forbidden SQL keyword found: {blocked}")

    return cleaned_sql


# Checks whether a SQL statement already contains a LIMIT clause.
def has_limit(sql: str) -> bool:
    parsed = sqlparse.parse(sql)
    if not parsed:
        return False

    return "LIMIT" in _flatten_token_values(parsed[0])


# Adds a LIMIT clause after validating the SQL is safe.
def ensure_limit(sql: str, limit: int = 100) -> str:
    cleaned_sql = validate_select_only(sql)
    if has_limit(cleaned_sql):
        return cleaned_sql

    return f"{cleaned_sql} LIMIT {limit}"
