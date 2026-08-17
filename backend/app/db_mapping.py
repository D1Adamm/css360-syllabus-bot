"""SQL row -> API dict helpers shared by the PostgreSQL repositories.

The database is snake_case because `db/schema.sql` is; the API is camelCase
because the frontend already is. This module is the single place that crosses
between them, so a column rename shows up in one file rather than six.

Two conventions worth stating once:

  - Optional string columns are omitted from the API dict when NULL rather than
    emitted as null. The TypeScript types spread optional fields conditionally
    (`...(typeof record.notes === 'string' ? { notes } : {})`), so an explicit
    null is a value those parsers would have to learn to ignore.
  - Timestamps go out as ISO 8601 in UTC with a `+00:00` offset. Firebase holds
    a mix of `...Z` and `...+00:00` because different writers produced them;
    reading from `TIMESTAMPTZ` means every timestamp this layer emits has one
    shape, which is what the frontend's `localeCompare` sorting needs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def to_iso(value: Any) -> str | None:
    """Serialize a timestamptz to ISO 8601 UTC, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return aware.astimezone(timezone.utc).isoformat()
    text = str(value).strip()
    return text or None


def optional_string(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
        return None
    if value is None:
        return None
    return str(value)


def as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def string_list(value: Any) -> list[str]:
    """Normalize a JSONB array column into a list of non-empty strings."""
    if value is None:
        return []
    items = list(value.values()) if isinstance(value, dict) else value
    if not isinstance(items, list):
        return []
    return [
        text
        for text in (optional_string(item) for item in items)
        if text
    ]


def put_optional(target: dict[str, Any], key: str, value: Any) -> None:
    """Set `key` only when there is something to set.

    Mirrors the conditional spreads in the TypeScript parsers, so a record that
    round-trips through PostgreSQL has the same key set it had in Firebase.
    """
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    target[key] = value


def build_patch(
    patch: Mapping[str, Any],
    column_for_field: Mapping[str, str],
) -> dict[str, Any]:
    """Translate a camelCase API patch into snake_case column assignments.

    Unknown keys are dropped rather than rejected: a client sending a field this
    layer does not store should not fail the whole update, and — more to the
    point — a key that reached SQL unchecked would be a column name chosen by a
    caller. Only names from `column_for_field` are ever interpolated into a
    statement; every value stays a bound parameter.
    """
    assignments: dict[str, Any] = {}
    for field, value in patch.items():
        column = column_for_field.get(field)
        if column is not None:
            assignments[column] = value
    return assignments


def update_statement(
    *,
    table: str,
    assignments: Mapping[str, Any],
    key_columns: list[str],
) -> str:
    """Build a course-scoped UPDATE with placeholders for every value.

    `table`, the assigned column names, and the key columns are all literals
    chosen by this module's callers — never by request data. Values are bound.
    """
    if not assignments:
        raise ValueError("update_statement requires at least one assignment.")

    sets = ", ".join(f"{column} = %({column})s" for column in assignments)
    where = " AND ".join(f"{column} = %({column})s" for column in key_columns)
    return f"UPDATE {table} SET {sets} WHERE {where}"
