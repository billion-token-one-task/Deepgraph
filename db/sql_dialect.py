"""Translate SQLite-oriented SQL to PostgreSQL where patterns are fixed."""

from __future__ import annotations

import re


def to_postgres(sql: str) -> str:
    """Best-effort ? -> %s and common SQLite idioms. Review generated SQL for edge cases."""
    out = sql.replace("?", "%s")
    out = re.sub(
        r"GROUP_CONCAT\s*\(\s*DISTINCT\s+([^)]+?)\s*\)",
        r"STRING_AGG(DISTINCT CAST(\1 AS TEXT), ',')",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"GROUP_CONCAT\s*\(\s*([^)]+?)\s*\)",
        r"STRING_AGG(CAST(\1 AS TEXT), ',')",
        out,
        flags=re.IGNORECASE,
    )
    # psycopg treats "%" as the start of a placeholder, so SQLite LIKE patterns
    # such as '.%' must be escaped after positional placeholders are rewritten.
    out = re.sub(r"(?<!%)%(?![%sbt(])", "%%", out)
    out = out.replace("CURRENT_TIMESTAMP", "NOW()")
    # INSERT OR IGNORE INTO papers -> ON CONFLICT DO NOTHING
    out = re.sub(
        r"INSERT OR IGNORE INTO papers\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)",
        r"INSERT INTO papers (\1) VALUES (\2) ON CONFLICT (id) DO NOTHING",
        out,
        flags=re.IGNORECASE | re.DOTALL,
    )
    out = re.sub(
        r"INSERT OR IGNORE INTO methods\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)",
        r"INSERT INTO methods (\1) VALUES (\2) ON CONFLICT (name) DO NOTHING",
        out,
        flags=re.IGNORECASE | re.DOTALL,
    )
    out = re.sub(
        r"INSERT OR IGNORE INTO taxonomy_nodes\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)",
        r"INSERT INTO taxonomy_nodes (\1) VALUES (\2) ON CONFLICT (id) DO NOTHING",
        out,
        flags=re.IGNORECASE | re.DOTALL,
    )
    out = out.replace("INSERT OR REPLACE INTO", "INSERT INTO")
    # Caller must add ON CONFLICT for REPLACE cases — handled in specialized paths
    return out


def strip_sqlite_master_query(sql: str) -> str | None:
    """Return None if query must be handled only on SQLite (skip on PG)."""
    s = sql.strip().upper()
    if "SQLITE_MASTER" in s or "PRAGMA" in s:
        return None
    return sql
