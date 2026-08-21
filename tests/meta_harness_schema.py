"""Skip helpers for tests whose tables the PostgreSQL-only migration owns.

The meta-harness v1 schema (agenda scope, resource grants, evidence state,
scientific decisions) is created by ``scripts/meta_harness_migration.py``,
which refuses to run on anything but PostgreSQL.  ``init_db()`` on SQLite
therefore produces a database that predates agenda scope: ``experiment_runs``
exists but has no ``agenda_id``, and ``resource_grants`` does not exist at all.

Tests that need that schema are covered on a real PostgreSQL process by
``scripts/run_isolated_postgres_tests.sh``.  Under SQLite they skip, with the
missing object named, rather than fail.  Re-creating the tables here by hand is
the drift that hid a production NotNullViolation once already; that is why the
answer is a skip and not a SQLite copy of the schema.
"""

from __future__ import annotations

import functools
import unittest

import pytest

from db import database


def _sqlite_has_table(name: str) -> bool:
    rows = database.fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return bool(rows)


def _sqlite_has_column(table: str, column: str) -> bool:
    if not _sqlite_has_table(table):
        return False
    rows = database.fetchall(f"PRAGMA table_info({table})")
    return any(str(dict(row).get("name")) == column for row in rows)


def _reason_for(table: str, column: str | None) -> str | None:
    """Return a skip reason, or None when the schema is actually present.

    On PostgreSQL the objects exist and nothing is skipped, so a test guarded
    by these helpers still runs -- and still fails if the behaviour regresses --
    under the isolated PostgreSQL fixture.
    """
    if database._use_pg():  # noqa: SLF001 - the backend switch is the whole point
        return None
    if column is None:
        if _sqlite_has_table(table):
            return None
        return (
            f"{table} is owned by the PostgreSQL-only meta-harness migration"
        )
    if _sqlite_has_column(table, column):
        return None
    return (
        f"{table}.{column} is owned by the PostgreSQL-only meta-harness "
        "migration"
    )


def require_meta_harness_schema(case: unittest.TestCase, table: str, column: str | None = None) -> None:
    """Skip ``case`` unless the named migration-owned object exists."""
    reason = _reason_for(table, column)
    if reason:
        case.skipTest(reason)


def skip_without_meta_harness_schema(table: str, column: str | None = None):
    """Decorator form, for plain pytest functions.

    The check runs when the test runs, not when the module is imported: at
    import time no database has been opened yet, so an import-time check would
    answer for the wrong backend.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            reason = _reason_for(table, column)
            if reason:
                pytest.skip(reason)
            return func(*args, **kwargs)

        return wrapper

    return decorate


def agenda_scope_reason() -> str | None:
    """Reason a test needing agenda-scoped ``experiment_runs`` cannot run."""
    return _reason_for("experiment_runs", "agenda_id")
