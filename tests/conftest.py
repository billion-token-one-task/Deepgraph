"""Hard database boundary for the candidate test entry point.

Unit and adapted-legacy tests always start with SQLite, even when the parent
process exported a database URL. Disposable PostgreSQL tests are explicit
integration modules and must opt in with their own guarded URL in a separate
process.
"""

from __future__ import annotations

import os


def _force_sqlite() -> None:
    # Assignment is intentional: setdefault would preserve a production URL.
    os.environ["DEEPGRAPH_DATABASE_URL"] = ""


_force_sqlite()


def pytest_configure(config) -> None:
    _force_sqlite()
    config.addinivalue_line(
        "markers",
        "isolated_postgres: explicit disposable PostgreSQL integration test",
    )


def _is_isolated_postgres_item(item) -> bool:
    path = str(getattr(item, "fspath", "")).replace("\\", "/")
    return "/tests/integration/" in path


def pytest_runtest_setup(item) -> None:
    if not _is_isolated_postgres_item(item):
        _force_sqlite()


def pytest_runtest_teardown(item, nextitem) -> None:
    if not _is_isolated_postgres_item(item):
        _force_sqlite()

