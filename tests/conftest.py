"""Hard database boundary for the candidate test entry point.

Unit and adapted-legacy tests always start with SQLite, even when the parent
process exported a database URL. Disposable PostgreSQL tests are explicit
integration modules and must opt in with their own guarded URL in a separate
process.
"""

from __future__ import annotations

import os
import shutil
import tempfile

# The default SQLite path is <repo>/deepgraph.db. A test that reaches the
# default database by accident used to create (or, worse, reuse) that file, so
# results depended on what an earlier run had left in the checkout. Point the
# default at a path inside a throwaway directory that starts out empty: a test
# that needs tables must create its own (tests/sqlite_isolation.py), and no run
# can leave anything behind in the repository. Assignment is intentional, for
# the same reason as the database URL below. This runs before `config` is
# imported, which is when the path is read.
_DEFAULT_DB_DIR = tempfile.mkdtemp(prefix="deepgraph-tests-default-db-")
os.environ["DEEPGRAPH_DB_PATH"] = os.path.join(_DEFAULT_DB_DIR, "default-unused.db")


def pytest_unconfigure(config) -> None:
    shutil.rmtree(_DEFAULT_DB_DIR, ignore_errors=True)


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

