"""Give a test its own SQLite database with the stock schema.

Some tests call production code that reads the database even though the test
only cares about the branch logic around it (a scheduler pass that first runs
a recovery query, a novelty score that first counts papers in the taxonomy).
They used to pass only because a gitignored ``deepgraph.db`` with the schema
happened to be sitting in the repository root; in a fresh checkout the default
database does not exist, SQLite creates an empty file on first connect, and the
first query dies with ``no such table``.  A test that needs tables should own
them, so these tests call :func:`use_temp_sqlite` in ``setUp``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from db import database


def _close_thread_connections() -> None:
    for attr in ("pg_conn", "sqlite_conn", "conn"):
        conn = getattr(database._local, attr, None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        setattr(database._local, attr, None)


def use_temp_sqlite(testcase: unittest.TestCase) -> Path:
    """Point ``db.database`` at a fresh, schema-initialised temp SQLite file.

    Everything is undone through ``addCleanup``: the original path and URL are
    restored, the thread-local connections are closed, and the temp directory
    is deleted.  Returns the path of the temp database.
    """
    tmpdir = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmpdir.cleanup)
    old_path = database.DB_PATH
    old_url = database.DATABASE_URL

    def restore() -> None:
        _close_thread_connections()
        database.DB_PATH = old_path
        database.DATABASE_URL = old_url

    # Registered after tmpdir.cleanup, so it runs first (LIFO): connections are
    # closed before the directory they point into is removed.
    testcase.addCleanup(restore)
    _close_thread_connections()
    database.DATABASE_URL = ""
    database.DB_PATH = Path(tmpdir.name) / "test.db"
    database.init_db()
    return database.DB_PATH
