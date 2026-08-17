"""A connection that dies before cursor() must still be recovered.

execute() already retried a read once when PostgreSQL dropped the session, but
it created the cursor *outside* the try block. psycopg raises "the connection is
closed" from cursor() itself, so that path skipped the retry and surfaced as a
500. /api/stats recovered because its own caller re-read; /api/agent_office had
no such caller and failed with exactly that message, repeatedly, in production
on 2026-08-17.
"""

import unittest
from unittest import mock

from db import database


class _DeadCursorConnection:
    """Raises on cursor() the first time, like a session dropped in between."""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.cursor_calls = 0
        self.rollbacks = 0

    def cursor(self):
        self.cursor_calls += 1
        if self.cursor_calls <= self.fail_times:
            raise RuntimeError("the connection is closed")
        return _Cursor()

    def rollback(self):
        self.rollbacks += 1


class _Cursor:
    def __init__(self):
        self.executed = None

    def execute(self, sql, params):
        self.executed = (sql, params)


class CursorReconnectTests(unittest.TestCase):
    def _run(self, sql, fail_times, replacement_fail_times=0):
        """Hand out a dead connection first, then a replacement.

        replacement_fail_times models the case where reconnecting does not help,
        so the retry must give up rather than loop.
        """
        handed_out = []

        def fake_get_conn():
            conn = _DeadCursorConnection(
                fail_times=fail_times if not handed_out else replacement_fail_times
            )
            handed_out.append(conn)
            return conn

        with (
            mock.patch.object(database, "_use_pg", return_value=True),
            mock.patch.object(database, "get_conn", side_effect=fake_get_conn),
            mock.patch.object(database, "_discard_pg_connection"),
        ):
            return database.execute(sql, ()), handed_out

    def test_read_survives_a_connection_that_dies_before_cursor(self):
        cur, handed_out = self._run("SELECT 1 AS x", fail_times=1)
        self.assertIsInstance(cur, _Cursor)
        self.assertEqual(len(handed_out), 2, "the read should have been retried once")

    def test_a_write_is_not_retried(self):
        with self.assertRaises(RuntimeError):
            self._run("UPDATE papers SET status='x' WHERE id='y'", fail_times=1)

    def test_a_persistently_dead_connection_still_raises(self):
        with self.assertRaises(RuntimeError):
            self._run("SELECT 1 AS x", fail_times=1, replacement_fail_times=1)


if __name__ == "__main__":
    unittest.main()
