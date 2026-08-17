import sqlite3
import unittest
from unittest import mock

from db import database


class SqliteLockRetryTests(unittest.TestCase):
    def test_execute_retries_transient_sqlite_lock_errors(self):
        class FakeConn:
            def __init__(self):
                self.calls = 0

            def execute(self, sql, params=()):
                self.calls += 1
                if self.calls < 3:
                    raise sqlite3.OperationalError("database is locked")
                return "cursor"

        fake = FakeConn()
        with (
            mock.patch.object(database, "_use_pg", return_value=False),
            mock.patch.object(database, "get_conn", return_value=fake),
            mock.patch.object(database, "SQLITE_LOCK_RETRY_SECONDS", 1.0),
            mock.patch.object(database.time, "sleep"),
        ):
            result = database.execute("SELECT 1")

        self.assertEqual(result, "cursor")
        self.assertEqual(fake.calls, 3)

    def test_execute_reraises_non_lock_sqlite_errors(self):
        class FakeConn:
            def execute(self, sql, params=()):
                raise sqlite3.OperationalError("no such table: missing")

        with (
            mock.patch.object(database, "_use_pg", return_value=False),
            mock.patch.object(database, "get_conn", return_value=FakeConn()),
            mock.patch.object(database.time, "sleep") as sleep_mock,
        ):
            with self.assertRaises(sqlite3.OperationalError):
                database.execute("SELECT * FROM missing")

        sleep_mock.assert_not_called()


class PostgresReconnectTests(unittest.TestCase):
    def tearDown(self):
        database._local.pg_conn = None

    def test_select_rebuilds_closed_connection_and_retries_once(self):
        class FakeCursor:
            def __init__(self, conn, result=None):
                self.conn = conn
                self.result = result

            def execute(self, sql, params=()):
                self.conn.calls += 1
                if self.conn.fail:
                    self.conn.closed = 1
                    raise RuntimeError("connection is closed")
                return self.result

        class FakeConn:
            def __init__(self, fail):
                self.fail = fail
                self.closed = 0
                self.calls = 0

            def cursor(self):
                return FakeCursor(self)

            def close(self):
                self.closed = 1

            def rollback(self):
                if self.closed:
                    raise RuntimeError("connection is closed")

        stale = FakeConn(fail=True)
        fresh = FakeConn(fail=False)
        database._local.pg_conn = stale
        with (
            mock.patch.object(database, "_use_pg", return_value=True),
            mock.patch.object(database, "_pg_connect", return_value=fresh) as reconnect,
        ):
            cursor = database.execute("SELECT 1")

        self.assertIsInstance(cursor, FakeCursor)
        reconnect.assert_called_once_with()
        self.assertEqual(stale.calls, 1)
        self.assertEqual(fresh.calls, 1)
        self.assertIs(database._local.pg_conn, fresh)

    def test_mutation_is_never_retried_after_ambiguous_disconnect(self):
        class FakeCursor:
            def __init__(self, conn):
                self.conn = conn

            def execute(self, sql, params=()):
                self.conn.closed = 1
                raise RuntimeError("server closed the connection unexpectedly")

        class FakeConn:
            closed = 0

            def cursor(self):
                return FakeCursor(self)

            def close(self):
                self.closed = 1

        stale = FakeConn()
        database._local.pg_conn = stale
        with (
            mock.patch.object(database, "_use_pg", return_value=True),
            mock.patch.object(database, "_pg_connect") as reconnect,
        ):
            with self.assertRaisesRegex(RuntimeError, "server closed"):
                database.execute("UPDATE papers SET status=? WHERE id=?", ("done", 1))

        reconnect.assert_not_called()
        self.assertIsNone(database._local.pg_conn)

    def test_rollback_discards_closed_connection_without_reconnecting(self):
        stale = mock.Mock(closed=1)
        database._local.pg_conn = stale
        with (
            mock.patch.object(database, "_use_pg", return_value=True),
            mock.patch.object(database, "_pg_connect") as reconnect,
        ):
            database.rollback()

        reconnect.assert_not_called()
        self.assertIsNone(database._local.pg_conn)


if __name__ == "__main__":
    unittest.main()
