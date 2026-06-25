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


if __name__ == "__main__":
    unittest.main()
