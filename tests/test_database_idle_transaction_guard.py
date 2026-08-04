"""Every PostgreSQL session must carry an idle-in-transaction reclaim bound.

A worker that dies between statements leaves its backend "idle in transaction"
holding row locks. With PostgreSQL's default of 0 nothing ever reclaims it, and
the pipeline freezes while the web process keeps answering 200 -- the silent
stall failure mode. The guard travels on the connection string rather than a
``SET`` statement because ``SET`` is transactional: a rollback would drop it at
exactly the moment a stuck transaction is being cleaned up.
"""

import unittest
from unittest import mock

import psycopg
from psycopg import conninfo as pg_conninfo

from db import database


TEST_URL = "postgresql://user:pw@127.0.0.1:5433/deepgraph"


def _options(conninfo: str) -> str:
    return str(pg_conninfo.conninfo_to_dict(conninfo).get("options") or "")


class IdleTransactionGuardTests(unittest.TestCase):
    def test_default_timeout_is_a_bounded_number_of_minutes(self):
        self.assertGreater(database.PG_IDLE_IN_TRANSACTION_TIMEOUT_MS, 0)
        minutes = database.PG_IDLE_IN_TRANSACTION_TIMEOUT_MS / 60000
        self.assertGreaterEqual(minutes, 5)
        self.assertLessEqual(minutes, 10)

    def test_conninfo_carries_the_timeout(self):
        with (
            mock.patch.object(database, "DATABASE_URL", TEST_URL),
            mock.patch.object(database, "PG_IDLE_IN_TRANSACTION_TIMEOUT_MS", 600000),
        ):
            options = _options(database._pg_conninfo())

        self.assertIn("-c idle_in_transaction_session_timeout=600000", options)

    def test_existing_conninfo_options_are_preserved(self):
        url = f"{TEST_URL}?options=-c%20statement_timeout%3D60s"
        with (
            mock.patch.object(database, "DATABASE_URL", url),
            mock.patch.object(database, "PG_IDLE_IN_TRANSACTION_TIMEOUT_MS", 600000),
        ):
            options = _options(database._pg_conninfo())

        self.assertIn("-c statement_timeout=60s", options)
        self.assertIn("-c idle_in_transaction_session_timeout=600000", options)

    def test_zero_restores_the_server_default(self):
        with (
            mock.patch.object(database, "DATABASE_URL", TEST_URL),
            mock.patch.object(database, "PG_IDLE_IN_TRANSACTION_TIMEOUT_MS", 0),
        ):
            self.assertEqual(database._pg_conninfo(), TEST_URL)

    def test_the_guard_is_not_issued_as_a_rollback_able_set(self):
        """A `SET` would be undone by the very rollback it needs to survive."""
        recorded = {}

        def fake_connect(conninfo, **kwargs):
            recorded["conninfo"] = conninfo
            recorded["autocommit"] = kwargs.get("autocommit")
            return mock.MagicMock()

        with (
            mock.patch.object(database, "DATABASE_URL", TEST_URL),
            mock.patch.object(database, "PG_IDLE_IN_TRANSACTION_TIMEOUT_MS", 600000),
            mock.patch.object(database.psycopg, "connect", fake_connect),
        ):
            conn = database._pg_connect()

        self.assertIn("idle_in_transaction_session_timeout", recorded["conninfo"])
        self.assertFalse(recorded["autocommit"])
        conn.execute.assert_not_called()
        conn.cursor.assert_not_called()

    def test_a_pooler_that_rejects_options_still_gets_a_connection(self):
        """Losing the guard is bad; losing the database is worse."""
        attempts = []

        def fake_connect(conninfo, **kwargs):
            attempts.append(conninfo)
            if "idle_in_transaction_session_timeout" in conninfo:
                raise psycopg.OperationalError("unsupported startup parameter: options")
            return mock.MagicMock()

        with (
            mock.patch.object(database, "DATABASE_URL", TEST_URL),
            mock.patch.object(database, "PG_IDLE_IN_TRANSACTION_TIMEOUT_MS", 600000),
            mock.patch.object(database.psycopg, "connect", fake_connect),
        ):
            database._pg_connect()

        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[1], TEST_URL)

    def test_an_unrelated_connection_failure_is_not_retried_or_swallowed(self):
        attempts = []

        def fake_connect(conninfo, **kwargs):
            attempts.append(conninfo)
            raise psycopg.OperationalError("connection refused")

        with (
            mock.patch.object(database, "DATABASE_URL", TEST_URL),
            mock.patch.object(database, "PG_IDLE_IN_TRANSACTION_TIMEOUT_MS", 600000),
            mock.patch.object(database.psycopg, "connect", fake_connect),
        ):
            with self.assertRaisesRegex(psycopg.OperationalError, "connection refused"):
                database._pg_connect()

        self.assertEqual(len(attempts), 1)


if __name__ == "__main__":
    unittest.main()
