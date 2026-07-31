"""Migration/idempotency tests for a disposable PostgreSQL restore only.

This module refuses to run unless the CI operator explicitly marks the
database as isolated. It must never be selected by production-host test jobs.
"""

from __future__ import annotations

import os
import re
import unittest
from urllib.parse import urlsplit

from scripts.meta_harness_migration import apply_to_isolated_restore


URL = os.environ.get("DEEPGRAPH_ISOLATED_POSTGRES_URL", "").strip()
ACK = os.environ.get("DEEPGRAPH_ALLOW_ISOLATED_INTEGRATION_TESTS") == "1"
SOURCE_COMMIT = os.environ.get("META_HARNESS_CANDIDATE_COMMIT", "").strip()
ISOLATED_MARKERS = ("test", "ci", "canary", "sandbox", "restore", "shadow")


def _safe_url() -> bool:
    if not URL or not ACK or not re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT):
        return False
    parsed = urlsplit(URL)
    database = parsed.path.lstrip("/").lower()
    if parsed.scheme not in {"postgres", "postgresql"}:
        return False
    if not any(marker in database for marker in ISOLATED_MARKERS):
        return False
    return URL != os.environ.get("DEEPGRAPH_DATABASE_URL", "").strip()


@unittest.skipUnless(_safe_url(), "explicit isolated PostgreSQL restore required")
class IsolatedPostgresMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import psycopg

        cls.psycopg = psycopg

    def _counts(self, only_tables=None):
        with self.psycopg.connect(URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema='public'
                      AND table_type='BASE TABLE'
                      AND table_name IN (
                        'research_agendas', 'deep_insights',
                        'auto_research_jobs', 'experiment_runs',
                        'manuscript_runs', 'submission_bundles'
                      )
                    ORDER BY table_name
                    """
                )
                tables = [row[0] for row in cur.fetchall()]
                if only_tables is not None:
                    tables = [table for table in tables if table in set(only_tables)]
                counts = {}
                for table in tables:
                    cur.execute(f'SELECT COUNT(*) FROM "{table}"')
                    counts[table] = int(cur.fetchone()[0])
                return counts

    def _nonnull_scope_counts(self):
        with self.psycopg.connect(URL) as conn:
            with conn.cursor() as cur:
                values = {}
                for table in ("deep_insights", "auto_research_jobs"):
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM information_schema.columns
                        WHERE table_schema='public' AND table_name=%s
                          AND column_name='agenda_id'
                        """,
                        (table,),
                    )
                    if int(cur.fetchone()[0]) == 0:
                        values[table] = 0
                        continue
                    cur.execute(
                        f'SELECT COUNT(*) FROM "{table}" WHERE agenda_id IS NOT NULL'
                    )
                    values[table] = int(cur.fetchone()[0])
                return values

    def test_additive_migration_is_idempotent_and_does_not_import_backlog(self):
        before_counts = self._counts()
        before_scoped = self._nonnull_scope_counts()
        first = apply_to_isolated_restore(URL, source_commit=SOURCE_COMMIT)
        self.assertEqual(first["status"], "applied")
        after_first_counts = self._counts(only_tables=before_counts)
        after_first_scoped = self._nonnull_scope_counts()
        second = apply_to_isolated_restore(URL, source_commit=SOURCE_COMMIT)
        self.assertEqual(second["status"], "already_applied")
        self.assertEqual(before_counts, after_first_counts)
        self.assertEqual(after_first_counts, self._counts(only_tables=before_counts))
        self.assertEqual(before_scoped, after_first_scoped)
        self.assertEqual(after_first_scoped, self._nonnull_scope_counts())


if __name__ == "__main__":
    unittest.main()
