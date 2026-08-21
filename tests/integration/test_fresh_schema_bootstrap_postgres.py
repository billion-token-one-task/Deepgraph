"""A fresh PostgreSQL database must be creatable from the shipped schema.

This is the case that was never covered and never true. The startup schema
applier split schema_postgres.sql on ";" and then discarded any statement whose
chunk began with a comment line, which silently threw away 19 of 151
statements -- papers, deep_insights, experiment_runs, auto_research_jobs,
manuscript_runs and the additive ALTER TABLE repairs among them. Production
never noticed because its tables predate the function.

Run through scripts/run_isolated_postgres_tests.sh, which hands this module a
database that has had nothing applied to it.
"""

from __future__ import annotations

import os
import re
import unittest
from urllib.parse import urlsplit


URL = os.environ.get("DEEPGRAPH_ISOLATED_POSTGRES_URL", "").strip()
ACK = os.environ.get("DEEPGRAPH_ALLOW_ISOLATED_INTEGRATION_TESTS") == "1"
SOURCE_COMMIT = os.environ.get("META_HARNESS_CANDIDATE_COMMIT", "").strip()
ISOLATED_MARKERS = ("test", "ci", "canary", "sandbox", "restore", "shadow")


def _safe_url() -> bool:
    if not URL or not ACK or not re.fullmatch(r"[0-9a-f]{40}", SOURCE_COMMIT):
        return False
    parsed = urlsplit(URL)
    database = parsed.path.lstrip("/").lower()
    return bool(
        parsed.scheme in {"postgres", "postgresql"}
        and any(marker in database for marker in ISOLATED_MARKERS)
        and URL != os.environ.get("DEEPGRAPH_DATABASE_URL", "").strip()
    )


@unittest.skipUnless(_safe_url(), "explicit isolated PostgreSQL process required")
class FreshSchemaBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["DEEPGRAPH_DATABASE_URL"] = URL

        from db import database

        if not database._use_pg() or database.DATABASE_URL.strip() != URL:  # noqa: SLF001
            raise RuntimeError("database module captured a non-isolated URL")
        cls.db = database
        # Both cases call init_db(), and unittest does not run them in the
        # order they are written, so emptiness is recorded once here rather
        # than asserted from inside whichever case happens to run first.
        cls.initial_tables = cls._table_names()

    @classmethod
    def _table_names(cls) -> set[str]:
        with cls.db.get_conn().cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_type='BASE TABLE'"
            )
            return {dict(row)["table_name"] for row in cur.fetchall()}

    def _tables(self) -> set[str]:
        with self.db.get_conn().cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_type='BASE TABLE'"
            )
            return {dict(row)["table_name"] for row in cur.fetchall()}

    def test_init_db_creates_the_schema_on_an_empty_database(self):
        self.assertEqual(
            self.initial_tables, set(), "fixture handed over a non-empty database"
        )

        self.db.init_db()

        tables = self._tables()
        # Every table the file declares under a section header used to be
        # dropped with that header; naming them keeps the defect from coming
        # back as a quieter version of itself.
        for name in (
            "papers",
            "taxonomy_nodes",
            "deep_insights",
            "experiment_runs",
            "auto_research_jobs",
            "manuscript_runs",
            "gpu_workers",
            "insight_events",
        ):
            self.assertIn(name, tables)

    def test_additive_column_repairs_are_applied(self):
        self.db.init_db()
        with self.db.get_conn().cursor() as cur:
            cur.execute("ALTER TABLE papers DROP COLUMN IF EXISTS processing_stage")
        self.db.commit()
        self.db._pg_init_done = False  # noqa: SLF001 - force the repair pass to rerun

        self.db.init_db()

        with self.db.get_conn().cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM information_schema.columns "
                "WHERE table_name='papers' AND column_name='processing_stage'"
            )
            self.assertEqual(int(dict(cur.fetchone())["n"]), 1)


if __name__ == "__main__":
    unittest.main()
