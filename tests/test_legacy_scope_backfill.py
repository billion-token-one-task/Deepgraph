"""Lineage-driven legacy scope backfill: audited, idempotent, NULL->agenda only."""

import tempfile
import unittest
from pathlib import Path

from db import database
from scripts import legacy_scope_backfill as backfill


def _reset_db(tmpdir: Path):
    for attr in ("pg_conn", "sqlite_conn", "conn"):
        if hasattr(database._local, attr):
            try:
                getattr(database._local, attr).close()
            except Exception:
                pass
            setattr(database._local, attr, None)
    database.DATABASE_URL = ""
    database.DB_PATH = tmpdir / "test.db"
    database.init_db()


def _create_meta_harness_tables():
    database.execute(
        """CREATE TABLE IF NOT EXISTS research_agendas (
            id INTEGER PRIMARY KEY, version TEXT, name TEXT, description TEXT,
            focus_json TEXT, prefer_json TEXT, reject_json TEXT,
            required_output_json TEXT, raw_config_json TEXT,
            is_active INTEGER DEFAULT 0, submitter TEXT,
            token_budget INTEGER DEFAULT 0, token_spent INTEGER DEFAULT 0,
            token_reserved INTEGER DEFAULT 0, gpu_hours_budget REAL DEFAULT 0,
            gpu_hours_spent REAL DEFAULT 0, gpu_hours_reserved REAL DEFAULT 0,
            max_concurrency INTEGER DEFAULT 1, backend_allowlist_json TEXT,
            backlog_policy TEXT, status TEXT,
            created_at TEXT DEFAULT '2026-08-01 00:00:00',
            updated_at TEXT DEFAULT '2026-08-01 00:00:00')"""
    )
    database.execute(
        """CREATE TABLE IF NOT EXISTS legacy_scope_imports (
            id INTEGER PRIMARY KEY, agenda_id INTEGER, entity_type TEXT,
            entity_id INTEGER, actor TEXT, reason TEXT, idempotency_key TEXT,
            imported_at TEXT DEFAULT '2026-08-01 00:00:00')"""
    )
    database.commit()


class LegacyScopeBackfillTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.old_db_path = database.DB_PATH
        self.old_database_url = database.DATABASE_URL
        _reset_db(Path(self.tmpdir.name))
        _create_meta_harness_tables()
        for table in ("experiment_runs", "experiment_iterations",
                      "experimental_claims", "experiment_artifacts",
                      "manuscript_runs", "submission_bundles",
                      "deep_insights", "auto_research_jobs"):
            try:
                database.execute(f"ALTER TABLE {table} ADD COLUMN agenda_id INTEGER")
            except Exception:
                pass
        # A scoped agenda + insight, and an orphan insight with a full lineage
        # chain hanging off it.
        database.execute(
            """INSERT INTO research_agendas
               (id, version, name, description, focus_json, status, is_active, token_budget)
               VALUES (1, 'v1', 'existing-agenda', 'd', '["existing"]', 'active', 1, 100)"""
        )
        database.execute(
            "INSERT INTO deep_insights (id, agenda_id, tier, title) VALUES (1, 1, 2, 'scoped idea')"
        )
        database.execute(
            "INSERT INTO deep_insights (id, tier, title) VALUES (2, 2, 'orphan idea')"
        )
        database.execute(
            "INSERT INTO experiment_runs (id, deep_insight_id, status) VALUES (10, 2, 'completed')"
        )
        database.execute(
            "INSERT INTO experiment_iterations (id, run_id, iteration_number, phase) VALUES (100, 10, 1, 'reproduction')"
        )
        database.execute(
            """INSERT INTO experimental_claims (id, run_id, deep_insight_id, claim_text, verdict)
               VALUES (200, 10, 2, 'c', 'inconclusive')"""
        )
        database.execute(
            "INSERT INTO experiment_runs (id, deep_insight_id, status) VALUES (11, 1, 'completed')"
        )
        database.execute(
            "INSERT INTO auto_research_jobs (id, deep_insight_id, status) VALUES (300, 1, 'completed')"
        )
        database.commit()

    def tearDown(self):
        database.DB_PATH = self.old_db_path
        database.DATABASE_URL = self.old_database_url
        self.tmpdir.cleanup()

    def test_dry_run_changes_nothing(self):
        agenda_id = backfill.ensure_orphan_agenda("legacy-test", execute=False)
        self.assertIsNone(agenda_id)
        backfill.import_orphan_insights(-1, execute=False)
        backfill.backfill_lineage(execute=False)
        row = database.fetchone("SELECT agenda_id FROM deep_insights WHERE id=2")
        self.assertIsNone(row["agenda_id"])
        row = database.fetchone("SELECT agenda_id FROM experiment_runs WHERE id=11")
        self.assertIsNone(row["agenda_id"])

    def test_execute_backfills_by_lineage_and_audits_imports(self):
        agenda_id = backfill.ensure_orphan_agenda("legacy-test", execute=True)
        self.assertIsNotNone(agenda_id)
        backfill.import_orphan_insights(agenda_id, execute=True)
        backfill.backfill_lineage(execute=True)
        backfill.import_scoped_jobs(execute=True)

        # Orphan insight went to the legacy agenda, audited.
        row = database.fetchone("SELECT agenda_id FROM deep_insights WHERE id=2")
        self.assertEqual(row["agenda_id"], agenda_id)
        audit = database.fetchone(
            "SELECT actor FROM legacy_scope_imports WHERE entity_type='deep_insight' AND entity_id=2"
        )
        self.assertEqual(audit["actor"], backfill.ACTOR)

        # Lineage propagation: run 10 follows insight 2 into the legacy agenda,
        # its iteration/claim follow the run; run 11 follows insight 1.
        self.assertEqual(
            database.fetchone("SELECT agenda_id FROM experiment_runs WHERE id=10")["agenda_id"],
            agenda_id,
        )
        self.assertEqual(
            database.fetchone("SELECT agenda_id FROM experiment_iterations WHERE id=100")["agenda_id"],
            agenda_id,
        )
        self.assertEqual(
            database.fetchone("SELECT agenda_id FROM experimental_claims WHERE id=200")["agenda_id"],
            agenda_id,
        )
        self.assertEqual(
            database.fetchone("SELECT agenda_id FROM experiment_runs WHERE id=11")["agenda_id"],
            1,
        )
        # Job followed its scoped insight, audited.
        self.assertEqual(
            database.fetchone("SELECT agenda_id FROM auto_research_jobs WHERE id=300")["agenda_id"],
            1,
        )

    def test_execute_is_idempotent_and_never_rescopes(self):
        agenda_id = backfill.ensure_orphan_agenda("legacy-test", execute=True)
        backfill.import_orphan_insights(agenda_id, execute=True)
        backfill.backfill_lineage(execute=True)
        # Second pass: no error, nothing re-scoped.
        again = backfill.ensure_orphan_agenda("legacy-test", execute=True)
        self.assertEqual(again, agenda_id)
        backfill.import_orphan_insights(agenda_id, execute=True)
        backfill.backfill_lineage(execute=True)
        count = database.fetchone(
            "SELECT COUNT(*) as c FROM legacy_scope_imports WHERE entity_type='deep_insight' AND entity_id=2"
        )["c"]
        self.assertEqual(count, 1)
        # A row already scoped elsewhere is never moved.
        self.assertEqual(
            database.fetchone("SELECT agenda_id FROM experiment_runs WHERE id=11")["agenda_id"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
