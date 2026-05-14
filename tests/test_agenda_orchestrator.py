"""Tests for agents.agenda_orchestrator (issue #9 step 4 - existing-loop hookup)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DEEPGRAPH_DATABASE_URL", "")


SAMPLE_AGENDA_DICT = {
    "version": "v1",
    "name": "orch_test_agenda",
    "focus": ["long context"],
    "prefer": {"keywords": ["linear attention"], "tiers": ["tier_2"]},
    "reject": {},
}


class OrchestratorLinkAndEnqueueTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["DEEPGRAPH_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        from db import database as db

        self._original_db_path = db.DB_PATH
        for attr in ("sqlite_conn", "pg_conn", "conn"):
            if hasattr(db._local, attr):
                try:
                    getattr(db._local, attr).close()
                except Exception:
                    pass
                delattr(db._local, attr)
        db.DB_PATH = Path(os.environ["DEEPGRAPH_DB_PATH"])
        db.init_db()
        self.db = db

    def tearDown(self):
        from db import database as db

        for attr in ("sqlite_conn", "pg_conn", "conn"):
            if hasattr(db._local, attr):
                try:
                    getattr(db._local, attr).close()
                except Exception:
                    pass
                delattr(db._local, attr)
        # Restore original DB_PATH so downstream tests don't inherit a deleted
        # tempdir path through the module-level singleton.
        db.DB_PATH = self._original_db_path
        self._tmpdir.cleanup()
        os.environ.pop("DEEPGRAPH_DB_PATH", None)

    def _seed_insight(self, insight_id=1, title="Linear attention long context"):
        self.db.execute(
            """
            INSERT INTO deep_insights
                (id, tier, status, title, problem_statement, adversarial_score,
                 novelty_status, resource_class, experimentability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                insight_id,
                2,
                "verified",
                title,
                "Quadratic attention is too expensive for long context.",
                8.5,
                "novel",
                "gpu_small",
                "easy",
            ),
        )
        self.db.commit()

    def _seed_run_with_bundle(self, insight_id):
        # experiment_run
        cur = self.db.execute(
            """
            INSERT INTO experiment_runs
                (deep_insight_id, status, phase, hypothesis_verdict,
                 baseline_metric_value, best_metric_value, effect_size, workdir)
            VALUES (?, 'completed', 'hypothesis_testing', 'confirmed', 0.65, 0.78, 0.13, '/tmp/wd')
            """,
            (insight_id,),
        )
        exp_id = cur.lastrowid
        # manuscript_run
        cur = self.db.execute(
            """
            INSERT INTO manuscript_runs
                (experiment_run_id, deep_insight_id, status, workdir)
            VALUES (?, ?, 'bundle_ready', '/tmp/manu')
            """,
            (exp_id, insight_id),
        )
        manu_id = cur.lastrowid
        # submission_bundle
        cur = self.db.execute(
            """
            INSERT INTO submission_bundles
                (manuscript_run_id, bundle_format, status, bundle_path, manifest_path)
            VALUES (?, 'conference', 'ready', '/tmp/bundle.zip', '/tmp/manifest.json')
            """,
            (manu_id,),
        )
        bundle_id = cur.lastrowid
        # backfill submission_bundle_id on experiment_run
        self.db.execute(
            "UPDATE experiment_runs SET submission_bundle_id=? WHERE id=?",
            (bundle_id, exp_id),
        )
        self.db.commit()
        return {"experiment_run_id": exp_id, "manuscript_run_id": manu_id, "submission_bundle_id": bundle_id}

    def _new_selection(self, insight_id):
        from agents.agenda_loader import parse_agenda, save_agenda
        from agents.agenda_selector import select_and_persist

        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        save_agenda(agenda)
        sel = select_and_persist(agenda)
        self.assertEqual(sel.selected_insight_id, insight_id)
        return sel

    def test_link_picks_up_existing_bundle(self):
        from agents.agenda_orchestrator import link_existing_artifacts
        from agents.agenda_selector import get_selection

        self._seed_insight(1)
        ids = self._seed_run_with_bundle(1)
        sel = self._new_selection(1)

        result = link_existing_artifacts(sel.selection_id)
        self.assertEqual(result["insight_id"], 1)
        self.assertEqual(result["experiment_run"]["id"], ids["experiment_run_id"])
        self.assertEqual(result["manuscript_run"]["id"], ids["manuscript_run_id"])
        self.assertEqual(result["submission_bundle"]["id"], ids["submission_bundle_id"])

        row = get_selection(sel.selection_id)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["experiment_run_id"], ids["experiment_run_id"])
        self.assertEqual(row["manuscript_run_id"], ids["manuscript_run_id"])
        self.assertEqual(row["submission_bundle_id"], ids["submission_bundle_id"])

    def test_link_with_no_artifacts_marks_blocked(self):
        from agents.agenda_orchestrator import link_existing_artifacts
        from agents.agenda_selector import get_selection

        self._seed_insight(1)
        sel = self._new_selection(1)

        result = link_existing_artifacts(sel.selection_id)
        self.assertIsNone(result["experiment_run"])
        row = get_selection(sel.selection_id)
        self.assertEqual(row["status"], "blocked")
        self.assertIn("No existing artifacts", row["error_message"])

    def test_enqueue_creates_auto_research_job(self):
        from agents.agenda_orchestrator import enqueue_for_auto_research
        from agents.agenda_selector import get_selection

        self._seed_insight(1)
        sel = self._new_selection(1)

        result = enqueue_for_auto_research(sel.selection_id)
        self.assertIsNotNone(result["auto_research_job"])
        self.assertEqual(result["auto_research_job"]["status"], "queued")

        row = get_selection(sel.selection_id)
        self.assertEqual(row["status"], "launched")
        self.assertEqual(row["auto_research_job_id"], result["auto_research_job"]["id"])

    def test_dispatch_auto_mode_links_when_possible(self):
        from agents.agenda_orchestrator import dispatch_selection

        self._seed_insight(1)
        self._seed_run_with_bundle(1)
        sel = self._new_selection(1)

        result = dispatch_selection(sel.selection_id, mode="auto")
        self.assertEqual(result["mode"], "link")
        self.assertIsNotNone(result["submission_bundle"])

    def test_dispatch_auto_mode_falls_back_to_enqueue(self):
        from agents.agenda_orchestrator import dispatch_selection

        self._seed_insight(1)
        sel = self._new_selection(1)

        result = dispatch_selection(sel.selection_id, mode="auto")
        self.assertEqual(result["mode"], "enqueue")
        self.assertIsNotNone(result["auto_research_job"])


if __name__ == "__main__":
    unittest.main()
