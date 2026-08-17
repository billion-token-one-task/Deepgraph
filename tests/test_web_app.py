import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from agents import workspace_layout
from db import database
from web import app as web_app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.client = web_app.app.test_client()

    def test_api_events_serializes_datetime_payloads(self):
        with mock.patch.object(
            web_app,
            "get_events",
            side_effect=[
                [{"seq": 1, "created_at": datetime(2026, 4, 21, 12, 0, 0)}],
                [{"seq": 1, "created_at": datetime(2026, 4, 21, 12, 0, 0)}],
            ],
        ):
            response = self.client.get("/api/events")
            first_chunk = next(response.response).decode("utf-8")

        self.assertIn("2026-04-21 12:00:00", first_chunk)
        self.assertIn('"seq": 1', first_chunk)

    def test_api_meta_includes_database_backend_summary(self):
        response = self.client.get("/api/meta")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("database", payload)
        self.assertIn("backend", payload["database"])
        self.assertNotIn("target", payload["database"])

    def test_runtime_and_provider_topology_are_not_public(self):
        self.assertEqual(self.client.get("/api/runtime-config").status_code, 404)
        self.assertEqual(self.client.get("/api/providers").status_code, 404)

    def test_manual_post_api_returns_gone_in_fixed_flow_mode(self):
        response = self.client.post("/api/experiments/run_full", json={"insight_id": 1})
        payload = response.get_json()

        self.assertEqual(response.status_code, 410)
        self.assertEqual(payload["mode"], "fixed_flow_read_only")
        self.assertIn("removed", payload["error"].lower())
        self.assertEqual(payload["replacement"], "/api/meta-harness/v1")

    def test_legacy_compute_get_does_not_start_scheduler(self):
        response = self.client.get("/api/gpu/status")
        self.assertEqual(response.status_code, 410)

    def test_unparseable_agenda_scope_is_still_refused(self):
        """Omitting agenda_id is allowed now, but garbage still is not.

        This replaces test_agenda_owned_read_requires_explicit_scope, which
        asserted 400 for an omitted scope. That rule was never an authorization
        boundary: the read API has no per-agenda access control, so any caller
        could enumerate agenda_id=1..N and see everything the 400 pretended to
        withhold. What it did do was split the dashboard in half -- the
        front-page counters sum across every agenda while every detail list was
        pinned to one, so the page showed large totals above empty tables.
        AgendaScopeApiTests covers the accepting side.
        """
        for value in ("abc", "-1", "1;drop"):
            with self.subTest(agenda_id=value):
                response = self.client.get(f"/api/deep_insights?agenda_id={value}")
                self.assertEqual(response.status_code, 400)


class ExperimentGroupApiTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test.db"
        self.old_db_path = database.DB_PATH
        self.old_database_url = database.DATABASE_URL
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(database._local, attr):
                try:
                    getattr(database._local, attr).close()
                except Exception:
                    pass
                setattr(database._local, attr, None)
        database.DATABASE_URL = ""
        database.DB_PATH = self.db_path
        database.init_db()
        for table in (
            "deep_insights",
            "auto_research_jobs",
            "experiment_runs",
            "experiment_artifacts",
            "experimental_claims",
        ):
            columns = {
                row["name"]
                for row in database.fetchall(f"PRAGMA table_info({table})")
            }
            if "agenda_id" not in columns:
                database.execute(
                    f"ALTER TABLE {table} ADD COLUMN agenda_id INTEGER"
                )
        self.workspace_root = Path(self.tmpdir.name) / "ideas"
        self.workspace_patch = mock.patch.object(workspace_layout, "IDEA_WORKSPACE_DIR", self.workspace_root)
        self.workspace_patch.start()
        self.client = web_app.app.test_client()

        database.execute(
            """
            INSERT INTO deep_insights
            (id, agenda_id, tier, title, submission_status, evidence_plan,
             experimental_plan)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                1,
                2,
                "Idea One",
                "not_started",
                json.dumps({"ablation": {"enabled": True}, "visualization": {"enabled": True}}),
                json.dumps({"ablations": [{"name": "drop_gate"}]}),
            ),
        )
        database.execute(
            """
            INSERT INTO auto_research_jobs
            (agenda_id, deep_insight_id, status, stage, last_note)
            VALUES (?, ?, ?, ?, ?)
            """,
            (1, 1, "running_gpu", "gpu_scheduler", "Main run still progressing"),
        )
        database.execute(
            """
            INSERT INTO experiment_runs
            (id, agenda_id, deep_insight_id, status, hypothesis_verdict,
             effect_pct, iterations_total, iterations_kept, workdir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (10, 1, 1, "completed", "confirmed", 12.5, 8, 3, str(self.workspace_root / "legacy_run_10")),
        )
        database.execute(
            """
            INSERT INTO experiment_runs
            (id, agenda_id, deep_insight_id, status, iterations_total,
             iterations_kept, workdir)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (11, 1, 1, "testing", 2, 0, str(self.workspace_root / "legacy_run_11")),
        )
        database.execute(
            """INSERT INTO experiment_artifacts
               (agenda_id, run_id, artifact_type, path) VALUES (?, ?, ?, ?)""",
            (1, 11, "plot", "/tmp/plot.svg"),
        )
        database.execute(
            """INSERT INTO experimental_claims
               (agenda_id, run_id, deep_insight_id, claim_text, verdict)
               VALUES (?, ?, ?, ?, ?)""",
            (1, 10, 1, "Improves metric", "confirmed"),
        )
        plan_root = self.workspace_root / "idea_1" / "plan"
        paper_root = self.workspace_root / "idea_1" / "paper" / "current"
        plan_root.mkdir(parents=True, exist_ok=True)
        paper_root.mkdir(parents=True, exist_ok=True)
        (plan_root / "latest_status.json").write_text(json.dumps({"stage": "testing", "status": "testing"}), encoding="utf-8")
        (plan_root / "experiment_spec.json").write_text(json.dumps({"run_id": 11, "note": "spec"}), encoding="utf-8")
        (paper_root / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")
        database.commit()

    def tearDown(self):
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(database._local, attr):
                try:
                    getattr(database._local, attr).close()
                except Exception:
                    pass
                setattr(database._local, attr, None)
        database.DATABASE_URL = self.old_database_url
        database.DB_PATH = self.old_db_path
        self.workspace_patch.stop()
        self.tmpdir.cleanup()

    def test_api_experiment_groups_returns_idea_centric_cards(self):
        response = self.client.get("/api/experiment_groups?agenda_id=1")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(payload), 1)
        group = payload[0]
        self.assertEqual(group["insight"]["title"], "Idea One")
        self.assertEqual(group["run_count"], 2)
        self.assertEqual(group["canonical_run"]["id"], 11)
        self.assertEqual(group["latest_run"]["id"], 11)
        self.assertEqual(group["auto_job"]["stage"], "gpu_scheduler")
        self.assertTrue(any(track["key"] == "ablation" and track["enabled"] for track in group["planned_tracks"]))
        # Server filesystem layout must never reach the browser.
        for private_key in ("workspace_root", "experiment_root", "plan_root", "paper_root"):
            self.assertNotIn(private_key, group)
        self.assertIn("latest_status", group["plan_snapshot"])
        self.assertIn("/papers/1", group["paper_preview_urls"]["index"])
        self.assertIn("agenda_id=1", group["paper_preview_urls"]["index"])

    def test_broken_workspace_degrades_listing_instead_of_500(self):
        # A dangling `current` symlink (or any OSError) on one idea's workspace
        # must not take down the whole experiment_groups listing.
        with mock.patch.object(
            web_app,
            "get_idea_workspace",
            side_effect=FileExistsError("File exists: .../current"),
        ):
            response = self.client.get("/api/experiment_groups?agenda_id=1")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["paper_assets"], [])
        self.assertEqual(payload[0]["paper_preview_urls"], {})

    def test_api_experiment_group_detail_includes_run_history_and_artifacts(self):
        response = self.client.get("/api/experiment_groups/1?agenda_id=1")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["insight"]["id"], 1)
        self.assertEqual(len(payload["runs"]), 2)
        active_run = payload["runs"][0]
        self.assertEqual(active_run["id"], 11)
        self.assertTrue(active_run["has_plot_artifacts"])
        historical_run = next(run for run in payload["runs"] if run["id"] == 10)
        self.assertEqual(historical_run["claim_count"], 1)

    def test_paper_preview_routes_serve_current_tex(self):
        index_response = self.client.get("/papers/1?agenda_id=1")
        tex_response = self.client.get("/papers/1/tex?agenda_id=1")

        self.assertEqual(index_response.status_code, 200)
        self.assertIn("Idea 1", index_response.get_data(as_text=True))
        self.assertEqual(tex_response.status_code, 200)
        self.assertIn("\\documentclass", tex_response.get_data(as_text=True))
        tex_response.close()


class AgendaScopeApiTests(unittest.TestCase):
    """Cross-agenda reads: the front-page counters and the lists must agree.

    The schema comes from database.init_db() rather than hand-rolled CREATE
    TABLE statements on purpose. Fixtures that build their own schema drift from
    the migrations, and that drift is what let a production NotNullViolation
    survive a green suite on 2026-08-17.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.old_db_path = database.DB_PATH
        self.old_database_url = database.DATABASE_URL
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(database._local, attr):
                try:
                    getattr(database._local, attr).close()
                except Exception:
                    pass
                setattr(database._local, attr, None)
        database.DATABASE_URL = ""
        database.DB_PATH = Path(self.tmpdir.name) / "test.db"
        database.init_db()
        # The V1 agenda column exists in PostgreSQL via the meta-harness
        # migration; SQLite's init_db() predates it.
        for table in ("deep_insights", "experiment_runs", "manuscript_runs"):
            rows = database.fetchall(f"PRAGMA table_info({table})")
            if not rows:
                continue
            if "agenda_id" not in {row["name"] for row in rows}:
                database.execute(f"ALTER TABLE {table} ADD COLUMN agenda_id INTEGER")
        # Two agendas, so "all" has to mean more than "the one we asked for".
        for insight_id, agenda_id, tier in ((1, 7, 1), (2, 9, 2)):
            database.execute(
                "INSERT INTO deep_insights (id, agenda_id, tier, title, status)"
                " VALUES (?, ?, ?, ?, 'candidate')",
                (insight_id, agenda_id, tier, f"Idea {insight_id} on agenda {agenda_id}"),
            )
        database.commit()
        self.client = web_app.app.test_client()

    def tearDown(self):
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(database._local, attr):
                try:
                    getattr(database._local, attr).close()
                except Exception:
                    pass
                setattr(database._local, attr, None)
        database.DB_PATH = self.old_db_path
        database.DATABASE_URL = self.old_database_url
        self.tmpdir.cleanup()

    def _ids(self, query: str) -> set:
        response = self.client.get(query)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return {int(row["id"]) for row in response.get_json()}

    def test_omitted_scope_returns_every_agenda(self):
        self.assertEqual(self._ids("/api/deep_insights"), {1, 2})

    def test_explicit_all_forms_match_the_omitted_form(self):
        for value in ("all", "0", ""):
            with self.subTest(agenda_id=value):
                self.assertEqual(self._ids(f"/api/deep_insights?agenda_id={value}"), {1, 2})

    def test_named_agenda_still_narrows(self):
        self.assertEqual(self._ids("/api/deep_insights?agenda_id=7"), {1})
        self.assertEqual(self._ids("/api/deep_insights?agenda_id=9"), {2})

    def test_tier_filter_spans_agendas_when_unscoped(self):
        """Tier 1 lived only on older agendas, so the filter looked broken."""
        self.assertEqual(self._ids("/api/deep_insights?tier=1"), {1})
        self.assertEqual(self._ids("/api/deep_insights?tier=2"), {2})
        self.assertEqual(self._ids("/api/deep_insights?tier=1&agenda_id=9"), set())

    def test_detail_by_id_no_longer_needs_the_owning_agenda(self):
        response = self.client.get("/api/deep_insights/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["agenda_id"], 7)

    def test_detail_still_respects_an_explicit_mismatched_scope(self):
        self.assertEqual(self.client.get("/api/deep_insights/1?agenda_id=9").status_code, 404)

    def _require_table(self, name: str):
        """Skip rather than hand-roll a table the migrations own.

        scripts/meta_harness_migration.py is PostgreSQL-only, so init_db() on
        SQLite has no meta-harness v1 schema. Re-creating those tables here by
        hand is exactly the drift that hid a production NotNullViolation, so
        these two cases skip instead.
        """
        rows = database.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        )
        if not rows:
            self.skipTest(f"{name} is owned by the PostgreSQL-only meta-harness migration")

    def test_scientific_decisions_are_listable(self):
        self._require_table("scientific_decision_records")
        response = self.client.get("/api/scientific_decisions")
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIn("decisions", payload)
        self.assertIn("counts_by_verdict", payload)
        self.assertEqual(payload["agenda_scope"], "all")

    def test_manuscripts_list_spans_agendas_when_unscoped(self):
        self._require_table("manuscript_runs")
        response = self.client.get("/api/manuscripts")
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        self.assertIsInstance(response.get_json(), list)


if __name__ == "__main__":
    unittest.main()
