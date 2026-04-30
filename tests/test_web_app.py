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
        self.assertIn("target", payload["database"])

    def test_manual_post_api_returns_gone_in_fixed_flow_mode(self):
        response = self.client.post("/api/experiments/run_full", json={"insight_id": 1})
        payload = response.get_json()

        self.assertEqual(response.status_code, 410)
        self.assertEqual(payload["mode"], "fixed_flow_read_only")
        self.assertIn("removed", payload["error"].lower())


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
        self.workspace_root = Path(self.tmpdir.name) / "ideas"
        self.workspace_patch = mock.patch.object(workspace_layout, "IDEA_WORKSPACE_DIR", self.workspace_root)
        self.workspace_patch.start()
        self.client = web_app.app.test_client()

        database.execute(
            """
            INSERT INTO deep_insights
            (id, tier, title, submission_status, evidence_plan, experimental_plan)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
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
            (deep_insight_id, status, stage, last_note)
            VALUES (?, ?, ?, ?)
            """,
            (1, "running_gpu", "gpu_scheduler", "Main run still progressing"),
        )
        database.execute(
            """
            INSERT INTO experiment_runs
            (id, deep_insight_id, status, hypothesis_verdict, effect_pct, iterations_total, iterations_kept, workdir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (10, 1, "completed", "confirmed", 12.5, 8, 3, str(self.workspace_root / "legacy_run_10")),
        )
        database.execute(
            """
            INSERT INTO experiment_runs
            (id, deep_insight_id, status, iterations_total, iterations_kept, workdir)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (11, 1, "testing", 2, 0, str(self.workspace_root / "legacy_run_11")),
        )
        database.execute(
            "INSERT INTO experiment_artifacts (run_id, artifact_type, path) VALUES (?, ?, ?)",
            (11, "plot", "/tmp/plot.svg"),
        )
        database.execute(
            "INSERT INTO experimental_claims (run_id, deep_insight_id, claim_text, verdict) VALUES (?, ?, ?, ?)",
            (10, 1, "Improves metric", "confirmed"),
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
        response = self.client.get("/api/experiment_groups")
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
        self.assertTrue(group["workspace_root"].endswith("idea_1"))
        self.assertIn("latest_status", group["plan_snapshot"])
        self.assertTrue(group["paper_preview_urls"]["index"].endswith("/papers/1"))

    def test_api_experiment_group_detail_includes_run_history_and_artifacts(self):
        response = self.client.get("/api/experiment_groups/1")
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
        index_response = self.client.get("/papers/1")
        tex_response = self.client.get("/papers/1/tex")

        self.assertEqual(index_response.status_code, 200)
        self.assertIn("Idea 1", index_response.get_data(as_text=True))
        self.assertEqual(tex_response.status_code, 200)
        self.assertIn("\\documentclass", tex_response.get_data(as_text=True))
        tex_response.close()


if __name__ == "__main__":
    unittest.main()
