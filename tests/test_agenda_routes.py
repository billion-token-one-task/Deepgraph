"""Smoke tests for web/agenda_routes.py (issue #9 step 6 API).

Covers the 9 endpoints end-to-end via Flask test_client, seeding the database
with a confirmed-verdict insight + experiment_run + bundle so that the full
loop (select -> link -> review -> plan -> loop snapshot) produces non-trivial
output.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DEEPGRAPH_DATABASE_URL", "")


SAMPLE_AGENDA_DICT = {
    "version": "v1",
    "name": "api_route_test_agenda",
    "focus": ["long context"],
    "prefer": {"keywords": ["linear attention"], "tiers": ["tier_2"]},
    "reject": {"keywords": ["closed-source dataset"]},
}


class AgendaRoutesSmokeTests(unittest.TestCase):
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

        from web import app as app_module
        self.app = app_module.app
        self.client = self.app.test_client()

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

    def _seed_full_run(self):
        """Seed one insight + experiment_run with verdict=confirmed + bundle."""
        self.db.execute(
            """
            INSERT INTO deep_insights
                (id, tier, status, title, problem_statement, adversarial_score,
                 novelty_status, resource_class, experimentability, evidence_plan)
            VALUES (1, 2, 'verified', 'Linear attention for long context',
                    'Quadratic attention is expensive.', 7.5, 'novel', 'gpu_small',
                    'easy',
                    '{"main_table": {"enabled": true, "priority": "required"}}')
            """
        )
        cur = self.db.execute(
            """
            INSERT INTO experiment_runs
                (deep_insight_id, status, phase, hypothesis_verdict,
                 baseline_metric_name, baseline_metric_value, best_metric_value,
                 effect_size, effect_pct, workdir)
            VALUES (1, 'completed', 'hypothesis_testing', 'confirmed',
                    'accuracy', 0.65, 0.78, 0.13, 20.0, '/tmp/exp')
            """
        )
        exp_id = cur.lastrowid
        self.db.execute(
            """
            INSERT INTO experimental_claims
                (run_id, deep_insight_id, claim_text, claim_type, verdict,
                 effect_size, confidence, p_value)
            VALUES (?, 1, 'Linear attention improves accuracy by 20% on long context.',
                    'experimental', 'confirmed', 0.13, 0.92, 0.01)
            """,
            (exp_id,),
        )
        cur = self.db.execute(
            """
            INSERT INTO manuscript_runs
                (experiment_run_id, deep_insight_id, status, workdir)
            VALUES (?, 1, 'bundle_ready', '/tmp/manu')
            """,
            (exp_id,),
        )
        manu_id = cur.lastrowid
        cur = self.db.execute(
            """
            INSERT INTO submission_bundles
                (manuscript_run_id, bundle_format, status, bundle_path, manifest_path)
            VALUES (?, 'conference', 'ready', '/tmp/bundle.zip', '/tmp/manifest.json')
            """,
            (manu_id,),
        )
        bundle_id = cur.lastrowid
        self.db.execute(
            "UPDATE experiment_runs SET submission_bundle_id=? WHERE id=?",
            (bundle_id, exp_id),
        )
        self.db.commit()

    # ---------- upload + list + current ----------

    def test_get_empty_returns_empty_list(self):
        r = self.client.get("/api/research_agenda")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), {"agendas": []})

    def test_get_current_returns_404_when_empty(self):
        r = self.client.get("/api/research_agenda/current")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.get_json(), {"agenda": None})

    def test_upload_yaml_body(self):
        yaml_body = (
            "version: v1\n"
            "name: smoke_yaml\n"
            "focus: [long context]\n"
            "prefer:\n"
            "  keywords: [linear attention]\n"
        )
        r = self.client.post(
            "/api/research_agenda",
            data=yaml_body,
            content_type="application/x-yaml",
        )
        self.assertEqual(r.status_code, 201)
        body = r.get_json()
        self.assertEqual(body["agenda"]["name"], "smoke_yaml")
        self.assertTrue(body["agenda"]["is_active"])
        self.assertEqual(body["agenda"]["focus"], ["long context"])

    def test_upload_json_body(self):
        r = self.client.post(
            "/api/research_agenda",
            data=json.dumps(SAMPLE_AGENDA_DICT),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 201)
        agenda = r.get_json()["agenda"]
        self.assertEqual(agenda["name"], "api_route_test_agenda")
        self.assertEqual(agenda["reject"], {"keywords": ["closed-source dataset"]})

    def test_upload_invalid_agenda_returns_400(self):
        # focus and prefer both empty -> contract violation
        r = self.client.post(
            "/api/research_agenda",
            data=json.dumps({"version": "v1", "name": "bad"}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"], "invalid_agenda")

    def test_list_after_upload(self):
        self.client.post(
            "/api/research_agenda",
            data=json.dumps(SAMPLE_AGENDA_DICT),
            content_type="application/json",
        )
        r = self.client.get("/api/research_agenda")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.get_json()["agendas"]), 1)

        r = self.client.get("/api/research_agenda/current")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["agenda"]["name"], "api_route_test_agenda")

    # ---------- select ----------

    def test_select_without_agenda_returns_404(self):
        r = self.client.post("/api/research_agenda/select", json={})
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.get_json()["error"], "no_active_agenda")

    def test_select_invalid_dispatch_mode_returns_400(self):
        self.client.post(
            "/api/research_agenda",
            data=json.dumps(SAMPLE_AGENDA_DICT),
            content_type="application/json",
        )
        r = self.client.post(
            "/api/research_agenda/select",
            json={"dispatch_mode": "wat"},
        )
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"], "invalid_dispatch_mode")

    def test_select_dispatch_none_returns_selection(self):
        self.client.post(
            "/api/research_agenda",
            data=json.dumps(SAMPLE_AGENDA_DICT),
            content_type="application/json",
        )
        self._seed_full_run()
        r = self.client.post(
            "/api/research_agenda/select",
            json={"dispatch_mode": "none"},
        )
        self.assertEqual(r.status_code, 201)
        body = r.get_json()
        self.assertIsNotNone(body["selection"])
        self.assertEqual(body["selection"]["selected_insight_id"], 1)
        # dispatch_mode=none -> dispatch field must be None
        self.assertIsNone(body["dispatch"])

    # ---------- full loop ----------

    def test_full_loop_end_to_end(self):
        # 1) upload agenda
        self.client.post(
            "/api/research_agenda",
            data=json.dumps(SAMPLE_AGENDA_DICT),
            content_type="application/json",
        )
        self._seed_full_run()

        # 2) select + dispatch=auto -> should link existing artifacts
        r = self.client.post(
            "/api/research_agenda/select",
            json={"dispatch_mode": "auto"},
        )
        self.assertEqual(r.status_code, 201)
        body = r.get_json()
        sel = body["selection"]
        self.assertEqual(sel["selected_insight_id"], 1)
        self.assertEqual(sel["status"], "completed")
        self.assertIsNotNone(sel["submission_bundle_id"])
        # dispatch succeeded -> dispatch_succeeded must be True
        self.assertTrue(body["dispatch_succeeded"])
        sid = sel["id"]

        # 3) GET single selection
        r = self.client.get(f"/api/research_agenda/selection/{sid}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["selection"]["id"], sid)

        # 4) GET latest selection
        r = self.client.get("/api/research_agenda/selection/latest")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["selection"]["id"], sid)

        # 5) run review
        r = self.client.post(
            f"/api/research_agenda/selection/{sid}/review",
            json={},
        )
        self.assertEqual(r.status_code, 201)
        review = r.get_json()["review"]
        self.assertIn(review["recommendation"],
                      ("accept", "minor_revision", "major_revision", "reject"))
        self.assertEqual(review["reviewer"], "internal_evidence_gate")

        # 6) build revision plan
        r = self.client.post(
            f"/api/research_agenda/selection/{sid}/plan",
            json={"review_id": review["id"]},
        )
        self.assertEqual(r.status_code, 201)
        plan = r.get_json()["plan"]
        self.assertIn(plan["status"], ("proposed", "noop"))

        # 7) loop inspection
        r = self.client.get(f"/api/research_agenda/loop/{sid}")
        self.assertEqual(r.status_code, 200)
        loop = r.get_json()["loop"]
        self.assertEqual(loop["selection"]["id"], sid)
        self.assertEqual(loop["insight"]["id"], 1)
        self.assertEqual(loop["experiment_run"]["hypothesis_verdict"], "confirmed")
        self.assertEqual(loop["submission_bundle"]["status"], "ready")
        self.assertEqual(loop["review"]["id"], review["id"])
        self.assertEqual(loop["revision_plan"]["id"], plan["id"])

    def test_plan_without_explicit_review_id_uses_latest(self):
        self.client.post(
            "/api/research_agenda",
            data=json.dumps(SAMPLE_AGENDA_DICT),
            content_type="application/json",
        )
        self._seed_full_run()
        r = self.client.post(
            "/api/research_agenda/select", json={"dispatch_mode": "auto"}
        )
        sid = r.get_json()["selection"]["id"]
        self.client.post(f"/api/research_agenda/selection/{sid}/review", json={})

        # call plan without review_id -> should auto-pick latest
        r = self.client.post(
            f"/api/research_agenda/selection/{sid}/plan", json={}
        )
        self.assertEqual(r.status_code, 201)

    def test_plan_without_any_review_returns_400(self):
        self.client.post(
            "/api/research_agenda",
            data=json.dumps(SAMPLE_AGENDA_DICT),
            content_type="application/json",
        )
        self._seed_full_run()
        r = self.client.post(
            "/api/research_agenda/select", json={"dispatch_mode": "auto"}
        )
        sid = r.get_json()["selection"]["id"]
        r = self.client.post(
            f"/api/research_agenda/selection/{sid}/plan", json={}
        )
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"], "no_review")

    def test_selection_not_found_returns_404(self):
        r = self.client.get("/api/research_agenda/selection/9999")
        self.assertEqual(r.status_code, 404)
        r = self.client.get("/api/research_agenda/loop/9999")
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
