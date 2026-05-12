"""HTTP smoke tests for the new agenda routes (bench + gate inspection).

Covers:
    POST /api/research_agenda/selection/<id>/bench
    POST /api/research_agenda/selection/<id>/gate
    GET  /api/research_agenda/selection/<id>/gate/latest
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DEEPGRAPH_DATABASE_URL", "")


SAMPLE_AGENDA_DICT = {
    "version": "v1",
    "name": "bench_route_test_agenda",
    "focus": ["long context"],
    "prefer": {"keywords": ["linear attention"], "tiers": ["tier_2"]},
    "reject": {},
}


def _reset_db_locals(db):
    for attr in ("sqlite_conn", "pg_conn", "conn"):
        if hasattr(db._local, attr):
            try:
                getattr(db._local, attr).close()
            except Exception:
                pass
            delattr(db._local, attr)


class BenchAndGateRoutesTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["DEEPGRAPH_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        from db import database as db
        _reset_db_locals(db)
        db.DB_PATH = Path(os.environ["DEEPGRAPH_DB_PATH"])
        db.init_db()
        self.db = db
        from web import app as app_module
        self.client = app_module.app.test_client()

    def tearDown(self):
        from db import database as db
        _reset_db_locals(db)
        self._tmpdir.cleanup()
        os.environ.pop("DEEPGRAPH_DB_PATH", None)

    def _seed_insight(self):
        self.db.execute(
            """
            INSERT INTO deep_insights
                (id, tier, status, title, problem_statement, adversarial_score,
                 novelty_status, resource_class, experimentability)
            VALUES (1, 2, 'verified', 'Linear attention long context',
                    'Quadratic attention is too expensive.', 8.5,
                    'novel', 'gpu_small', 'easy')
            """,
        )
        self.db.commit()

    def _new_selection(self):
        from agents.agenda_loader import parse_agenda, save_agenda
        from agents.agenda_selector import select_and_persist
        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        save_agenda(agenda)
        return select_and_persist(agenda)

    def test_bench_endpoint_runs_full_pipeline(self):
        self._seed_insight()
        sel = self._new_selection()
        resp = self.client.post(
            f"/api/research_agenda/selection/{sel.selection_id}/bench",
            json={"seq_len": 128, "head_dim": 32, "seed": 1729, "repeats": 2},
        )
        self.assertEqual(resp.status_code, 201, resp.get_data(as_text=True))
        body = resp.get_json()
        self.assertEqual(body["selection_id"], sel.selection_id)
        self.assertEqual(body["evidence_gate"]["status"], "pass")
        self.assertTrue(body["manuscript_created"])
        self.assertIn("packet_path", body["experiment_result"])

    def test_gate_endpoints_pass_and_fetch(self):
        self._seed_insight()
        sel = self._new_selection()

        # 1) bench runs gate as side-effect
        self.client.post(
            f"/api/research_agenda/selection/{sel.selection_id}/bench",
            json={"seq_len": 128, "head_dim": 32, "seed": 1729, "repeats": 2},
        )

        # 2) explicit re-run of gate via dedicated endpoint
        resp = self.client.post(
            f"/api/research_agenda/selection/{sel.selection_id}/gate"
        )
        self.assertEqual(resp.status_code, 201)
        gate = resp.get_json()["gate"]
        self.assertEqual(gate["status"], "pass")

        # 3) latest fetch
        resp = self.client.get(
            f"/api/research_agenda/selection/{sel.selection_id}/gate/latest"
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["gate"]["status"], "pass")

    def test_gate_endpoint_blocks_when_no_run(self):
        self._seed_insight()
        sel = self._new_selection()
        resp = self.client.post(
            f"/api/research_agenda/selection/{sel.selection_id}/gate"
        )
        self.assertEqual(resp.status_code, 201)
        gate = resp.get_json()["gate"]
        self.assertEqual(gate["status"], "block")
        reqs = [b["requirement"] for b in gate["blockers"]]
        self.assertIn("experiment_run", reqs)


if __name__ == "__main__":
    unittest.main()
