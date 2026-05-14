"""Smoke tests for web/manuscript_routes.py (issue #15 / D4 API).

Coverage:
1. GET  /api/manuscript/venues          → all 6 adapters with column_layout/max_pages
2. POST /api/manuscript/route           → CV state picks cvpr2024
3. POST /api/manuscript/route + tiebreak → returns needs_tiebreak + tiebreak dict
4. POST /api/manuscript/lint  (happy)   → pass=True on normalised ICLR source
5. POST /api/manuscript/lint  (bad)     → pass=False on missing-documentclass source
6. POST /api/manuscript/lint/<sel_id>   → persists; GET /lint_run/<id> reads back
7. Bad template_id → 404 with registered list in body
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DEEPGRAPH_DATABASE_URL", "")


class ManuscriptRoutesSmokeTests(unittest.TestCase):
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

    # ------------------------------------------------------------------
    def test_get_venues_returns_all_six_with_column_layout(self):
        rv = self.client.get("/api/manuscript/venues")
        self.assertEqual(rv.status_code, 200)
        venues = rv.get_json()["venues"]
        ids = {v["template_id"] for v in venues}
        self.assertSetEqual(
            ids,
            {"iclr2026", "arxiv_plain", "neurips2024", "icml2024", "acl_arr", "cvpr2024"},
        )
        for v in venues:
            self.assertIn(v["column_layout"], {"single_column", "two_column"})
            self.assertGreater(v["max_pages"], 0)

    def test_preview_route_cv_state_picks_cvpr2024(self):
        state = {
            "title": "Diffusion-based image detection at scale",
            "claim_type": "empirical",
            "domain": "vision",
            "has_real_data": True,
            "tier": 1,
            "page_count_estimate": 8,
        }
        rv = self.client.post("/api/manuscript/route", json=state)
        self.assertEqual(rv.status_code, 200)
        body = rv.get_json()
        self.assertEqual(body["selected"]["template_id"], "cvpr2024")
        self.assertTrue(any(s["template_id"] == "iclr2026" for s in body["all_scored"]))

    def test_preview_route_with_tiebreak_flag_returns_tiebreak_block(self):
        state = {
            "title": "Stochastic optimization for deep learning generalization",
            "claim_type": "empirical",
            "domain": "ml",
            "has_real_data": True,
            "tier": 1,
            "page_count_estimate": 9,
            "include_tiebreak": True,
        }
        rv = self.client.post("/api/manuscript/route", json=state)
        self.assertEqual(rv.status_code, 200)
        body = rv.get_json()
        self.assertIn("needs_tiebreak", body)
        self.assertIn("tiebreak", body)
        self.assertIn("chosen_template_id", body["tiebreak"])

    def test_preview_lint_happy_path(self):
        source = (
            r"\documentclass{article}"
            "\n\\begin{document}\nbody\n\\bibliography{refs}\n\\end{document}\n"
        )
        rv = self.client.post(
            "/api/manuscript/lint",
            json={"template_id": "iclr2026", "source": source, "page_count": 8},
        )
        self.assertEqual(rv.status_code, 200)
        body = rv.get_json()
        self.assertTrue(body["lint"]["pass"], body)
        self.assertEqual(len(body["lint"]["checks"]), 7)

    def test_preview_lint_missing_documentclass_fails(self):
        source = r"\begin{document}body\end{document}"
        rv = self.client.post(
            "/api/manuscript/lint",
            json={"template_id": "iclr2026", "source": source, "normalize": False},
        )
        self.assertEqual(rv.status_code, 200)
        body = rv.get_json()
        self.assertFalse(body["lint"]["pass"])

    def test_persist_lint_and_readback_via_lint_run_endpoint(self):
        source = (
            r"\documentclass{article}\usepackage{graphicx}\usepackage{amsmath}"
            r"\usepackage{hyperref}\begin{document}body"
            r"\bibliographystyle{iclr2026_conference}\bibliography{refs}\end{document}"
        )
        rv = self.client.post(
            "/api/manuscript/lint/99999",
            json={"template_id": "iclr2026", "source": source, "page_count": 8, "normalize": False},
        )
        self.assertEqual(rv.status_code, 201, rv.get_json())
        body = rv.get_json()
        run_id = body["run_id"]
        self.assertIsInstance(run_id, int)
        rv2 = self.client.get(f"/api/manuscript/lint_run/{run_id}")
        self.assertEqual(rv2.status_code, 200)
        body2 = rv2.get_json()
        self.assertEqual(body2["lint_run"]["template_id"], "iclr2026")
        self.assertEqual(body2["lint_run"]["selection_id"], 99999)

    def test_unknown_template_id_returns_404_with_registered_list(self):
        rv = self.client.post(
            "/api/manuscript/lint",
            json={"template_id": "nope_2099", "source": "x"},
        )
        self.assertEqual(rv.status_code, 404)
        body = rv.get_json()
        self.assertEqual(body["error"], "unknown_template_id")
        self.assertIn("iclr2026", body["registered"])


if __name__ == "__main__":
    unittest.main()
