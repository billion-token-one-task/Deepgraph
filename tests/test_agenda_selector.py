"""Tests for agents.agenda_selector (issue #9 selection layer)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DEEPGRAPH_DATABASE_URL", "")


SAMPLE_AGENDA_DICT = {
    "version": "v1",
    "name": "token_scale_v1",
    "description": "demo",
    "focus": ["sub-quadratic attention", "long context", "scaling laws"],
    "prefer": {
        "keywords": ["linear attention", "state space model"],
        "tiers": ["tier_1", "tier_2"],
        "paradigm_score_min": 0.3,
        "resource_class": ["cpu", "gpu_small"],
    },
    "reject": {
        "keywords": ["closed-source dataset"],
        "statuses": ["rejected", "failed"],
        "novelty_status": ["duplicate", "exists"],
    },
    "required_output": ["experiment_plan", "manuscript_bundle"],
}


def _make_insight(**overrides):
    base = {
        "id": 1,
        "tier": 2,
        "status": "verified",
        "title": "Linear attention for long context",
        "problem_statement": "Quadratic attention is too expensive for long sequences.",
        "formal_structure": "",
        "proposed_method": '{"name": "LinAttn", "type": "architecture"}',
        "adversarial_score": 7.5,
        "novelty_status": "novel",
        "resource_class": "gpu_small",
        "experimentability": "easy",
    }
    base.update(overrides)
    return base


class AgendaScoringTests(unittest.TestCase):
    def test_focus_keyword_matches_boost_score(self):
        from agents.agenda_loader import parse_agenda
        from agents.agenda_selector import score_insight

        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        insight = _make_insight()
        breakdown = score_insight(agenda, insight)
        self.assertFalse(breakdown["blocked"])
        self.assertGreater(breakdown["score"], 0.4)
        # title contains "long context" and "linear attention"
        self.assertIn("long context", breakdown["matched_focus"])
        self.assertIn("linear attention", breakdown["matched_prefer_keywords"])

    def test_reject_keyword_blocks(self):
        from agents.agenda_loader import parse_agenda
        from agents.agenda_selector import score_insight

        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        insight = _make_insight(
            problem_statement="Uses closed-source dataset only.",
        )
        breakdown = score_insight(agenda, insight)
        self.assertTrue(breakdown["blocked"])
        self.assertTrue(any("closed-source dataset" in r for r in breakdown["block_reasons"]))

    def test_reject_status_blocks(self):
        from agents.agenda_loader import parse_agenda
        from agents.agenda_selector import score_insight

        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        insight = _make_insight(status="failed")
        breakdown = score_insight(agenda, insight)
        self.assertTrue(breakdown["blocked"])

    def test_paradigm_score_gate(self):
        from agents.agenda_loader import parse_agenda
        from agents.agenda_selector import score_insight

        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        insight = _make_insight(adversarial_score=1.0)  # paradigm=0.1 below 0.3 min
        breakdown = score_insight(agenda, insight)
        self.assertTrue(breakdown["blocked"])
        self.assertTrue(any("paradigm_score" in r for r in breakdown["block_reasons"]))


class AgendaCandidateEvaluationTests(unittest.TestCase):
    def test_evaluate_picks_highest_eligible(self):
        from agents.agenda_loader import parse_agenda
        from agents.agenda_selector import evaluate_candidates

        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        candidates = [
            _make_insight(id=1, title="Linear attention scaling laws", adversarial_score=8.5),
            _make_insight(id=2, title="Random unrelated CV paper", problem_statement="image segmentation"),
            _make_insight(id=3, status="failed", title="closed-source"),  # blocked
        ]
        result = evaluate_candidates(agenda, candidates)
        self.assertIsNotNone(result["selected"])
        self.assertEqual(result["selected"]["insight"]["id"], 1)
        # at least the blocked one should appear in rejected
        rejected_ids = {r["insight_id"] for r in result["rejected"]}
        self.assertIn(3, rejected_ids)

    def test_evaluate_all_blocked_returns_none(self):
        from agents.agenda_loader import parse_agenda
        from agents.agenda_selector import evaluate_candidates

        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        candidates = [
            _make_insight(id=10, status="rejected"),
            _make_insight(id=11, novelty_status="exists"),
        ]
        result = evaluate_candidates(agenda, candidates)
        self.assertIsNone(result["selected"])
        self.assertEqual(len(result["rejected"]), 2)


class AgendaPersistenceFlowTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["DEEPGRAPH_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        from db import database as db

        for attr in ("sqlite_conn", "pg_conn", "conn"):
            if hasattr(db._local, attr):
                try:
                    getattr(db._local, attr).close()
                except Exception:
                    pass
                delattr(db._local, attr)
        db.DB_PATH = Path(os.environ["DEEPGRAPH_DB_PATH"])
        db.init_db()

    def tearDown(self):
        from db import database as db

        for attr in ("sqlite_conn", "pg_conn", "conn"):
            if hasattr(db._local, attr):
                try:
                    getattr(db._local, attr).close()
                except Exception:
                    pass
                delattr(db._local, attr)
        self._tmpdir.cleanup()
        os.environ.pop("DEEPGRAPH_DB_PATH", None)

    def _seed_insight(self, **overrides):
        from db import database as db

        row = _make_insight(**overrides)
        db.execute(
            """
            INSERT INTO deep_insights
                (id, tier, status, title, problem_statement, formal_structure,
                 proposed_method, adversarial_score, novelty_status, resource_class,
                 experimentability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["tier"],
                row["status"],
                row["title"],
                row["problem_statement"],
                row["formal_structure"],
                row["proposed_method"],
                row["adversarial_score"],
                row["novelty_status"],
                row["resource_class"],
                row["experimentability"],
            ),
        )
        db.commit()

    def test_select_and_persist_writes_row(self):
        from agents.agenda_loader import parse_agenda, save_agenda
        from agents.agenda_selector import get_latest_selection, select_and_persist

        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        save_agenda(agenda)
        self._seed_insight(id=1, title="Linear attention for long context")
        self._seed_insight(id=2, title="CNN on ImageNet", problem_statement="image classification")
        self._seed_insight(id=3, status="failed", title="something")

        sel = select_and_persist(agenda)
        self.assertEqual(sel.selected_insight_id, 1)
        self.assertEqual(sel.status, "pending")
        self.assertIsNotNone(sel.selection_id)

        latest = get_latest_selection(agenda.agenda_id)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["selected_insight_id"], 1)
        # rejected list should mention insight 3 (blocked) and possibly 2 (lower score)
        rejected_ids = {r["insight_id"] for r in latest["rejected_candidates"]}
        self.assertIn(3, rejected_ids)

    def test_empty_pool_creates_blocked_selection(self):
        from agents.agenda_loader import parse_agenda, save_agenda
        from agents.agenda_selector import select_and_persist

        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        save_agenda(agenda)
        sel = select_and_persist(agenda)
        self.assertEqual(sel.status, "blocked")
        self.assertIsNone(sel.selected_insight_id)

    def test_update_selection_progress(self):
        from agents.agenda_loader import parse_agenda, save_agenda
        from agents.agenda_selector import (
            get_selection,
            select_and_persist,
            update_selection_progress,
        )

        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        save_agenda(agenda)
        self._seed_insight(id=1, title="Linear attention long context", adversarial_score=8.5)
        sel = select_and_persist(agenda)

        update_selection_progress(
            sel.selection_id,
            status="launched",
            auto_research_job_id=42,
        )
        row = get_selection(sel.selection_id)
        self.assertEqual(row["status"], "launched")
        self.assertEqual(row["auto_research_job_id"], 42)


if __name__ == "__main__":
    unittest.main()
