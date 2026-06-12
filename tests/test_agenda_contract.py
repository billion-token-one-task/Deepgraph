"""Tests for the agenda configuration layer (issue #9).

Covers:
- ResearchAgenda contract parsing + validation (required fields, focus/prefer rule)
- AgendaSelection / AgendaReview / AgendaRevisionPlan contract validation
- YAML loader produces an equivalent agenda to the dict loader
- Persistence round-trip via agents.agenda_loader (insert → fetch → list → active toggle)
- Schema is loaded (tables exist after init_db)
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

# Force SQLite for tests
os.environ["DEEPGRAPH_DATABASE_URL"] = ""  # force SQLite tmpdir; never touch a real DB from the environment

from contracts.agenda import (  # noqa: E402
    AgendaReview,
    AgendaRevisionPlan,
    AgendaSelection,
    ResearchAgenda,
)
from contracts.base import ContractValidationError  # noqa: E402


SAMPLE_AGENDA_DICT = {
    "version": "v1",
    "name": "token_scale_v1",
    "description": "demo",
    "focus": ["sub-quadratic attention", "long context"],
    "prefer": {
        "keywords": ["linear attention"],
        "tiers": ["tier_1", "tier_2"],
        "paradigm_score_min": 0.45,
    },
    "reject": {
        "keywords": ["closed-source dataset"],
        "statuses": ["rejected"],
    },
    "required_output": ["experiment_plan", "manuscript_bundle"],
}


class ResearchAgendaContractTests(unittest.TestCase):
    def test_parse_valid_agenda_dict(self):
        from agents.agenda_loader import parse_agenda

        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        self.assertEqual(agenda.name, "token_scale_v1")
        self.assertEqual(agenda.version, "v1")
        self.assertIn("sub-quadratic attention", agenda.focus)
        self.assertIn("keywords", agenda.prefer)
        self.assertEqual(agenda.prefer["paradigm_score_min"], 0.45)
        self.assertIn("rejected", agenda.reject["statuses"])

    def test_agenda_requires_name(self):
        from agents.agenda_loader import parse_agenda

        bad = dict(SAMPLE_AGENDA_DICT, name="")
        with self.assertRaises(ContractValidationError):
            parse_agenda(bad)

    def test_agenda_requires_focus_or_prefer(self):
        with self.assertRaises(ContractValidationError):
            ResearchAgenda(
                name="empty_agenda",
                focus=[],
                prefer={},
            ).validate()

    def test_yaml_loader_matches_dict_loader(self):
        from agents.agenda_loader import load_agenda_from_file, parse_agenda

        repo_root = Path(__file__).resolve().parent.parent
        yaml_path = repo_root / "research_agendas" / "token_scale_v1.yaml"
        if not yaml_path.exists():
            self.skipTest("sample agenda yaml missing")
        try:
            agenda_yaml = load_agenda_from_file(yaml_path)
        except RuntimeError as exc:
            self.skipTest(f"yaml loader unavailable: {exc}")

        self.assertEqual(agenda_yaml.name, "token_scale_efficiency_v1")
        self.assertEqual(agenda_yaml.version, "v1")
        # Same shape after parse_agenda on the same raw_config dict
        agenda_dict = parse_agenda(agenda_yaml.raw_config)
        self.assertEqual(set(agenda_yaml.focus), set(agenda_dict.focus))


class AgendaSelectionContractTests(unittest.TestCase):
    def test_valid_selection(self):
        sel = AgendaSelection(
            agenda_id=1,
            selected_insight_id=13,
            score=0.82,
            rationale="matches focus on long context",
            rejected_candidates=[{"insight_id": 9, "score": 0.21, "reason": "off focus"}],
            status="pending",
        )
        sel.validate()
        self.assertEqual(sel.status, "pending")

    def test_invalid_status_rejected(self):
        sel = AgendaSelection(agenda_id=1, status="wat")
        with self.assertRaises(ContractValidationError):
            sel.validate()

    def test_missing_agenda_id_rejected(self):
        with self.assertRaises(ContractValidationError):
            AgendaSelection(agenda_id=0).validate()


class AgendaReviewContractTests(unittest.TestCase):
    def test_valid_review(self):
        review = AgendaReview(
            selection_id=7,
            submission_bundle_id=3,
            reviewer="internal_evidence_gate",
            recommendation="minor_revision",
            confidence=0.7,
            strengths=["clear motivation"],
            weaknesses=["needs more baselines"],
            required_revisions=["add baseline X"],
        )
        review.validate()
        self.assertEqual(review.recommendation, "minor_revision")

    def test_invalid_recommendation(self):
        review = AgendaReview(selection_id=7, recommendation="lgtm")
        with self.assertRaises(ContractValidationError):
            review.validate()


class AgendaRevisionPlanContractTests(unittest.TestCase):
    def test_valid_plan(self):
        plan = AgendaRevisionPlan(
            selection_id=7,
            review_id=2,
            rationale="add baseline + ablation",
            next_experiments=[{"name": "baseline_X", "priority": "high"}],
            status="proposed",
        )
        plan.validate()

    def test_requires_either_rationale_or_experiments(self):
        plan = AgendaRevisionPlan(selection_id=7, review_id=2, rationale="", next_experiments=[])
        with self.assertRaises(ContractValidationError):
            plan.validate()


class AgendaPersistenceTests(unittest.TestCase):
    def setUp(self):
        # Each test uses a fresh sqlite db
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["DEEPGRAPH_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        # Reset cached module-level db connection
        from db import database as db

        self._original_db_path = db.DB_PATH
        if hasattr(db, "_local"):
            for attr in ("sqlite_conn", "pg_conn", "conn"):
                if hasattr(db._local, attr):
                    try:
                        getattr(db._local, attr).close()
                    except Exception:
                        pass
                    delattr(db._local, attr)
        # Re-read DB_PATH from env (config caches it; patch directly)
        db.DB_PATH = Path(os.environ["DEEPGRAPH_DB_PATH"])
        db.init_db()

    def tearDown(self):
        from db import database as db

        if hasattr(db, "_local"):
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

    def test_save_and_get_agenda(self):
        from agents.agenda_loader import (
            get_active_agenda,
            get_agenda,
            list_agendas,
            parse_agenda,
            save_agenda,
        )

        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        agenda_id = save_agenda(agenda)
        self.assertGreater(agenda_id, 0)

        fetched = get_agenda(agenda_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, "token_scale_v1")
        self.assertEqual(fetched.focus, agenda.focus)
        self.assertEqual(fetched.prefer["paradigm_score_min"], 0.45)

        active = get_active_agenda()
        self.assertIsNotNone(active)
        self.assertEqual(active.agenda_id, agenda_id)

        rows = list_agendas()
        self.assertEqual(len(rows), 1)

    def test_multiple_agendas_stay_active(self):
        from agents.agenda_loader import (
            get_active_agenda,
            get_agenda,
            list_agendas,
            parse_agenda,
            save_agenda,
        )

        a1 = save_agenda(parse_agenda(dict(SAMPLE_AGENDA_DICT, name="a1")))
        a2 = save_agenda(parse_agenda(dict(SAMPLE_AGENDA_DICT, name="a2")))
        # Both agendas run concurrently; saving the second one must not
        # deactivate the first (isolation is per agenda_id, not per flag).
        self.assertTrue(get_agenda(a1).is_active)
        self.assertTrue(get_agenda(a2).is_active)
        active_ids = {a.agenda_id for a in list_agendas(only_active=True)}
        self.assertEqual(active_ids, {a1, a2})
        # The single-agenda convenience accessor returns the newest active row.
        self.assertEqual(get_active_agenda().agenda_id, a2)

    def test_schema_tables_present(self):
        from db import database as db

        for table in (
            "research_agendas",
            "agenda_selections",
            "agenda_reviews",
            "agenda_revision_plans",
        ):
            row = db.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            self.assertIsNotNone(row, f"missing table {table}")


if __name__ == "__main__":
    unittest.main()
