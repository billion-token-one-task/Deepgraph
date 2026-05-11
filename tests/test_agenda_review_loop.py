"""Tests for reviewer_adapter + revision_planner (issue #9 step 5)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DEEPGRAPH_DATABASE_URL", "")


SAMPLE_AGENDA_DICT = {
    "version": "v1",
    "name": "review_loop_test",
    "focus": ["long context"],
    "prefer": {"keywords": ["linear attention"], "tiers": ["tier_2"]},
    "reject": {},
}


class ReviewLoopTests(unittest.TestCase):
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
        self._tmpdir.cleanup()
        os.environ.pop("DEEPGRAPH_DB_PATH", None)

    def _seed_full_run(self, *, verdict: str = "confirmed", with_bundle: bool = True):
        # insight
        self.db.execute(
            """
            INSERT INTO deep_insights
                (id, tier, status, title, problem_statement,
                 adversarial_score, novelty_status, resource_class,
                 experimentability, evidence_plan, predictions)
            VALUES (1, 2, 'verified', 'Linear attention for long context',
                    'Quadratic attention is expensive.', 7.5, 'novel', 'gpu_small',
                    'easy',
                    '{"main_table": {"enabled": true, "priority": "required"}}',
                    '[]')
            """
        )
        # experiment_run
        cur = self.db.execute(
            """
            INSERT INTO experiment_runs
                (deep_insight_id, status, phase, hypothesis_verdict,
                 baseline_metric_name, baseline_metric_value, best_metric_value,
                 effect_size, effect_pct, workdir)
            VALUES (1, 'completed', 'hypothesis_testing', ?, 'accuracy', 0.65, 0.78,
                    0.13, 20.0, '/tmp/exp')
            """,
            (verdict,),
        )
        exp_id = cur.lastrowid
        # claims
        self.db.execute(
            """
            INSERT INTO experimental_claims
                (run_id, deep_insight_id, claim_text, claim_type, verdict, effect_size,
                 confidence, p_value)
            VALUES (?, 1, 'Linear attention improves accuracy by 20% on long context.',
                    'experimental', ?, 0.13, 0.92, 0.01)
            """,
            (exp_id, verdict),
        )
        bundle_id = None
        manu_id = None
        if with_bundle:
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
        return {"experiment_run_id": exp_id, "manuscript_run_id": manu_id, "submission_bundle_id": bundle_id}

    def _setup_selection_with_artifacts(self, **kwargs):
        from agents.agenda_loader import parse_agenda, save_agenda
        from agents.agenda_orchestrator import link_existing_artifacts
        from agents.agenda_selector import select_and_persist

        artifacts = self._seed_full_run(**kwargs)
        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        save_agenda(agenda)
        sel = select_and_persist(agenda)
        self.assertEqual(sel.selected_insight_id, 1)
        link_existing_artifacts(sel.selection_id)
        return sel, artifacts

    def test_reviewer_registry_has_default(self):
        from agents.reviewer_adapter import list_reviewers

        self.assertIn("internal_evidence_gate", list_reviewers())

    def test_confirmed_run_yields_accept_or_minor(self):
        from agents.reviewer_adapter import run_review

        sel, _ = self._setup_selection_with_artifacts(verdict="confirmed", with_bundle=True)
        review = run_review(sel.selection_id)
        self.assertIn(review.recommendation, ("accept", "minor_revision"))
        self.assertEqual(review.reviewer, "internal_evidence_gate")
        self.assertGreaterEqual(len(review.strengths), 1)

    def test_refuted_run_yields_reject(self):
        from agents.reviewer_adapter import run_review

        sel, _ = self._setup_selection_with_artifacts(verdict="refuted", with_bundle=True)
        review = run_review(sel.selection_id)
        self.assertEqual(review.recommendation, "reject")

    def test_inconclusive_run_yields_major_revision(self):
        from agents.reviewer_adapter import run_review

        sel, _ = self._setup_selection_with_artifacts(verdict="inconclusive", with_bundle=True)
        review = run_review(sel.selection_id)
        self.assertEqual(review.recommendation, "major_revision")

    def test_missing_bundle_pushes_to_minor(self):
        from agents.reviewer_adapter import run_review

        sel, _ = self._setup_selection_with_artifacts(verdict="confirmed", with_bundle=False)
        review = run_review(sel.selection_id)
        self.assertEqual(review.recommendation, "minor_revision")
        self.assertTrue(
            any("manuscript bundle" in r.lower() for r in review.required_revisions)
        )

    def test_unknown_reviewer_raises(self):
        from agents.reviewer_adapter import run_review
        from contracts.base import ContractValidationError

        sel, _ = self._setup_selection_with_artifacts(verdict="confirmed", with_bundle=True)
        with self.assertRaises(ContractValidationError):
            run_review(sel.selection_id, reviewer="paperreview_ai")

    def test_revision_plan_from_minor_revision(self):
        from agents.reviewer_adapter import run_review
        from agents.revision_planner import build_revision_plan

        sel, _ = self._setup_selection_with_artifacts(verdict="confirmed", with_bundle=False)
        review = run_review(sel.selection_id)
        plan = build_revision_plan(review.review_id)
        self.assertEqual(plan.selection_id, sel.selection_id)
        self.assertGreater(len(plan.next_experiments), 0)
        kinds = {e.get("kind") for e in plan.next_experiments}
        self.assertTrue(kinds.intersection({"evidence_gap", "baseline", "ablation", "robustness", "prediction_test"}))
        for e in plan.next_experiments:
            self.assertIn("name", e)
            self.assertIn("rationale", e)
            self.assertIn("priority", e)

    def test_revision_plan_accept_is_noop(self):
        from agents.reviewer_adapter import run_review
        from agents.revision_planner import build_revision_plan

        sel, _ = self._setup_selection_with_artifacts(verdict="confirmed", with_bundle=True)
        review = run_review(sel.selection_id)
        # if review came back as accept, revision plan should be noop status
        plan = build_revision_plan(review.review_id)
        if review.recommendation == "accept":
            self.assertEqual(plan.status, "noop")
        else:
            self.assertEqual(plan.status, "proposed")

    def test_get_latest_review_and_plan(self):
        from agents.reviewer_adapter import get_latest_review, run_review
        from agents.revision_planner import build_revision_plan, get_latest_plan_for_selection

        sel, _ = self._setup_selection_with_artifacts(verdict="inconclusive", with_bundle=True)
        review = run_review(sel.selection_id)
        plan = build_revision_plan(review.review_id)

        latest_review = get_latest_review(sel.selection_id)
        self.assertEqual(latest_review["id"], review.review_id)
        latest_plan = get_latest_plan_for_selection(sel.selection_id)
        self.assertEqual(latest_plan["id"], plan.plan_id)


if __name__ == "__main__":
    unittest.main()
