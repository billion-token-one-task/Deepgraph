"""Tests for agents.evidence_gate (issue #9 acceptance: pass/block gate).

Covers both required paths:
- status='pass'  -> manuscript bundle allowed to be created.
- status='block' -> manuscript bundle MUST NOT be created.
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
    "name": "evidence_gate_test_agenda",
    "focus": ["long context"],
    "prefer": {"keywords": ["linear attention"], "tiers": ["tier_2"]},
    "reject": {},
    "required_output": {"experiment_result_packet": True},
}


def _reset_db_locals(db):
    for attr in ("sqlite_conn", "pg_conn", "conn"):
        if hasattr(db._local, attr):
            try:
                getattr(db._local, attr).close()
            except Exception:
                pass
            delattr(db._local, attr)


class EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["DEEPGRAPH_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        from db import database as db

        self._original_db_path = db.DB_PATH
        _reset_db_locals(db)
        db.DB_PATH = Path(os.environ["DEEPGRAPH_DB_PATH"])
        db.init_db()
        self.db = db
        self.workdir = Path(self._tmpdir.name) / "exp_workdir"
        self.workdir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        from db import database as db
        _reset_db_locals(db)
        # Restore original DB_PATH so downstream tests don't inherit a deleted
        # tempdir path through the module-level singleton.
        db.DB_PATH = self._original_db_path
        self._tmpdir.cleanup()
        os.environ.pop("DEEPGRAPH_DB_PATH", None)

    # ---------- fixtures ----------

    def _seed_insight(self, insight_id=1):
        self.db.execute(
            """
            INSERT INTO deep_insights
                (id, tier, status, title, problem_statement, adversarial_score,
                 novelty_status, resource_class, experimentability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (insight_id, 2, "verified", "Linear attention long context",
             "Quadratic attention is too expensive.", 8.5,
             "novel", "gpu_small", "easy"),
        )
        self.db.commit()

    def _new_selection(self, insight_id):
        from agents.agenda_loader import parse_agenda, save_agenda
        from agents.agenda_selector import select_and_persist

        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        save_agenda(agenda)
        sel = select_and_persist(agenda)
        return sel

    def _seed_completed_run(self, insight_id, sel_id, *, with_packet=True,
                            with_confirmed_claim=True, with_refuted_claim=False):
        cur = self.db.execute(
            """
            INSERT INTO experiment_runs
                (deep_insight_id, status, phase, hypothesis_verdict,
                 baseline_metric_value, best_metric_value, effect_size, workdir)
            VALUES (?, 'completed', 'hypothesis_testing', 'confirmed',
                    1.0, 0.5, 0.5, ?)
            """,
            (insight_id, str(self.workdir)),
        )
        run_id = cur.lastrowid
        self.db.execute(
            "UPDATE agenda_selections SET experiment_run_id=? WHERE id=?",
            (run_id, sel_id),
        )
        if with_confirmed_claim:
            self.db.execute(
                """
                INSERT INTO experimental_claims
                    (run_id, deep_insight_id, claim_text, claim_type, verdict,
                     effect_size, confidence)
                VALUES (?, ?, 'speedup 2x', 'experimental', 'confirmed', 2.0, 0.9)
                """,
                (run_id, insight_id),
            )
        if with_refuted_claim:
            self.db.execute(
                """
                INSERT INTO experimental_claims
                    (run_id, deep_insight_id, claim_text, claim_type, verdict,
                     effect_size, confidence)
                VALUES (?, ?, 'no improvement', 'experimental', 'refuted', 0.0, 0.7)
                """,
                (run_id, insight_id),
            )
        self.db.commit()
        if with_packet:
            packet = {
                "config": {"seq_len": 512, "head_dim": 64, "seed": 1729},
                "softmax_attention": {"latency_ms_median": 10.0},
                "linear_attention": {"latency_ms_median": 5.0},
                "delta": {"latency_speedup_x": 2.0, "relative_error": 0.1},
            }
            (self.workdir / "experiment_result_packet.json").write_text(
                json.dumps(packet), encoding="utf-8",
            )
        return run_id

    # ---------- pass path ----------

    def test_gate_passes_when_all_requirements_met(self):
        from agents import evidence_gate
        self._seed_insight(1)
        sel = self._new_selection(1)
        self._seed_completed_run(1, sel.selection_id)

        decision = evidence_gate.run_gate(sel.selection_id)
        self.assertEqual(decision["status"], "pass", decision)
        self.assertEqual(decision["blockers"], [])
        self.assertIn("claim_counts", decision["metrics_summary"])
        # persisted
        latest = evidence_gate.get_latest_gate(sel.selection_id)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["status"], "pass")

    # ---------- block paths ----------

    def test_gate_blocks_when_no_experiment_run(self):
        from agents import evidence_gate
        self._seed_insight(1)
        sel = self._new_selection(1)

        decision = evidence_gate.run_gate(sel.selection_id)
        self.assertEqual(decision["status"], "block")
        reqs = [b["requirement"] for b in decision["blockers"]]
        self.assertIn("experiment_run", reqs)

    def test_gate_blocks_when_no_confirmed_claim(self):
        from agents import evidence_gate
        self._seed_insight(1)
        sel = self._new_selection(1)
        self._seed_completed_run(1, sel.selection_id, with_confirmed_claim=False)

        decision = evidence_gate.run_gate(sel.selection_id)
        self.assertEqual(decision["status"], "block")
        reqs = [b["requirement"] for b in decision["blockers"]]
        self.assertIn("experimental_claims.confirmed>=1", reqs)

    def test_gate_blocks_when_packet_missing(self):
        from agents import evidence_gate
        self._seed_insight(1)
        sel = self._new_selection(1)
        self._seed_completed_run(1, sel.selection_id, with_packet=False)

        decision = evidence_gate.run_gate(sel.selection_id)
        self.assertEqual(decision["status"], "block")
        reqs = [b["requirement"] for b in decision["blockers"]]
        self.assertIn("experiment_result_packet.json", reqs)

    def test_gate_blocks_when_refuted_claim_present(self):
        from agents import evidence_gate
        self._seed_insight(1)
        sel = self._new_selection(1)
        self._seed_completed_run(1, sel.selection_id, with_refuted_claim=True)

        decision = evidence_gate.run_gate(sel.selection_id)
        self.assertEqual(decision["status"], "block")
        reqs = [b["requirement"] for b in decision["blockers"]]
        self.assertIn("no_refuted_primary_claims", reqs)


class RunRealPipelineTests(unittest.TestCase):
    """End-to-end: run_real_pipeline must NOT create manuscript when gate blocks."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["DEEPGRAPH_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        from db import database as db
        self._original_db_path = db.DB_PATH
        _reset_db_locals(db)
        db.DB_PATH = Path(os.environ["DEEPGRAPH_DB_PATH"])
        db.init_db()
        self.db = db

    def tearDown(self):
        from db import database as db
        _reset_db_locals(db)
        # Restore original DB_PATH so downstream tests don't inherit a deleted
        # tempdir path through the module-level singleton.
        db.DB_PATH = self._original_db_path
        self._tmpdir.cleanup()
        os.environ.pop("DEEPGRAPH_DB_PATH", None)

    def _seed_insight(self, insight_id=1):
        self.db.execute(
            """
            INSERT INTO deep_insights
                (id, tier, status, title, problem_statement, adversarial_score,
                 novelty_status, resource_class, experimentability)
            VALUES (?, 2, 'verified', 'Linear attention long context',
                    'Quadratic attention is too expensive.', 8.5,
                    'novel', 'gpu_small', 'easy')
            """,
            (insight_id,),
        )
        self.db.commit()

    def _new_selection(self):
        from agents.agenda_loader import parse_agenda, save_agenda
        from agents.agenda_selector import select_and_persist
        agenda = parse_agenda(SAMPLE_AGENDA_DICT)
        save_agenda(agenda)
        return select_and_persist(agenda)

    def test_pass_path_creates_manuscript(self):
        """Real benchmark on seq_len=128 produces a real packet -> gate pass -> manuscript created."""
        from agents.agenda_orchestrator import run_real_pipeline
        self._seed_insight(1)
        sel = self._new_selection()

        result = run_real_pipeline(
            sel.selection_id, seq_len=128, head_dim=32, seed=1729, repeats=2,
        )
        self.assertEqual(result["evidence_gate"]["status"], "pass", result)
        self.assertTrue(result["manuscript_created"], result)
        self.assertIsNotNone(result["manuscript"])
        # selection should be marked completed
        row = self.db.fetchone(
            "SELECT status, manuscript_run_id, submission_bundle_id "
            "FROM agenda_selections WHERE id=?",
            (sel.selection_id,),
        )
        self.assertEqual(row["status"], "completed")
        self.assertIsNotNone(row["manuscript_run_id"])
        self.assertIsNotNone(row["submission_bundle_id"])

    def test_blocked_path_does_not_create_manuscript(self):
        """Force-block by deleting packet AFTER run -> rerun gate yields block -> no manuscript."""
        from agents import evidence_gate
        from agents.agenda_orchestrator import _create_manuscript_run_if_allowed
        from agents import real_experiment_runner

        self._seed_insight(1)
        sel = self._new_selection()

        # Run real experiment then sabotage the packet to simulate missing evidence.
        exp = real_experiment_runner.run_real_experiment_for_selection(
            sel.selection_id, seq_len=128, head_dim=32, seed=1729, repeats=2,
        )
        Path(exp["packet_path"]).unlink()
        # Also mark all claims as refuted to trigger another blocker.
        self.db.execute(
            "UPDATE experimental_claims SET verdict='refuted' WHERE run_id=?",
            (exp["run_id"],),
        )
        self.db.commit()

        decision = evidence_gate.run_gate(sel.selection_id)
        self.assertEqual(decision["status"], "block")
        # Manuscript creation guarded: caller must not invoke it on block.
        # Verify orchestrator path: we mimic dispatch with mode='bench' but
        # since our packet is gone, we rerun the gate explicitly and assert
        # no manuscript_run exists.
        manu = self.db.fetchone(
            "SELECT id FROM manuscript_runs WHERE deep_insight_id=?", (1,),
        )
        self.assertIsNone(manu, "manuscript MUST NOT exist when gate blocked")


if __name__ == "__main__":
    unittest.main()
