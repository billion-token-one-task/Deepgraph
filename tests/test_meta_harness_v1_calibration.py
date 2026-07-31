"""OutcomeRecord calibration contracts for isolated CI; not run on this host."""

from __future__ import annotations

import unittest

from contracts.meta_harness import OutcomeRecord
from meta_harness.calibration import build_calibration_report
from tests.test_meta_harness_v1_contracts import _decision


class CalibrationTests(unittest.TestCase):
    def test_actual_outcome_produces_prediction_error_report(self):
        decision = _decision(41)
        outcome = OutcomeRecord(
            agenda_id=11,
            idea_id=41,
            resource_grant_id=71,
            actual_tokens=25_000,
            actual_gpu_hours=0.25,
            wall_seconds=300,
            execution_result="completed_with_audited_artifacts",
            effect=0.5,
            baseline=0.4,
            verdict="supported",
            new_information={"frontier_changed": True},
            state_decision="scientifically_decided",
            prediction_error={"source": "computed_after_outcome"},
            artifact_manifest={"raw_metrics": "sha256:abc"},
        )
        report = build_calibration_report([decision], [outcome])
        self.assertEqual(report["sample_count"], 1)
        self.assertIn("success_brier_score", report)
        self.assertFalse(report["policy_update_allowed"])

    def test_no_outcome_never_auto_updates_policy(self):
        report = build_calibration_report([_decision(41)], [])
        self.assertEqual(report["status"], "insufficient_data")
        self.assertFalse(report["policy_update_allowed"])


if __name__ == "__main__":
    unittest.main()
