import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from triage_cggr_audit_failure import classify_audit_blockers, classify_claim_values, write_triage_report


class TriageCggrAuditFailureTests(unittest.TestCase):
    def test_engineering_blockers_are_retryable_without_method_change(self):
        result = classify_audit_blockers(
            [
                "missing or empty results/raw_predictions.jsonl",
                "method/dataset/seed cell below paper gate: method=CGGR dataset=MuSiQue seed=0 count=64/128",
            ]
        )

        self.assertEqual(result["evidence_scope_blockers"], [])
        self.assertEqual(result["unclassified_blockers"], [])
        self.assertEqual(len(result["engineering_retry_blockers"]), 2)

    def test_top_venue_missing_methods_are_evidence_scope_not_method_failure(self):
        result = classify_audit_blockers(
            [
                "required method missing: CAR-Style Certainty Adaptive Routing",
                "required method missing: Self-Route-Style Mode Routing",
            ]
        )

        self.assertEqual(result["engineering_retry_blockers"], [])
        self.assertEqual(len(result["evidence_scope_blockers"]), 2)

    def test_claim_rejection_requires_preregistered_method_iteration(self):
        result = classify_claim_values(
            {
                "claim_support_decision": "rejected",
                "cggr_vs_baseline_delta": -0.01,
                "paired_permutation_p": 0.42,
            }
        )

        self.assertTrue(result["scientific_iteration_required"])
        self.assertIn("claim_support_decision=rejected", result["scientific_iteration_reasons"])
        self.assertIn("CGGR utility delta is non-positive", result["scientific_iteration_reasons"])

    def test_write_triage_report_preserves_immutable_evidence_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = write_triage_report(
                root,
                audit_result={
                    "ok": False,
                    "blockers": ["missing or empty results/raw_predictions.jsonl"],
                    "warnings": [],
                },
            )

            self.assertTrue(Path(report["written"]).exists())
            self.assertTrue(report["immutable_evidence_policy"]["do_not_overwrite_current_artifact"])
            self.assertEqual(report["recommended_next_actions"][0]["type"], "engineering_retry")


if __name__ == "__main__":
    unittest.main()
