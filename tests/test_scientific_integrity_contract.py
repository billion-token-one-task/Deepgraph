import json
import unittest
from pathlib import Path

from contracts.scientific_evidence import (
    EvidenceDecisionInput,
    audit_presentation_transform,
    decide_evidence,
)


FIXTURES = Path(__file__).parent / "fixtures" / "integrity"


def _load(name: str) -> EvidenceDecisionInput:
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return EvidenceDecisionInput.from_partial_dict(payload)


class ScientificEvidenceContractTests(unittest.TestCase):
    def test_p_equals_one_is_not_significant(self):
        decision = decide_evidence(_load("p_eq_one.json"))
        self.assertFalse(decision.significant)
        self.assertFalse(decision.confirmation_allowed)
        self.assertIn("not_significant", decision.blockers)

    def test_missing_p_cannot_claim_significance(self):
        decision = decide_evidence(_load("missing_p.json"))
        self.assertFalse(decision.significant)
        self.assertFalse(decision.confirmation_allowed)
        self.assertIn("p_value_missing", decision.blockers)

    def test_refuted_cannot_be_positive_even_with_low_p(self):
        decision = decide_evidence(_load("refuted_low_p.json"))
        self.assertFalse(decision.positive_claim_allowed)
        self.assertFalse(decision.confirmation_allowed)
        self.assertEqual(decision.max_claim_strength, "none")

    def test_zero_baseline_cannot_confirm(self):
        decision = decide_evidence(_load("zero_baseline.json"))
        self.assertFalse(decision.confirmation_allowed)
        self.assertIn("baseline_zero", decision.blockers)

    def test_missing_metric_and_incomplete_benchmark_cannot_confirm(self):
        decision = decide_evidence(
            EvidenceDecisionInput(
                verdict="supported",
                p_value=0.001,
                metric_value=None,
                baseline_value=1.0,
                full_benchmark_complete=False,
                raw_artifacts_complete=True,
                claim_ledger_complete=True,
            )
        )
        self.assertFalse(decision.confirmation_allowed)
        self.assertIn("metric_missing", decision.blockers)
        self.assertIn("full_benchmark_incomplete", decision.blockers)

    def test_missing_independent_evaluator_cannot_confirm(self):
        decision = decide_evidence(
            EvidenceDecisionInput(
                verdict="supported",
                p_value=0.001,
                metric_value=0.71,
                baseline_value=0.70,
                full_benchmark_complete=True,
                raw_artifacts_complete=True,
                claim_ledger_complete=True,
                evaluator_id="",
            )
        )
        self.assertFalse(decision.confirmation_allowed)
        self.assertIn("independent_evaluator_missing", decision.blockers)

    def test_presentation_cannot_introduce_a_number(self):
        audit = audit_presentation_transform(
            "The observed result is inconclusive.",
            "The observed result improved by 7.2%.",
        )
        self.assertFalse(audit.passed)
        self.assertEqual(audit.introduced_numbers, ["7.2%"])

    def test_presentation_cannot_raise_claim_strength(self):
        audit = audit_presentation_transform(
            "The point estimate is descriptive.",
            "The result is a statistically significant confirmed improvement.",
        )
        self.assertFalse(audit.passed)
        self.assertIn("presentation_strengthened_claim", audit.blockers)


class SourceIntegrityTests(unittest.TestCase):
    def test_removed_fixed_value_and_caveat_rewriters_stay_removed(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "agents" / "paper_orchestra_pipeline.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_complete_known_main_results_rows", source)
        self.assertNotIn("_deemphasize_significance_caveats", source)
        self.assertNotIn("0.777 & 0.778 & 6.07 & 0.28 & 0.019", source)

    def test_generic_registry_has_no_topic_runner(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "agents" / "agent_registry.py").read_text(encoding="utf-8")
        self.assertNotIn("scripts.audit_cggr", source)
        self.assertNotIn("scripts.watch_cggr", source)
        self.assertNotIn("scripts.merge_cggr", source)


if __name__ == "__main__":
    unittest.main()
