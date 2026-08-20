"""The not_significant blocker must describe the statistics, not the verdict.

decide_evidence folded `and verdict == "supported"` into its `significant`
flag, so every negative verdict carrying a p-value was recorded
not_significant regardless of what the p-value said. run 207 measured
0.816 -> 0.156 at p=0.000999 -- the most decisive result in the whole run set
-- and its decision record called it not significant.

The gate never depended on this: confirmation_allowed requires
positive_allowed (verdict == "supported") alongside the flag, so the verdict
clause was redundant for the decision and wrong for the record. What it
damaged was the evidence itself. A T2 infeasibility dossier is built out of
these rows, and a decisive negative result labelled "not significant" argues
the opposite of what it measured.

These tests pin both halves: the label now tracks alpha alone, and the gate
still refuses every positive claim it refused before.
"""

import unittest

from contracts.scientific_evidence import EvidenceDecisionInput, decide_evidence


def _decide(**over):
    payload = dict(
        verdict="supported",
        p_value=0.01,
        metric_value=0.72,
        baseline_value=0.665,
        full_benchmark_complete=True,
        raw_artifacts_complete=True,
        claim_ledger_complete=True,
        evaluator_id="vendor:model",
    )
    payload.update(over)
    return decide_evidence(EvidenceDecisionInput(**payload))


class SignificanceBlockerTests(unittest.TestCase):
    def test_a_decisive_negative_is_not_called_insignificant(self):
        # run 207, verbatim.
        decision = _decide(
            verdict="refuted", p_value=0.000999, metric_value=0.156,
            baseline_value=0.816,
        )
        self.assertNotIn("not_significant", decision.blockers)
        self.assertIn("evaluator_refuted", decision.blockers)

    def test_an_inconclusive_run_over_alpha_still_is(self):
        decision = _decide(verdict="inconclusive", p_value=0.4)
        self.assertIn("not_significant", decision.blockers)

    def test_p_of_one_is_insignificant_whatever_the_verdict(self):
        for verdict in ("supported", "refuted", "inconclusive"):
            with self.subTest(verdict=verdict):
                self.assertIn(
                    "not_significant", _decide(verdict=verdict, p_value=1.0).blockers
                )

    def test_the_gate_is_unchanged(self):
        # The one combination that may pass.
        self.assertTrue(_decide().confirmation_allowed)
        # And every neighbouring one that may not.
        for label, over in (
            ("negative verdict", dict(verdict="refuted")),
            ("inconclusive verdict", dict(verdict="inconclusive")),
            ("over alpha", dict(p_value=0.06)),
            ("no p-value", dict(p_value=None)),
            ("no metric", dict(metric_value=None)),
            ("zero baseline", dict(baseline_value=0.0)),
            ("no evaluator", dict(evaluator_id=" ")),
            ("ledger incomplete", dict(claim_ledger_complete=False)),
        ):
            with self.subTest(label):
                self.assertFalse(_decide(**over).confirmation_allowed)

    def test_a_significant_negative_still_cannot_claim_anything_positive(self):
        decision = _decide(verdict="refuted", p_value=0.000999)
        self.assertFalse(decision.confirmation_allowed)
        self.assertFalse(decision.positive_claim_allowed)
        self.assertEqual(decision.max_claim_strength, "none")


if __name__ == "__main__":
    unittest.main()
