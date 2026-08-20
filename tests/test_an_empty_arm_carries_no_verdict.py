"""An arm that generated nothing did not measure the intervention.

run 235 walked the entire evidence ladder -- sanity, full benchmark, claim
ledger, cross-vendor audit, disjoint holdout -- and was recorded `refuted` at
p=0.000999, baseline 0.32 against candidate 0.0.

The candidate's 200 predictions were all empty.

Nothing in the chain was lying. The recomputation was honest: empty predictions
score 0.0. The permutation test was honest: 0.0 really is significantly below
0.32. The evaluator concurred. Every gate did its job on a number that meant
nothing, because the arm never ran -- the same double-wrapped prompt that made
the model emit EOS immediately, which the M3 session had already diagnosed and
retracted three of its own results for.

A system that records `refuted` for experiments that produced no output is
manufacturing negative results. That is worse than producing none: the
nineteen-verdict sign test that concluded "these interventions are
systematically harmful" is exactly the kind of claim this would quietly
poison. (Audited: of the 26 runs on the ladder, only run 235 is affected -- the
other 25 have 0.0% empty predictions across all 50 arms, so that analysis
stands.)

The audit already refuses when a recomputed arm disagrees with its reported
value. This is the same refusal for the case where there was nothing to
recompute.
"""

import unittest

from meta_harness.evidence_audit import (
    MAX_BLANK_PREDICTION_RATE,
    EvidenceAuditError,
    _verify_arms,
)

def _final(*, baseline=1.0, candidate=1.0):
    """The reported values must match what the rows recompute to.

    The audit refuses on a recompute mismatch before it looks at anything
    else, so a fixture that gets this wrong tests the wrong refusal.
    """
    return {
        "metric_name": "numeric_accuracy",
        "baseline_method": "unmodified_input_baseline",
        "candidate_method": "direct_balanced_5shot_icl",
        "baseline_metric_value": baseline,
        "metric_value": candidate,
        "p_value": 0.01,
    }


_FINAL = _final()


def _rows(*, candidate_blank, n=20):
    rows = []
    for index in range(n):
        rows.append(
            {
                "method": "unmodified_input_baseline",
                "prediction": "#### 4",
                "target": "#### 4",
                "sample_index": index,
            }
        )
        blank = index < candidate_blank
        rows.append(
            {
                "method": "direct_balanced_5shot_icl",
                "prediction": "" if blank else "#### 4",
                "target": "#### 4",
                "sample_index": index,
            }
        )
    return rows


class EmptyArmTests(unittest.TestCase):
    def test_a_fully_empty_arm_is_refused(self):
        # run 235, in miniature.
        with self.assertRaises(EvidenceAuditError) as caught:
            _verify_arms(_FINAL, _rows(candidate_blank=20))
        message = str(caught.exception)
        self.assertIn("empty predictions", message)
        self.assertIn("measured nothing", message)

    def test_the_refusal_names_the_arm(self):
        with self.assertRaises(EvidenceAuditError) as caught:
            _verify_arms(_FINAL, _rows(candidate_blank=20))
        self.assertIn("direct_balanced_5shot_icl", str(caught.exception))

    def test_a_majority_empty_arm_is_refused(self):
        with self.assertRaises(EvidenceAuditError):
            _verify_arms(_FINAL, _rows(candidate_blank=11))

    def test_a_baseline_that_generated_nothing_is_refused_too(self):
        # The rule is about arms, not about which side is being defended.
        rows = _rows(candidate_blank=0)
        for row in rows:
            if row["method"] == "unmodified_input_baseline":
                row["prediction"] = ""
        with self.assertRaises(EvidenceAuditError) as caught:
            _verify_arms(_FINAL, rows)
        self.assertIn("unmodified_input_baseline", str(caught.exception))

    def test_a_whitespace_only_prediction_counts_as_empty(self):
        rows = _rows(candidate_blank=0)
        for row in rows:
            if row["method"] == "direct_balanced_5shot_icl":
                row["prediction"] = "   \n "
        with self.assertRaises(EvidenceAuditError):
            _verify_arms(_FINAL, rows)


class ItDoesNotRefuseRealMeasurementsTests(unittest.TestCase):
    """Measured: 50 arms across 25 audited runs, every one at 0.0% empty."""

    def test_an_arm_with_no_blanks_passes(self):
        verified = _verify_arms(_FINAL, _rows(candidate_blank=0))
        self.assertEqual(verified["baseline"], 1.0)
        self.assertEqual(verified["candidate"], 1.0)

    def test_a_few_blanks_are_tolerated(self):
        # A model may legitimately fail to answer some items; that is a result,
        # not a failure to run. Nine of twenty stays under the threshold.
        verified = _verify_arms(
            _final(candidate=0.55), _rows(candidate_blank=9)
        )
        self.assertEqual(verified["candidate"], 0.55)

    def test_the_threshold_sits_between_the_two_observed_modes(self):
        self.assertGreater(MAX_BLANK_PREDICTION_RATE, 0.0)
        self.assertLess(MAX_BLANK_PREDICTION_RATE, 1.0)


if __name__ == "__main__":
    unittest.main()
