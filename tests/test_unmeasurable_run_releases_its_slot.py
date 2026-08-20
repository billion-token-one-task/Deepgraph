"""A run that can never be audited must not hold a concurrency slot forever.

Refusing to score an empty arm was correct and immediately produced a second
defect, one level down.

Before the refusal existed, run 236 would have settled on a fabricated verdict
-- candidate 0.0 against baseline 0.32, p=0.000999, from 200 empty predictions
-- and its grant would have closed on the way past. The wrong answer was also,
accidentally, the thing that kept the pipeline moving.

With the refusal in place and nothing else changed, agenda 14 deadlocked inside
the hour:

    active grants   4        grant 287  evidence_audit  idea 203 (run 236)
    max_concurrency 4        -> no new work could start

Every audit pass re-read the same artifacts, reached the same conclusion, and
logged the same refusal: four times in the first hour, and it would have
continued indefinitely.

The distinction that fixes it is whether re-reading can change the answer. A
lost evaluator response, a holdout killed in transport, a provider that refused
-- retrying those reads new facts. Predictions already written to disk are what
the run produced; retrying reads the same bytes. So the empty-arm refusal is a
permanent one: fail the run, settle the grant, give the slot back.
"""

import unittest

from meta_harness.evidence_audit import (
    EvidenceAuditError,
    EvidenceAuditPermanentError,
    _verify_arms,
)


def _final(*, baseline=1.0, candidate=1.0):
    return {
        "metric_name": "numeric_accuracy",
        "baseline_method": "unmodified_input_baseline",
        "candidate_method": "code_prompting_2shot",
        "baseline_metric_value": baseline,
        "metric_value": candidate,
        "p_value": 0.01,
    }


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
        rows.append(
            {
                "method": "code_prompting_2shot",
                "prediction": "" if index < candidate_blank else "#### 4",
                "target": "#### 4",
                "sample_index": index,
            }
        )
    return rows


class PermanentFailureTests(unittest.TestCase):
    def test_an_empty_arm_raises_the_permanent_kind(self):
        with self.assertRaises(EvidenceAuditPermanentError):
            _verify_arms(_final(), _rows(candidate_blank=20))

    def test_it_is_still_an_audit_error(self):
        # Callers that only know the base class must keep catching it, or a
        # narrowed exception becomes an uncaught crash in the advance loop.
        with self.assertRaises(EvidenceAuditError):
            _verify_arms(_final(), _rows(candidate_blank=20))

    def test_a_recompute_mismatch_stays_retryable(self):
        # Reported values that disagree with the rows may be a mid-write
        # artifact, so that one keeps its retry.
        with self.assertRaises(EvidenceAuditError) as caught:
            _verify_arms(_final(baseline=0.5), _rows(candidate_blank=0))
        self.assertNotIsInstance(caught.exception, EvidenceAuditPermanentError)

    def test_a_missing_arm_stays_retryable(self):
        rows = [r for r in _rows(candidate_blank=0) if r["method"].startswith("unmod")]
        with self.assertRaises(EvidenceAuditError) as caught:
            _verify_arms(_final(), rows)
        self.assertNotIsInstance(caught.exception, EvidenceAuditPermanentError)


class TheAdvanceLoopHandlesItTests(unittest.TestCase):
    """The handler must exist, run first, and actually free the slot."""

    def _source(self):
        import inspect

        from scripts import auto_advance

        return inspect.getsource(auto_advance.advance_evidence_audit)

    def test_the_permanent_case_is_caught_before_the_general_one(self):
        source = self._source()
        permanent = source.find("except EvidenceAuditPermanentError")
        general = source.find("except EvidenceAuditError")
        self.assertNotEqual(permanent, -1, "no handler for the permanent case")
        self.assertLess(
            permanent,
            general,
            "the general handler shadows the permanent one; Python matches the "
            "first except clause a subclass satisfies",
        )

    def test_it_terminates_the_run_and_settles_the_grant(self):
        source = self._source()
        head = source[source.find("except EvidenceAuditPermanentError") :]
        head = head[: head.find("except EvidenceAuditError")]
        self.assertIn("status='failed'", head)
        self.assertIn("_settle_completed_grants", head)

    def test_it_is_logged_distinctly_from_a_retryable_block(self):
        # "blocked" reads as "waiting"; this one is finished.
        source = self._source()
        self.assertIn("evidence_audit_unmeasurable", source)


if __name__ == "__main__":
    unittest.main()
