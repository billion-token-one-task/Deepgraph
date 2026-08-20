"""A supported verdict must be able to reach scientifically_decided.

The evidence audit recomputes both arms from raw_predictions.jsonl and refuses
on any mismatch, then -- until 2026-08-20 -- threw those numbers away before
building the transition context. metric_value, baseline_value and p_value all
defaulted to None, so decide_evidence reported
metric_missing/baseline_missing/p_value_missing on every run and
confirmation_allowed was False by construction.

Negative verdicts never consult that decision, so the defect was invisible:
nineteen runs reached scientifically_decided, every one of them carrying an
empty decision record. Probed against the exact field set the audit builds:

    verdict=refuted        -> scientifically_decided
    verdict=inconclusive   -> scientifically_decided
    verdict=supported      -> BLOCKED positive_evidence_decision_failed

V1's success terminal state was unreachable in code. No choice of experiment,
model or baseline could have produced a recordable supported verdict.

These tests pin both directions, because the fix must not become a bypass:
supplying the measurements is what the gate asks for, and a positive verdict
without them -- or with p >= alpha -- must still be refused.
"""

import unittest

from contracts.scientific_evidence import EvidenceDecisionInput, decide_evidence
from meta_harness.evidence_state import (
    EvidenceTransitionContext,
    EvidenceTransitionError,
    advance,
)

_HASH = "a" * 64

# Mirrors meta_harness.evidence_audit.base_context: every field that function
# sets, and nothing it does not.
_AUDIT_CONTEXT = dict(
    resource_grant_valid=True,
    resource_grant_id=1,
    execution_succeeded=True,
    pilot_only=False,
    full_benchmark_complete=True,
    raw_artifacts_present=True,
    claim_ledger_present=True,
    evaluator_passed=True,
    holdout_passed=True,
    raw_artifacts_hash=_HASH,
    claim_ledger_hash=_HASH,
    benchmark_contract_hash=_HASH,
    evaluator_ref="vendor:model",
    evaluator_hash=_HASH,
    holdout_ref="ds:rev:test[200:400]",
    holdout_hash=_HASH,
    verdict_hash=_HASH,
)

# What _verify_arms returns for a run whose candidate genuinely won.
_WON = dict(
    metric_value=0.72,
    baseline_value=0.665,
    p_value=0.012,
)


def _decide(verdict, **measured):
    return decide_evidence(
        EvidenceDecisionInput(
            verdict=verdict,
            p_value=measured.get("p_value"),
            metric_value=measured.get("metric_value"),
            baseline_value=measured.get("baseline_value"),
            full_benchmark_complete=True,
            raw_artifacts_complete=True,
            claim_ledger_complete=True,
            evaluator_id="vendor:model",
        )
    )


def _advance(verdict, **measured):
    context = dict(_AUDIT_CONTEXT, verdict=verdict, **measured)
    if measured:
        context["evidence_decision_passed"] = _decide(
            verdict, **measured
        ).confirmation_allowed
    return advance(
        "evidence_audited",
        "scientifically_decided",
        EvidenceTransitionContext(**context),
    )


class SupportedVerdictReachesTheLadderTests(unittest.TestCase):
    def test_a_measured_win_can_be_recorded(self):
        # The regression: this raised positive_evidence_decision_failed.
        self.assertEqual(_advance("supported", **_WON), "scientifically_decided")

    def test_a_win_without_measurements_is_still_refused(self):
        # The audit's pre-2026-08-20 context, which is also what an artifact
        # re-verification failure now falls back to.
        with self.assertRaises(EvidenceTransitionError) as caught:
            _advance("supported")
        self.assertIn("positive_evidence_decision_failed", str(caught.exception))

    def test_an_insignificant_win_is_still_refused(self):
        with self.assertRaises(EvidenceTransitionError):
            _advance("supported", **dict(_WON, p_value=0.4))

    def test_a_win_over_a_zero_baseline_is_still_refused(self):
        with self.assertRaises(EvidenceTransitionError):
            _advance("supported", **dict(_WON, baseline_value=0.0))

    def test_negative_verdicts_are_unaffected(self):
        for verdict in ("refuted", "inconclusive"):
            with self.subTest(verdict=verdict):
                self.assertEqual(_advance(verdict), "scientifically_decided")
                self.assertEqual(
                    _advance(verdict, **_WON), "scientifically_decided"
                )

    def test_the_decision_carries_the_numbers_it_judged(self):
        decision = _decide("supported", **_WON)
        self.assertTrue(decision.confirmation_allowed)
        self.assertEqual(list(decision.blockers), [])
        # And the pre-fix input produced exactly the blockers seen in all
        # nineteen persisted decision records.
        blind = _decide("supported")
        self.assertFalse(blind.confirmation_allowed)
        for missing in ("metric_missing", "baseline_missing", "p_value_missing"):
            self.assertIn(missing, blind.blockers)


if __name__ == "__main__":
    unittest.main()
