"""The last ladder rung, which had no executor before 2026-08-20.

No run has ever been `supported`, so none of this can be verified against
production yet. These tests ARE the verification: every path that must not
advance a run is exercised here, because the first time the happy path runs for
real it will be deciding whether the system may publish.

The gate's contract:

  * only a `supported` decision is even considered;
  * an explicit `concur: true` against a readable ledger is the ONLY thing that
    advances a run;
  * every other outcome -- refusal, unparseable answer, unreachable reviewer,
    missing ledger, missing verdict hash, exhausted attempts -- leaves the run
    at scientifically_decided, which is where it already was.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from meta_harness.manuscript_gate import (
    MANUSCRIPT_PURPOSE,
    MANUSCRIPT_REVIEWER_ID,
    MANUSCRIPT_REVIEWER_KEY_ID,
    MAX_MANUSCRIPT_REVIEW_ATTEMPTS,
    ManuscriptGateError,
    build_review_prompt,
    run_manuscript_gate,
    sign_manuscript_approval,
)
from meta_harness.reviewer_approval import (
    ReviewerApprovalError,
    ReviewerApprovalVerifier,
    scientific_manuscript_subject,
)

_SECRET = "s" * 64
_HASH = "a" * 64


def _subject(run_id=207, agenda_id=10, verdict_hash=_HASH):
    return scientific_manuscript_subject(
        agenda_id=agenda_id, experiment_run_id=run_id, verdict_hash=verdict_hash
    )


class SigningTests(unittest.TestCase):
    def setUp(self):
        self.verifier = ReviewerApprovalVerifier(
            {MANUSCRIPT_REVIEWER_KEY_ID: "env:_MANUSCRIPT_TEST_SECRET"}
        )
        patcher = patch.dict("os.environ", {"_MANUSCRIPT_TEST_SECRET": _SECRET})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _envelope(self, approval):
        return {
            "reviewer_id": approval.reviewer_id,
            "key_id": approval.key_id,
            "purpose": approval.purpose,
            "subject": approval.subject,
            "issued_at": approval.issued_at,
            "signature": approval.signature,
        }

    def test_a_minted_approval_verifies(self):
        subject = _subject()
        approval = sign_manuscript_approval(subject=subject, secret=_SECRET)
        verified = self.verifier.verify(
            self._envelope(approval), purpose=MANUSCRIPT_PURPOSE, subject=subject
        )
        self.assertEqual(verified.reviewer_id, MANUSCRIPT_REVIEWER_ID)

    def test_the_identity_is_not_the_operator(self):
        # The 34 pre-existing approvals are all reviewer_id='operator' with
        # purpose='retrospective_review'. An AI approval must never be
        # mistakable for one of those.
        approval = sign_manuscript_approval(subject=_subject(), secret=_SECRET)
        self.assertNotEqual(approval.reviewer_id, "operator")
        self.assertNotEqual(approval.key_id, "operator-20260817")
        self.assertEqual(approval.purpose, MANUSCRIPT_PURPOSE)

    def test_a_signature_does_not_travel_to_another_run(self):
        approval = sign_manuscript_approval(subject=_subject(207), secret=_SECRET)
        with self.assertRaises(ReviewerApprovalError):
            self.verifier.verify(
                self._envelope(approval),
                purpose=MANUSCRIPT_PURPOSE,
                subject=_subject(208),
            )

    def test_a_signature_does_not_travel_to_another_purpose(self):
        approval = sign_manuscript_approval(subject=_subject(), secret=_SECRET)
        with self.assertRaises(ReviewerApprovalError):
            self.verifier.verify(
                self._envelope(approval),
                purpose="retrospective_review",
                subject=_subject(),
            )

    def test_a_different_verdict_hash_is_a_different_subject(self):
        # The subject binds the approval to the exact evidence that earned it,
        # so re-running the experiment invalidates the old approval.
        self.assertNotEqual(_subject(verdict_hash=_HASH), _subject(verdict_hash="b" * 64))


class PromptTests(unittest.TestCase):
    def test_the_prompt_asks_a_different_question_from_the_audit(self):
        prompt = build_review_prompt(
            decision={"verdict": "supported"}, ledger={"metric": "x"}, holdout={}
        )
        # The evidence audit already judged verdict-vs-measurement. Asking it
        # again would be the same judge twice.
        self.assertIn("NOT re-judging", prompt)
        self.assertIn("sufficient to justify", prompt)
        # Refusal must be framed as free, the way the audit evaluator's is.
        self.assertIn("Dissent freely", prompt)


class _GateHarness:
    """Minimal stand-ins for the four things the gate touches."""

    def __init__(self, workdir, *, verdict="supported", verdict_hash=_HASH):
        self.workdir = workdir
        self.decision = {
            "verdict": verdict,
            "verdict_hash": verdict_hash,
            "evidence_decision_json": {},
        }
        self.attempts = 0
        self.advanced = []

    def fetchone(self, sql, params=None):
        if "scientific_decision_records" in sql:
            return dict(self.decision)
        if "COUNT(*)" in sql:
            return {"n": self.attempts}
        return None

    def run_row(self):
        return {
            "id": 207,
            "agenda_id": 10,
            "deep_insight_id": 42,
            "resource_grant_id": 900,
            "workdir": str(self.workdir),
        }

    def repository(self):
        harness = self

        class _Repo:
            def advance_experiment_state(self, **kwargs):
                harness.advanced.append(kwargs)
                return "manuscript_allowed"

        return _Repo


class GateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)
        results = self.workdir / "results"
        results.mkdir()
        (results / "claim_ledger.json").write_text(json.dumps({"metric": "acc"}))

    def _run(self, harness, llm):
        with patch("meta_harness.manuscript_gate.db.fetchone", harness.fetchone), patch(
            "meta_harness.manuscript_gate.call_llm_for_role", llm
        ), patch(
            "meta_harness.manuscript_gate.MetaHarnessRepository", harness.repository()
        ):
            return run_manuscript_gate(harness.run_row(), secret=_SECRET, log=lambda *a: None)

    @staticmethod
    def _answer(concur):
        return lambda *a, **k: (
            json.dumps({"concur": concur, "reasons": ["r"]}),
            10,
            {"provider": "novita_deepseek", "model": "deepseek-v4-flash"},
        )

    def test_concurrence_advances_the_run(self):
        harness = _GateHarness(self.workdir)
        self.assertEqual(self._run(harness, self._answer(True)), "manuscript_allowed")
        self.assertEqual(len(harness.advanced), 1)
        call = harness.advanced[0]
        self.assertEqual(call["target"], "manuscript_allowed")
        # The verifier requires actor == the signed reviewer, so the audit
        # trail has to name the AI rather than an operator.
        self.assertEqual(call["actor"], MANUSCRIPT_REVIEWER_ID)
        self.assertEqual(call["context"].verdict, "supported")
        self.assertEqual(
            call["context"].reviewer_approval["purpose"], MANUSCRIPT_PURPOSE
        )
        # public_record() carries signature_hash, not signature; sending that
        # would verify as an incomplete envelope.
        self.assertIn("signature", call["context"].reviewer_approval)

    def test_refusal_leaves_the_run_where_it_was(self):
        harness = _GateHarness(self.workdir)
        self.assertEqual(self._run(harness, self._answer(False)), "refused")
        self.assertEqual(harness.advanced, [])

    def test_a_non_supported_run_is_never_reviewed(self):
        for verdict in ("refuted", "inconclusive"):
            with self.subTest(verdict=verdict):
                harness = _GateHarness(self.workdir, verdict=verdict)

                def _boom(*a, **k):  # the reviewer must not be called at all
                    raise AssertionError("reviewer called for a negative verdict")

                self.assertEqual(self._run(harness, _boom), "not_supported")
                self.assertEqual(harness.advanced, [])

    def test_an_unreachable_reviewer_is_not_approval(self):
        harness = _GateHarness(self.workdir)

        def _down(*a, **k):
            raise ConnectionError("provider unreachable")

        self.assertEqual(self._run(harness, _down), "review_failed")
        self.assertEqual(harness.advanced, [])

    def test_an_unparseable_answer_is_not_approval(self):
        harness = _GateHarness(self.workdir)
        garbage = lambda *a, **k: ("I think it is fine", 10, {"provider": "p", "model": "m"})
        self.assertEqual(self._run(harness, garbage), "review_failed")
        self.assertEqual(harness.advanced, [])

    def test_exhausted_attempts_do_not_advance(self):
        harness = _GateHarness(self.workdir)
        harness.attempts = MAX_MANUSCRIPT_REVIEW_ATTEMPTS
        self.assertEqual(self._run(harness, self._answer(True)), "review_failed")
        self.assertEqual(harness.advanced, [])

    def test_a_missing_ledger_does_not_advance(self):
        (self.workdir / "results" / "claim_ledger.json").unlink()
        harness = _GateHarness(self.workdir)
        self.assertEqual(self._run(harness, self._answer(True)), "no_ledger")
        self.assertEqual(harness.advanced, [])

    def test_a_missing_verdict_hash_does_not_advance(self):
        harness = _GateHarness(self.workdir, verdict_hash="")
        self.assertEqual(self._run(harness, self._answer(True)), "no_verdict_hash")
        self.assertEqual(harness.advanced, [])

    def test_a_run_without_a_grant_does_not_advance(self):
        harness = _GateHarness(self.workdir)
        row = harness.run_row() | {"resource_grant_id": 0}
        with patch("meta_harness.manuscript_gate.db.fetchone", harness.fetchone):
            self.assertEqual(
                run_manuscript_gate(row, secret=_SECRET, log=lambda *a: None), "no_grant"
            )
        self.assertEqual(harness.advanced, [])


class AttemptBoundTests(unittest.TestCase):
    def test_the_error_names_the_exhausted_budget(self):
        from meta_harness.manuscript_gate import review_manuscript_readiness

        with patch(
            "meta_harness.manuscript_gate._review_attempts",
            lambda _grant: MAX_MANUSCRIPT_REVIEW_ATTEMPTS,
        ):
            with self.assertRaises(ManuscriptGateError) as caught:
                review_manuscript_readiness(
                    agenda_id=10, idea_id=42, resource_grant_id=900,
                    decision={}, ledger={}, holdout=None, verdict_hash=_HASH,
                )
        self.assertIn("attempts exhausted", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
