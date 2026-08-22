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
from unittest.mock import ANY, patch

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

    def __init__(
        self,
        workdir,
        *,
        verdict="supported",
        verdict_hash=_HASH,
        grant_stage="evidence_audit",
    ):
        self.workdir = workdir
        self.decision = {
            "verdict": verdict,
            "verdict_hash": verdict_hash,
            "evidence_decision_json": {},
        }
        self.attempts = 0
        self.advanced = []
        self.terminal = None
        self.records = []
        self.completed = []
        self.grant_stage = grant_stage
        self.state = "scientifically_decided"
        self.commits = 0
        self.rollbacks = 0

    def fetchone(self, sql, params=None):
        if "scientific_decision_records" in sql:
            return dict(self.decision)
        if "FROM experiment_runs" in sql:
            return {
                "resource_grant_id": 900,
                "scientific_evidence_state": self.state,
            }
        if "FROM resource_grants" in sql:
            return {"stage": self.grant_stage}
        if "FROM resource_grant_usage_reservations" in sql:
            return {"id": 901, "status": "settled"}
        return None

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def run_row(self):
        return {
            "id": 207,
            "agenda_id": 10,
            "deep_insight_id": 42,
            "resource_grant_id": 900,
            "workdir": str(self.workdir),
            "scientific_evidence_state": "scientifically_decided",
        }

    def repository(self):
        harness = self

        class _Repo:
            def begin_manuscript_gate_attempt(self, **kwargs):
                if harness.attempts >= MAX_MANUSCRIPT_REVIEW_ATTEMPTS:
                    raise ManuscriptGateError("manuscript review attempts exhausted")
                harness.attempts += 1
                # The real repository commits this append before provider work.
                harness.commit()
                return {
                    "id": harness.attempts,
                    "attempt_number": harness.attempts,
                    "idempotency_key": (
                        f"manuscript-gate:10:run207:{_HASH}:a{harness.attempts}"
                    ),
                    "grant_stage": harness.grant_stage,
                }

            def count_manuscript_gate_attempts(self, **kwargs):
                return harness.attempts

            def load_manuscript_gate_record(self, **kwargs):
                return harness.terminal

            def record_manuscript_gate_result(self, **kwargs):
                harness.records.append(kwargs)
                harness.terminal = {
                    "id": 1,
                    "disposition": kwargs["disposition"],
                    "resource_grant_id": 900,
                }
                return 1

            def advance_experiment_state(self, **kwargs):
                harness.advanced.append(kwargs)
                harness.state = "manuscript_allowed"
                return "manuscript_allowed"

            def complete_manuscript_grant(self, **kwargs):
                harness.completed.append(kwargs)
                return 10

        return _Repo


class GateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workdir = Path(self._tmp.name)
        results = self.workdir / "results"
        results.mkdir()
        (results / "claim_ledger.json").write_text(json.dumps({"metric": "acc"}))

    def _run(self, harness, llm, *, secret=_SECRET):
        with patch("meta_harness.manuscript_gate.db.fetchone", harness.fetchone), patch(
            "meta_harness.manuscript_gate.db.commit", harness.commit
        ), patch(
            "meta_harness.manuscript_gate.db.rollback", harness.rollback
        ), patch(
            "meta_harness.manuscript_gate.call_llm_for_role", llm
        ), patch(
            "meta_harness.manuscript_gate.configured_role_prompt_version",
            return_value="reviewer_v1",
        ), patch(
            "meta_harness.manuscript_gate.MetaHarnessRepository", harness.repository()
        ):
            return run_manuscript_gate(
                harness.run_row(), secret=secret, log=lambda *a: None
            )

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
        self.assertTrue(call["context"].resource_grant_valid)
        self.assertEqual(call["context"].resource_grant_id, 900)
        self.assertTrue(call["context"].execution_succeeded)
        self.assertEqual(
            call["context"].reviewer_approval["purpose"], MANUSCRIPT_PURPOSE
        )
        # public_record() carries signature_hash, not signature; sending that
        # would verify as an incomplete envelope.
        self.assertIn("signature", call["context"].reviewer_approval)
        self.assertFalse(call["commit"])
        self.assertFalse(harness.records[0]["commit"])
        self.assertEqual(harness.commits, 2)

    def test_request_uses_the_actual_grant_stage(self):
        harness = _GateHarness(self.workdir, grant_stage="manuscript")
        # A prior released reservation may have no route observation.  The
        # allocator must still move on instead of reusing its occupied key.
        harness.attempts = 1
        captured = {}

        def answer(*args, **kwargs):
            captured.update(kwargs)
            return self._answer(False)(*args, **kwargs)

        self.assertEqual(self._run(harness, answer), "refused")
        self.assertEqual(captured["stage"], "manuscript")
        self.assertIn(":run207:", captured["idempotency_key"])
        self.assertTrue(captured["idempotency_key"].endswith(":a2"))
        self.assertEqual(captured["prompt_version"], "reviewer_v1")
        self.assertEqual(harness.records[0]["prompt_ref"], "reviewer_v1")
        self.assertEqual(len(harness.completed), 1)
        self.assertFalse(harness.completed[0]["commit"])

    def test_cached_approval_resumes_without_another_llm_call(self):
        harness = _GateHarness(self.workdir)
        harness.terminal = {
            "id": 1,
            "disposition": "approved",
            "resource_grant_id": 900,
        }

        def boom(*args, **kwargs):
            raise AssertionError("cached terminal result must not call the reviewer")

        self.assertEqual(self._run(harness, boom), "manuscript_allowed")
        self.assertEqual(len(harness.advanced), 1)

    def test_refusal_leaves_the_run_where_it_was(self):
        harness = _GateHarness(self.workdir)
        self.assertEqual(self._run(harness, self._answer(False)), "refused")
        self.assertEqual(harness.advanced, [])
        self.assertFalse(harness.records[0]["commit"])

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
        self.assertEqual(self._run(harness, self._answer(True)), "technical_failed")
        self.assertEqual(harness.advanced, [])

    def test_two_pre_reservation_failures_are_terminal_and_settle(self):
        harness = _GateHarness(self.workdir, grant_stage="manuscript")

        def boom(*args, **kwargs):
            raise AssertionError("missing signing secret must fail before provider")

        self.assertEqual(
            self._run(harness, boom, secret=""),
            "review_failed",
        )
        self.assertIsNone(harness.terminal)
        self.assertEqual(
            self._run(harness, boom, secret=""),
            "technical_failed",
        )
        self.assertEqual(harness.attempts, 2)
        self.assertEqual(harness.records[0]["disposition"], "technical_failed")
        self.assertNotIn("grant_usage_reservation_id", harness.records[0])
        self.assertEqual(len(harness.completed), 1)

    def test_a_missing_ledger_does_not_advance(self):
        (self.workdir / "results" / "claim_ledger.json").unlink()
        harness = _GateHarness(self.workdir)
        self.assertEqual(self._run(harness, self._answer(True)), "review_failed")
        self.assertEqual(harness.attempts, 1)
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
    def test_success_without_settled_usage_is_rejected(self):
        from meta_harness.manuscript_gate import review_manuscript_readiness

        with patch(
            "meta_harness.manuscript_gate.call_llm_for_role",
            return_value=(
                '{"concur": true}',
                10,
                {"provider": "p", "model": "m"},
            ),
        ), patch("meta_harness.manuscript_gate.db.fetchone", return_value=None):
            with self.assertRaises(ManuscriptGateError) as caught:
                review_manuscript_readiness(
                    agenda_id=10, idea_id=42, resource_grant_id=900,
                    grant_stage="manuscript", attempt_key="attempt:a1",
                    prompt_ref="reviewer_v1",
                    decision={}, ledger={}, holdout=None, verdict_hash=_HASH,
                )
        self.assertIn("usage was not settled", str(caught.exception))


class EvidenceAuditCrashRecoveryTests(unittest.TestCase):
    def test_manuscript_allowed_run_idempotently_settles_audit_grant(self):
        from meta_harness import evidence_audit

        run = {
            "id": 207,
            "agenda_id": 10,
            "deep_insight_id": 42,
            "resource_grant_id": 900,
            "scientific_evidence_state": "manuscript_allowed",
        }
        with patch.object(evidence_audit.db, "fetchone", return_value=run), patch.object(
            evidence_audit, "_settle_completed_grants"
        ) as settle:
            status = evidence_audit.run_evidence_audit_phase(
                agenda_id=10,
                idea_id=42,
                run_id=207,
                resource_grant_id=900,
                log=lambda *args: None,
            )

        self.assertEqual(status, "decided")
        settle.assert_called_once_with(run, log=ANY)


if __name__ == "__main__":
    unittest.main()
