"""Pure contract/policy tests. No database, app startup, or provider calls."""

from __future__ import annotations

import hashlib
import hmac
import unittest
from unittest import mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agents.agenda_selector import select_next
from contracts.agenda import ResearchAgenda
from contracts.base import ContractValidationError
from contracts.meta_harness import (
    Estimate,
    FrontierPacket,
    IdeaDecisionPacket,
    ResourceGrant,
)
from meta_harness.evidence_state import (
    EvidenceTransitionContext,
    EvidenceTransitionError,
    advance,
)
from meta_harness.frontier import evaluate_frontier
from meta_harness.frontier_builder import (
    FrontierBuildError,
    RetrievalSnapshot,
    build_frontier_packet,
)
from meta_harness.candidate_workspace import CandidateSandbox
from meta_harness.grants import GrantDeniedError, ResourceRequest, authorize
from meta_harness.harness_evolution import (
    EvaluationRun,
    HarnessCandidate,
    HarnessPatch,
    HarnessPolicy,
    HarnessPolicyError,
    evaluate_candidate,
    validate_candidate,
    validate_patch,
)
from meta_harness.portfolio import PortfolioPolicy, decide_portfolio
from meta_harness.reviewer_approval import (
    ReviewerApproval,
    ReviewerApprovalError,
    ReviewerApprovalVerifier,
    harness_candidate_subject,
)


def _estimate(value: float, *, lower: float = 0.0, upper: float = 1.0) -> Estimate:
    return Estimate(
        value=value,
        lower=lower,
        upper=upper,
        evaluator="judge-v1",
        provider="provider-b",
        model="model-b",
        evidence_sources=["frontier:1"],
    )


def _decision(idea_id: int, *, obsolete: float = 0.1) -> IdeaDecisionPacket:
    return IdeaDecisionPacket(
        agenda_id=11,
        idea_id=idea_id,
        frontier_packet_id=31,
        expected_impact=_estimate(0.7),
        success_probability=_estimate(0.6),
        novelty=_estimate(0.7),
        obsolescence_probability=_estimate(obsolete),
        falsification_value=_estimate(0.8),
        reuse_value=_estimate(0.4),
        expected_token_cost=_estimate(20_000, lower=10_000, upper=30_000),
        expected_gpu_cost=_estimate(0.2, lower=0, upper=0.5),
        time_to_feedback=_estimate(4, lower=1, upper=12),
        execution_risk=_estimate(0.2),
        information_value=_estimate(0.8),
        candidate_family=f"family-{idea_id}",
        correlation_keys=[f"mechanism:{idea_id}"],
        decision="park",
        reason_codes=["awaiting_portfolio"],
        revisit_condition={"on": ["budget_available"]},
    )


class AgendaContractTests(unittest.TestCase):
    def test_zero_token_budget_is_never_unlimited(self):
        agenda = ResearchAgenda(
            name="bounded",
            focus=["robustness"],
            token_budget=0,
            backend_allowlist=["cpu", "llm"],
        )
        with self.assertRaises(ContractValidationError):
            agenda.validate()

    def test_selector_never_reads_another_agenda(self):
        class FakeRepository:
            def __init__(self):
                self.saved = []

            def get(self, agenda_id):
                return ResearchAgenda(
                    agenda_id=agenda_id,
                    name="scope",
                    focus=["robustness"],
                    token_budget=1000,
                    backend_allowlist=["cpu", "llm"],
                )

            def candidates(self, agenda_id, *, limit):
                return [
                    {
                        "id": 7,
                        "agenda_id": agenda_id,
                        "title": "robustness benchmark",
                        "resource_class": "cpu",
                    }
                ]

            def save_selection(self, selection):
                selection.selection_id = 3
                self.saved.append(selection)

            def queue_selected_insight(self, selection):
                selection.auto_research_job_id = 5
                selection.status = "awaiting_portfolio_decision"
                return 5

        repository = FakeRepository()
        selection = select_next(11, repository=repository)
        self.assertEqual(selection.agenda_id, 11)
        self.assertEqual(selection.selected_insight_id, 7)


class FrontierAndPortfolioTests(unittest.TestCase):
    def test_frontier_builder_requires_auditable_query_refs(self):
        snapshot = RetrievalSnapshot(
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            date_start="2025-01-01",
            date_end="2026-07-30",
            source_indexes=("primary-index",),
            query_refs=(),
            strongest_recent_work=({"id": "paper-a"},),
            latest_benchmarks=({"id": "bench-a"},),
            nearest_prior_art=({"id": "paper-a"},),
        )
        with self.assertRaises(FrontierBuildError):
            build_frontier_packet(
                agenda_id=11,
                research_problem_id=21,
                snapshot=snapshot,
                problem_status="open",
                contribution_delta={"delta": "new mechanism"},
                why_not_obsolete="The mechanism is not tested by prior work.",
                minimum_falsification_experiment={"name": "reject mechanism"},
                evaluator="frontier-judge",
                provider="provider-b",
                model="model-b",
                prompt_version="frontier-v1",
            )

    def test_frontier_gate_rejects_obsolete_problem(self):
        packet = FrontierPacket(
            agenda_id=11,
            research_problem_id=21,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            coverage={"sources": ["primary-index"]},
            problem_status="obsolete",
            strongest_recent_work=[{"id": "paper-a"}],
            latest_benchmarks=[{"id": "bench-a"}],
            nearest_prior_art=[{"id": "paper-a"}],
            contribution_delta={"delta": "none"},
            obsolete_or_duplicate_evidence=[{"reason": "already solved"}],
            why_not_obsolete="No defensible distinction remains.",
            minimum_falsification_experiment={"name": "small rejection test"},
            evaluator="frontier-judge",
            provider="provider-b",
            model="model-b",
            prompt_version="frontier-v1",
        )
        decision = evaluate_frontier(packet)
        self.assertFalse(decision.allowed)
        self.assertIn("frontier_obsolete", decision.reason_codes)

    def test_killed_signature_prevents_regeneration(self):
        packet = _decision(41)
        decided = decide_portfolio(
            [packet],
            killed_signatures={"mechanism:41"},
            policy=PortfolioPolicy(promote_count=1),
        )
        self.assertEqual(decided[0].decision, "kill")
        self.assertIn("similar_to_previously_killed_idea", decided[0].reason_codes)

    def test_parked_idea_has_revisit_trigger(self):
        packets = [_decision(41), _decision(42)]
        decided = decide_portfolio(
            packets,
            policy=PortfolioPolicy(promote_count=1),
        )
        parked = [packet for packet in decided if packet.decision == "park"]
        self.assertEqual(len(parked), 1)
        self.assertTrue(parked[0].revisit_condition or parked[0].revisit_after)


class ResourceGrantAndEvidenceTests(unittest.TestCase):
    def _grant(self) -> ResourceGrant:
        return ResourceGrant(
            agenda_id=11,
            idea_id=41,
            decision_packet_id=51,
            stage="pilot",
            token_cap=1000,
            gpu_class="small",
            max_gpu_hours=0.25,
            backend_allowlist=["cpu", "local_gpu", "llm"],
            artifact_requirements=["raw_metrics", "claim_ledger"],
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            grant_reason="portfolio_score_selected",
            idempotency_key="grant-41-pilot",
            grant_id=61,
        )

    def test_gpu_request_without_grant_is_denied(self):
        with self.assertRaises(GrantDeniedError):
            authorize(
                None,
                ResourceRequest(11, 41, "pilot", "local_gpu", gpu_hours=0.1),
            )

    def test_grant_cannot_cross_agenda(self):
        with self.assertRaises(GrantDeniedError):
            authorize(
                self._grant(),
                ResourceRequest(12, 41, "pilot", "local_gpu", gpu_hours=0.1),
            )

    def test_pilot_cannot_become_full_benchmark(self):
        with self.assertRaises(EvidenceTransitionError):
            advance(
                "sanity_passed",
                "full_benchmark_complete",
                EvidenceTransitionContext(
                    resource_grant_valid=True,
                    execution_succeeded=True,
                    pilot_only=True,
                    raw_artifacts_present=True,
                    full_benchmark_complete=True,
                ),
            )

    def test_failure_cannot_advance_evidence(self):
        with self.assertRaises(EvidenceTransitionError):
            advance(
                "planned",
                "sanity_passed",
                EvidenceTransitionContext(
                    resource_grant_valid=True,
                    execution_succeeded=False,
                    raw_artifacts_present=True,
                ),
            )

    def test_evidence_audit_requires_content_addressed_holdout_inputs(self):
        with self.assertRaises(EvidenceTransitionError):
            advance(
                "full_benchmark_complete",
                "evidence_audited",
                EvidenceTransitionContext(
                    resource_grant_valid=True,
                    execution_succeeded=True,
                    raw_artifacts_present=True,
                    claim_ledger_present=True,
                    evaluator_passed=True,
                    holdout_passed=True,
                ),
            )

    def test_supported_decision_requires_integrity_decision(self):
        digest = "a" * 64
        with self.assertRaisesRegex(
            EvidenceTransitionError,
            "positive_evidence_decision_failed",
        ):
            advance(
                "evidence_audited",
                "scientifically_decided",
                EvidenceTransitionContext(
                    resource_grant_valid=True,
                    execution_succeeded=True,
                    verdict="supported",
                    verdict_hash=digest,
                    raw_artifacts_hash=digest,
                    claim_ledger_hash=digest,
                    benchmark_contract_hash=digest,
                    evaluator_hash=digest,
                    holdout_hash=digest,
                    evaluator_ref="test:evaluator",
                    holdout_ref="test:holdout",
                    evidence_decision_passed=False,
                ),
            )


class HarnessIsolationTests(unittest.TestCase):
    def test_candidate_environment_does_not_inherit_secrets_or_production_db(self):
        candidate = HarnessCandidate(
            agenda_id=11,
            candidate_ref="candidate-1",
            base_commit="a" * 40,
            worktree_path="/tmp/meta-harness-candidates/candidate-1",
            database_namespace="meta_harness_candidate_1",
            artifact_namespace="meta_harness_candidate_1/artifacts",
        )
        sandbox = CandidateSandbox(
            candidate=candidate,
            database_url_ref="env:DEEPGRAPH_CANDIDATE_DATABASE_URL",
            artifact_root="/tmp/meta-harness-artifacts/candidate-1",
            evaluator_root="/readonly/evaluator",
            holdout_root="/readonly/holdout",
            policy_root="/readonly/policy",
        )
        environment = sandbox.candidate_environment(
            {
                "HOME": "/unsafe",
                "DEEPGRAPH_DATABASE_URL": "postgresql://production",
                "OPENAI_API_KEY": "secret",
                "PUBLIC_SETTING": "kept",
            }
        )
        self.assertNotIn("HOME", environment)
        self.assertNotIn("DEEPGRAPH_DATABASE_URL", environment)
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertEqual(environment["PUBLIC_SETTING"], "kept")
        self.assertEqual(environment["DEEPGRAPH_PROTECTED_INPUTS_READ_ONLY"], "1")

    def test_candidate_cannot_overlap_production(self):
        policy = HarnessPolicy(candidate_root="/tmp/meta-harness-candidates")
        candidate = HarnessCandidate(
            agenda_id=11,
            candidate_ref="candidate-1",
            base_commit="a" * 40,
            worktree_path="/home/billion-token/Deepgraph",
            database_namespace="meta_harness_candidate_1",
            artifact_namespace="meta_harness_candidate_1/artifacts",
        )
        with self.assertRaises(HarnessPolicyError):
            validate_candidate(
                candidate,
                policy=policy,
                production_path="/home/billion-token/Deepgraph",
                production_database_namespace="deepgraph",
            )

    def test_candidate_cannot_modify_holdout_or_policy(self):
        policy = HarnessPolicy(candidate_root="/tmp/meta-harness-candidates")
        changed = ("meta_harness/held_out/secret_cases.json",)
        material = "\n".join(["a" * 40, *changed, "10", "2"]).encode()
        patch = HarnessPatch(
            agenda_id=11,
            candidate_ref="candidate-1",
            base_commit="a" * 40,
            changed_paths=changed,
            added_lines=10,
            deleted_lines=2,
            patch_hash=hashlib.sha256(material).hexdigest(),
        )
        with self.assertRaises(HarnessPolicyError):
            validate_patch(patch, policy=policy)

    def test_all_three_suites_still_require_reviewer(self):
        runs = [
            EvaluationRun(
                agenda_id=11,
                suite=name,
                status="passed",
                evaluator_ref=f"readonly:{name}",
                evaluator_hash=f"{name}-hash",
                artifact_manifest={"files": ["result"]},
            )
            for name in ("held_in", "held_out", "canary")
        ]
        report = evaluate_candidate(runs)
        self.assertEqual(report.decision, "awaiting_approval")

    def test_harness_approval_is_signed_and_subject_bound(self):
        runs = [
            EvaluationRun(
                agenda_id=11,
                suite=name,
                status="passed",
                evaluator_ref=f"readonly:{name}",
                evaluator_hash=f"{name}-hash",
                artifact_manifest={"files": ["result"]},
            )
            for name in ("held_in", "held_out", "canary")
        ]
        secret = b"isolated-test-secret"
        subject = harness_candidate_subject(
            agenda_id=11,
            candidate_id=41,
            patch_hash="a" * 64,
        )
        unsigned = ReviewerApproval(
            reviewer_id="reviewer-1",
            key_id="test-key",
            purpose="harness_upgrade",
            subject=subject,
            issued_at=datetime.now(timezone.utc).isoformat(),
            signature="pending",
        )
        approval = ReviewerApproval(
            **{
                **unsigned.__dict__,
                "signature": hmac.new(
                    secret,
                    unsigned.signing_payload(),
                    hashlib.sha256,
                ).hexdigest(),
            }
        )
        verifier = ReviewerApprovalVerifier(
            {"test-key": "env:DEEPGRAPH_TEST_REVIEWER_SECRET"}
        )
        with mock.patch.dict(
            "os.environ",
            {"DEEPGRAPH_TEST_REVIEWER_SECRET": secret.decode()},
        ):
            report = evaluate_candidate(
                runs,
                candidate_id=41,
                patch_hash="a" * 64,
                reviewer_approval=approval,
                approval_verifier=verifier,
            )
        self.assertEqual(report.decision, "approved")
        self.assertEqual(report.reviewer, "reviewer-1")

        wrong_subject = ReviewerApproval(
            **{**approval.__dict__, "subject": subject + ":other"}
        )
        with mock.patch.dict(
            "os.environ",
            {"DEEPGRAPH_TEST_REVIEWER_SECRET": secret.decode()},
        ):
            with self.assertRaises(ReviewerApprovalError):
                evaluate_candidate(
                    runs,
                    candidate_id=41,
                    patch_hash="a" * 64,
                    reviewer_approval=wrong_subject,
                    approval_verifier=verifier,
                )

    def test_cross_agenda_evaluations_are_rejected(self):
        runs = [
            EvaluationRun(
                agenda_id=11 if name != "canary" else 12,
                suite=name,
                status="passed",
                evaluator_ref=f"readonly:{name}",
                evaluator_hash=f"{name}-hash",
                artifact_manifest={"files": ["result"]},
            )
            for name in ("held_in", "held_out", "canary")
        ]
        report = evaluate_candidate(runs)
        self.assertEqual(report.decision, "reject")
        self.assertIn("cross_agenda_evaluation", report.blockers)


class StaticMigrationContractTests(unittest.TestCase):
    def test_migration_is_additive_and_scopes_legacy_rows(self):
        root = Path(__file__).resolve().parents[1]
        sql = (root / "db/migrations/0001_meta_harness_v1.sql").read_text(
            encoding="utf-8"
        )
        upper = sql.upper()
        self.assertNotIn("DROP TABLE", upper)
        self.assertNotIn("TRUNCATE", upper)
        self.assertNotIn("DELETE FROM", upper)
        self.assertIn(
            "ALTER TABLE IF EXISTS DEEP_INSIGHTS ADD COLUMN IF NOT EXISTS AGENDA_ID",
            upper,
        )
        self.assertIn(
            "ALTER TABLE IF EXISTS DEEP_INSIGHTS ADD COLUMN IF NOT EXISTS RESEARCH_PROBLEM_ID",
            upper,
        )
        self.assertNotIn("UPDATE DEEP_INSIGHTS SET AGENDA_ID", upper)


if __name__ == "__main__":
    unittest.main()
