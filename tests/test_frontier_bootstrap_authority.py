"""The bootstrap authority must break the deadlock without widening authority."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from contracts.base import ContractValidationError
from contracts.meta_harness import FrontierEvaluationAuthority
from meta_harness.frontier_authority import (
    FrontierAuthorityError,
    FrontierEvaluationRequest,
    authorize,
)
from meta_harness import frontier_bootstrap
from meta_harness.frontier_bootstrap import (
    BootstrapUsage,
    FrontierBootstrapError,
    run_bootstrap_evaluation,
)


NOW = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)


def _authority(**overrides) -> FrontierEvaluationAuthority:
    values = {
        "agenda_id": 5,
        "research_problem_id": 7,
        "token_cap": 8_000,
        "issued_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(minutes=30)).isoformat(),
        "idempotency_key": "agenda-5-problem-7-bootstrap-1",
        "provider": "provider-b",
        "model": "model-b",
        "model_family": "family-b",
        "prompt_version": "frontier-bootstrap-v1",
        "evaluator": "frontier-bootstrap-evaluator",
        "issued_by": "operator:recovery",
        "issue_reason": "agenda 5 has no frontier packet yet",
        "authority_id": 31,
        "reservation_id": 91,
    }
    values.update(overrides)
    return FrontierEvaluationAuthority(**values)


def _request(**overrides) -> FrontierEvaluationRequest:
    values = {
        "agenda_id": 5,
        "research_problem_id": 7,
        "operation": "frontier_assessment",
        "token_cap": 8_000,
    }
    values.update(overrides)
    return FrontierEvaluationRequest(**values)


class AuthorityContractTests(unittest.TestCase):
    def test_authority_is_bounded_by_hard_ceilings(self):
        with self.assertRaises(ContractValidationError):
            _authority(token_cap=20_001).validate()
        with self.assertRaises(ContractValidationError):
            _authority(token_cap=0).validate()
        with self.assertRaises(ContractValidationError):
            _authority(
                expires_at=(NOW + timedelta(hours=6)).isoformat()
            ).validate()
        with self.assertRaises(ContractValidationError):
            _authority(expires_at=NOW.isoformat()).validate()

    def test_authority_can_never_reach_gpu_or_a_backend(self):
        authority = _authority()

        self.assertEqual(authority.backend_allowlist, ("llm",))
        self.assertEqual(authority.max_gpu_hours, 0.0)
        self.assertEqual(authority.allowed_operations, ("frontier_assessment",))

    def test_authority_requires_full_provenance(self):
        for field in ("provider", "model", "model_family", "prompt_version",
                      "evaluator", "issued_by", "issue_reason",
                      "idempotency_key"):
            with self.subTest(field=field), self.assertRaises(
                ContractValidationError
            ):
                _authority(**{field: ""}).validate()


class AuthorityAdmissionTests(unittest.TestCase):
    def test_missing_authority_is_denied(self):
        with self.assertRaisesRegex(
            FrontierAuthorityError, "frontier_evaluation_authority_required"
        ):
            authorize(None, _request(), now=NOW)

    def test_scoped_authority_admits_its_own_problem(self):
        self.assertIs(
            authorize(_authority(), _request(), now=NOW).authority_id, 31
        )

    def test_another_agenda_or_problem_is_denied(self):
        with self.assertRaisesRegex(FrontierAuthorityError, "agenda_scope_mismatch"):
            authorize(_authority(), _request(agenda_id=6), now=NOW)
        with self.assertRaisesRegex(
            FrontierAuthorityError, "research_problem_scope_mismatch"
        ):
            authorize(_authority(), _request(research_problem_id=8), now=NOW)

    def test_expired_or_closed_authority_is_denied(self):
        with self.assertRaisesRegex(FrontierAuthorityError, "authority_expired"):
            authorize(_authority(), _request(), now=NOW + timedelta(hours=2))
        with self.assertRaisesRegex(FrontierAuthorityError, "authority_consumed"):
            authorize(_authority(status="consumed"), _request(), now=NOW)
        with self.assertRaisesRegex(FrontierAuthorityError, "authority_revoked"):
            authorize(_authority(status="revoked"), _request(), now=NOW)

    def test_other_operations_gpu_and_backends_are_denied(self):
        with self.assertRaisesRegex(FrontierAuthorityError, "operation_not_allowed"):
            authorize(_authority(), _request(operation="experiment_run"), now=NOW)
        with self.assertRaisesRegex(FrontierAuthorityError, "gpu_not_allowed"):
            authorize(_authority(), _request(gpu_hours=0.1), now=NOW)
        with self.assertRaisesRegex(FrontierAuthorityError, "backend_not_allowed"):
            authorize(_authority(), _request(backend="ssh_gpu"), now=NOW)

    def test_token_cap_cannot_be_widened(self):
        with self.assertRaisesRegex(FrontierAuthorityError, "token_cap_exceeded"):
            authorize(_authority(), _request(token_cap=8_001), now=NOW)

    def test_evaluator_must_be_independent_of_the_proposer(self):
        with self.assertRaisesRegex(
            FrontierAuthorityError, "evaluator_not_independent_of_proposer"
        ):
            authorize(
                _authority(),
                _request(
                    proposer_provider="provider-b",
                    proposer_model_family="family-b",
                ),
                now=NOW,
            )
        # A different family on the same provider is still independent enough
        # for the router's existing rule.
        authorize(
            _authority(),
            _request(
                proposer_provider="provider-b",
                proposer_model_family="family-a",
            ),
            now=NOW,
        )


class _FakeAuthorityRepository:
    def __init__(self, authority: FrontierEvaluationAuthority):
        self.authority = authority
        self.usage: list[dict] = []
        self.settlements: list[dict] = []
        self.completed: int | None = None

    def load(self, authority_id, *, agenda_id, research_problem_id):
        if (
            int(agenda_id) != self.authority.agenda_id
            or int(research_problem_id) != self.authority.research_problem_id
        ):
            raise FrontierAuthorityError("scoped frontier authority not found")
        return self.authority

    def completed_packet_id(self, authority_id, *, agenda_id):
        return self.completed

    def record_usage(self, **kwargs):
        self.usage.append(kwargs)
        return len(self.usage)

    def settle(self, authority, *, tokens_used, cost_usd, outcome):
        self.settlements.append({"tokens_used": tokens_used, "outcome": outcome})


class _FakeSource:
    def __init__(self, *, briefing=None, error=None, packet=None):
        self._briefing = briefing or {
            "agenda_id": 5,
            "research_problem_id": 7,
            "problem_statement": "does X hold",
            "papers": [{"paper_id": "p1"}],
            "benchmarks": [],
            "negative_evidence": [],
            "query_ref": "deepgraph:evidence-graph:sha256:abc",
        }
        self._error = error
        self._packet = packet
        self.built_with = None

    def evidence_briefing(self, *, agenda_id, research_problem_id):
        if self._error:
            raise self._error
        return dict(self._briefing)

    def build(self, *, agenda_id, research_problem_id, assessment):
        self.built_with = assessment
        if self._packet is None:
            raise AssertionError("build should not be reached")
        return self._packet


class _FakePackets:
    def __init__(self):
        self.saved = []

    def save_frontier(self, packet):
        self.saved.append(packet)
        return 77


def _packet():
    from contracts.meta_harness import FrontierPacket

    return FrontierPacket(
        agenda_id=5,
        research_problem_id=7,
        retrieved_at=NOW.isoformat(),
        coverage={"query_refs": ["deepgraph:evidence-graph:sha256:abc"]},
        problem_status="open",
        strongest_recent_work=[{"paper_id": "p1"}],
        latest_benchmarks=[{"id": 1}],
        nearest_prior_art=[{"paper_id": "p1"}],
        contribution_delta={"claim": "new mechanism"},
        why_not_obsolete="prior work does not test this mechanism",
        minimum_falsification_experiment={"metric": "accuracy"},
        evaluator="frontier-bootstrap-evaluator",
        provider="provider-b",
        model="model-b",
        prompt_version="frontier-bootstrap-v1",
    )


def _good_output() -> dict:
    return {
        "problem_status": "open",
        "contribution_delta": {"claim": "new mechanism", "versus": "prior art"},
        "why_not_obsolete": "prior work does not test this mechanism",
        "minimum_falsification_experiment": {
            "metric": "accuracy",
            "baseline": "dense",
            "decisive_comparison": "paired seeds",
        },
        "coverage_start": "2025-01-01",
        "coverage_end": "2026-08-01",
    }


class BootstrapRunTests(unittest.TestCase):
    def _run(self, executor, *, authority=None, source=None):
        authorities = _FakeAuthorityRepository(authority or _authority())
        packets = _FakePackets()
        result = run_bootstrap_evaluation(
            authority_id=31,
            agenda_id=5,
            research_problem_id=7,
            executor=executor,
            authority_repository=authorities,
            source=source or _FakeSource(packet=_packet()),
            frontier_repository=packets,
            now=NOW,
        )
        return result, authorities, packets

    def test_successful_bootstrap_produces_a_gated_packet_and_ledger(self):
        def executor(call):
            self.assertIn("INDEPENDENT Frontier evaluator", call.system_prompt)
            self.assertIn("query_ref", call.user_prompt)
            self.assertEqual(call.token_cap, 8_000)
            return _good_output(), BootstrapUsage(output_tokens=1_200)

        result, authorities, packets = self._run(executor)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["frontier_packet_id"], 77)
        self.assertTrue(result["gate_allowed"], result["gate_reason_codes"])
        self.assertEqual(result["tokens_used"], 1_200)
        self.assertEqual(len(packets.saved), 1)
        self.assertEqual(authorities.usage[-1]["status"], "succeeded")
        self.assertEqual(authorities.usage[-1]["frontier_packet_id"], 77)
        self.assertEqual(authorities.settlements[-1]["outcome"], "consumed")

    def test_provenance_comes_from_the_authority_not_the_model(self):
        def executor(call):
            output = _good_output()
            output.update(
                {
                    "evaluator": "self-declared-human-reviewer",
                    "provider": "provider-a",
                    "model": "model-a",
                    "prompt_version": "made-up",
                }
            )
            return output, BootstrapUsage(output_tokens=10)

        source = _FakeSource(packet=_packet())
        self._run(executor, source=source)

        self.assertEqual(source.built_with.evaluator, "frontier-bootstrap-evaluator")
        self.assertEqual(source.built_with.provider, "provider-b")
        self.assertEqual(source.built_with.model, "model-b")
        self.assertEqual(source.built_with.prompt_version, "frontier-bootstrap-v1")

    def test_unavailable_provider_fails_closed_and_settles(self):
        def executor(call):
            raise RuntimeError("provider down")

        authorities = _FakeAuthorityRepository(_authority())
        with self.assertRaisesRegex(FrontierBootstrapError, "no fallback"):
            run_bootstrap_evaluation(
                authority_id=31,
                agenda_id=5,
                research_problem_id=7,
                executor=executor,
                authority_repository=authorities,
                source=_FakeSource(packet=_packet()),
                frontier_repository=_FakePackets(),
                now=NOW,
            )

        self.assertEqual(authorities.usage[-1]["status"], "failed")
        self.assertIn("provider_unavailable", authorities.usage[-1]["failure_reason"])
        self.assertEqual(authorities.settlements[-1]["outcome"], "revoked")

    def test_malformed_output_fails_closed(self):
        for output in ("not json", {"problem_status": "open"}, [], 42):
            with self.subTest(output=output):
                authorities = _FakeAuthorityRepository(_authority())
                packets = _FakePackets()
                with self.assertRaises(FrontierBootstrapError):
                    run_bootstrap_evaluation(
                        authority_id=31,
                        agenda_id=5,
                        research_problem_id=7,
                        executor=lambda call: (output, BootstrapUsage(output_tokens=5)),
                        authority_repository=authorities,
                        source=_FakeSource(packet=_packet()),
                        frontier_repository=packets,
                        now=NOW,
                    )
                self.assertEqual(packets.saved, [])
                self.assertEqual(authorities.settlements[-1]["outcome"], "revoked")

    def test_missing_linked_evidence_fails_closed_before_any_call(self):
        from meta_harness.frontier_builder import FrontierBuildError

        called = []
        authorities = _FakeAuthorityRepository(_authority())
        with self.assertRaisesRegex(FrontierBootstrapError, "linked evidence"):
            run_bootstrap_evaluation(
                authority_id=31,
                agenda_id=5,
                research_problem_id=7,
                executor=lambda call: called.append(call),
                authority_repository=authorities,
                source=_FakeSource(error=FrontierBuildError("no linked papers")),
                frontier_repository=_FakePackets(),
                now=NOW,
            )

        self.assertEqual(called, [])
        self.assertEqual(authorities.usage[-1]["input_tokens"], 0)
        self.assertEqual(authorities.settlements[-1]["outcome"], "revoked")

    def test_usage_above_the_cap_is_rejected(self):
        authorities = _FakeAuthorityRepository(_authority())
        packets = _FakePackets()
        with self.assertRaisesRegex(FrontierBootstrapError, "exceeded the authority cap"):
            run_bootstrap_evaluation(
                authority_id=31,
                agenda_id=5,
                research_problem_id=7,
                executor=lambda call: (
                    _good_output(),
                    BootstrapUsage(output_tokens=8_001),
                ),
                authority_repository=authorities,
                source=_FakeSource(packet=_packet()),
                frontier_repository=packets,
                now=NOW,
            )
        self.assertEqual(packets.saved, [])
        self.assertEqual(authorities.usage[-1]["failure_reason"], "token_cap_exceeded")

    def test_expired_authority_never_calls_the_provider(self):
        called = []
        with self.assertRaisesRegex(FrontierAuthorityError, "authority_expired"):
            run_bootstrap_evaluation(
                authority_id=31,
                agenda_id=5,
                research_problem_id=7,
                executor=lambda call: called.append(call),
                authority_repository=_FakeAuthorityRepository(_authority()),
                source=_FakeSource(packet=_packet()),
                frontier_repository=_FakePackets(),
                now=NOW + timedelta(hours=3),
            )
        self.assertEqual(called, [])

    def test_consumed_authority_replays_instead_of_spending_again(self):
        authorities = _FakeAuthorityRepository(_authority(status="consumed"))
        authorities.completed = 77
        called = []

        result = run_bootstrap_evaluation(
            authority_id=31,
            agenda_id=5,
            research_problem_id=7,
            executor=lambda call: called.append(call),
            authority_repository=authorities,
            source=_FakeSource(packet=_packet()),
            frontier_repository=_FakePackets(),
            now=NOW,
        )

        self.assertEqual(result["status"], "already_completed")
        self.assertEqual(result["frontier_packet_id"], 77)
        self.assertEqual(called, [])
        self.assertEqual(authorities.usage, [])

    def test_proposer_route_cannot_be_relabelled_as_the_evaluator(self):
        called = []
        with self.assertRaisesRegex(
            FrontierAuthorityError, "evaluator_not_independent_of_proposer"
        ):
            run_bootstrap_evaluation(
                authority_id=31,
                agenda_id=5,
                research_problem_id=7,
                proposer_provider="provider-b",
                proposer_model_family="family-b",
                executor=lambda call: called.append(call),
                authority_repository=_FakeAuthorityRepository(_authority()),
                source=_FakeSource(packet=_packet()),
                frontier_repository=_FakePackets(),
                now=NOW,
            )
        self.assertEqual(called, [])


class AuthorityPersistenceTests(unittest.TestCase):
    def test_issue_refuses_an_inactive_agenda(self):
        from meta_harness.frontier_authority import (
            FrontierAuthorityPersistenceError,
            FrontierAuthorityRepository,
        )

        with mock.patch(
            "meta_harness.frontier_authority.db.fetchone",
            side_effect=[None, {"id": 5, "status": "paused_budget"}],
        ), mock.patch("meta_harness.frontier_authority.db.rollback"):
            with self.assertRaisesRegex(
                FrontierAuthorityPersistenceError, "agenda is not active"
            ):
                FrontierAuthorityRepository().issue(_authority(authority_id=None))

    def test_issue_refuses_a_problem_from_another_agenda(self):
        from meta_harness.frontier_authority import (
            FrontierAuthorityPersistenceError,
            FrontierAuthorityRepository,
        )

        with mock.patch(
            "meta_harness.frontier_authority.db.fetchone",
            side_effect=[None, {"id": 5, "status": "active"}, None],
        ), mock.patch("meta_harness.frontier_authority.db.rollback"):
            with self.assertRaisesRegex(
                FrontierAuthorityPersistenceError, "not bound to this agenda"
            ):
                FrontierAuthorityRepository().issue(_authority(authority_id=None))

    def test_usage_ledger_rejects_an_invalid_status(self):
        from meta_harness.frontier_authority import (
            FrontierAuthorityPersistenceError,
            FrontierAuthorityRepository,
        )

        with self.assertRaises(FrontierAuthorityPersistenceError):
            FrontierAuthorityRepository().record_usage(
                authority=_authority(),
                operation="frontier_assessment",
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                status="maybe",
            )


class ReservationLifecycleTests(unittest.TestCase):
    """An unused authority must give its reserved tokens back."""

    def test_expiry_releases_the_agenda_reservation(self):
        from meta_harness.frontier_authority import FrontierAuthorityRepository

        releases = []
        agenda_repository = mock.Mock()
        agenda_repository.release.side_effect = lambda rid, *, reason: releases.append(
            (rid, reason)
        )
        with mock.patch(
            "meta_harness.frontier_authority.db.fetchall",
            return_value=[{"id": 31, "reservation_id": 91}],
        ), mock.patch(
            "agents.agenda_repository.AgendaRepository", return_value=agenda_repository
        ), mock.patch("meta_harness.frontier_authority.db.execute"), mock.patch(
            "meta_harness.frontier_authority.db.commit"
        ):
            expired = FrontierAuthorityRepository().expire_stale(agenda_id=5)

        self.assertEqual(expired, 1)
        self.assertEqual(releases, [(91, "frontier_authority_expired_unused")])

    def test_nothing_to_expire_touches_no_ledger(self):
        from meta_harness.frontier_authority import FrontierAuthorityRepository

        with mock.patch(
            "meta_harness.frontier_authority.db.fetchall", return_value=[]
        ), mock.patch("meta_harness.frontier_authority.db.execute") as execute:
            self.assertEqual(
                FrontierAuthorityRepository().expire_stale(agenda_id=5), 0
            )
        execute.assert_not_called()

    def test_revoking_an_unused_authority_releases_its_reservation(self):
        from meta_harness.frontier_authority import FrontierAuthorityRepository

        agenda_repository = mock.Mock()
        with mock.patch(
            "meta_harness.frontier_authority.db.fetchone",
            side_effect=[{"count": 0}, {"id": 31, "reservation_id": 91}],
        ), mock.patch(
            "agents.agenda_repository.AgendaRepository", return_value=agenda_repository
        ), mock.patch("meta_harness.frontier_authority.db.execute"), mock.patch(
            "meta_harness.frontier_authority.db.commit"
        ):
            self.assertTrue(
                FrontierAuthorityRepository().revoke_unused(
                    31, agenda_id=5, reason="no independent evaluator route"
                )
            )
        self.assertEqual(agenda_repository.release.call_count, 1)

    def test_an_authority_that_spent_tokens_cannot_be_revoked_as_unused(self):
        from meta_harness.frontier_authority import (
            FrontierAuthorityPersistenceError,
            FrontierAuthorityRepository,
        )

        with mock.patch(
            "meta_harness.frontier_authority.db.fetchone", return_value={"count": 2}
        ):
            with self.assertRaisesRegex(
                FrontierAuthorityPersistenceError, "already recorded usage"
            ):
                FrontierAuthorityRepository().revoke_unused(
                    31, agenda_id=5, reason="changed my mind"
                )


if __name__ == "__main__":
    unittest.main()


class ProviderFailureDetailTests(unittest.TestCase):
    """A recorded provider failure must say which failure it was.

    Every frontier attempt on 2026-08-10 recorded the bare string
    `provider_unavailable:HTTPStatusError`. Establishing that it was a 429 from
    a saturated relay - transient - rather than a 401 on an exhausted key took
    a live probe against the paid endpoint.
    """

    def test_http_status_is_recorded_when_present(self):
        response = mock.Mock(status_code=429)
        exc = RuntimeError("saturated")
        exc.response = response
        self.assertEqual(
            frontier_bootstrap._provider_failure_detail(exc), "RuntimeError:429"
        )

    def test_detail_falls_back_to_the_exception_type(self):
        self.assertEqual(
            frontier_bootstrap._provider_failure_detail(ValueError("x")), "ValueError"
        )
