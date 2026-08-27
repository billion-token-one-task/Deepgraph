import unittest
import json
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest import mock

from contracts.meta_harness import ResourceGrant
from meta_harness import grant_usage, proposal_checkpoint
from meta_harness.llm_routing import (
    LLMRouteError,
    LLMRouter,
    ProviderRoute,
    RouteRequest,
    RouteUsage,
)
from orchestrator import bounded_proposal
from agents import paper_idea_agent
from agents.candidate_contract import ContractReview, ContractViolation
from agents import problem_first


def _request():
    return bounded_proposal.BoundedProposalRequest(
        job_id=110,
        agenda_id=2,
        idea_id=115,
        resource_grant_id=501,
    )


def _scope(**overrides):
    value = {
        "job_id": 110,
        "agenda_id": 2,
        "deep_insight_id": 115,
        "job_status": "deferred",
        "job_stage": "proposal_generation_granted",
        "job_resource_grant_id": 501,
        "insight_status": "proposal_pending",
        "research_problem_id": 9,
        "grant_stage": "proposal",
        "grant_status": "active",
        "grant_live": 1,
        "token_cap": 32000,
        "max_gpu_hours": 0.0,
        "backend_allowlist_json": '["llm"]',
        "expires_at": "2099-01-01T00:00:00Z",
        "grant_tokens_used": 0,
    }
    value.update(overrides)
    return value


def _exact_problem_scope(**overrides):
    value = {
        "job_id": 110,
        "job_status": "deferred",
        "job_stage": "proposal_generation_granted",
        "resource_grant_id": 501,
        "idea_id": 115,
        "insight_status": "proposal_pending",
        "insight_title": "Bounded persisted problem",
        "insight_problem_statement": "Test a bounded failure mode.",
        "insight_node_ids": '["ml.test"]',
        "insight_paper_ids": '["p1", "p2"]',
        "insight_signal_refs": '{"signals": []}',
        "research_problem_id": 9,
        "problem_statement": "Test a bounded failure mode.",
        "source_signal_ref": '{"table": "claim_method_gaps"}',
        "node_ids": '["ml.test"]',
        "paper_ids": '["p1", "p2"]',
        "ruled_out_approaches": "[]",
        "problem_quality_score": 4.2,
        "problem_status": "open",
        "attempts_count": 0,
        "grant_id": 501,
        "grant_stage": "proposal",
        "grant_status": "active",
        "token_cap": 32000,
        "max_gpu_hours": 0.0,
        "backend_allowlist_json": '["llm"]',
        "grant_live": 1,
    }
    value.update(overrides)
    return value


class BoundedProposalTests(unittest.TestCase):
    def test_authorization_requires_exact_job_scope_and_llm_only_grant(self):
        with mock.patch.object(bounded_proposal.db, "fetchone", return_value=None):
            with self.assertRaisesRegex(
                bounded_proposal.BoundedProposalError, "exact proposal"
            ):
                bounded_proposal.authorize_bounded_proposal(_request())

        with mock.patch.object(
            bounded_proposal.db,
            "fetchone",
            return_value=_scope(backend_allowlist_json='["llm","cpu"]'),
        ):
            with self.assertRaisesRegex(
                bounded_proposal.BoundedProposalError, "llm-only"
            ):
                bounded_proposal.authorize_bounded_proposal(_request())

    def test_exact_candidate_is_stored_then_grant_is_settled(self):
        candidate = {
            "proposal_candidate_id": 115,
            "resource_grant_id": 501,
            "agenda_id": 2,
        }
        repository = mock.Mock()
        repository.complete_proposal_generation.return_value = 1234
        discover = mock.Mock(return_value=[candidate])
        store = mock.Mock(return_value=115)
        with (
            mock.patch.object(bounded_proposal.db, "fetchone", return_value=_scope()),
            mock.patch.object(bounded_proposal, "log_event"),
        ):
            result = bounded_proposal.execute_bounded_proposal(
                _request(),
                actor="ops:controlled-recovery",
                repository=repository,
                discover=discover,
                store=store,
            )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tokens_used, 1234)
        discover.assert_called_once_with(_request())
        store.assert_called_once_with(candidate)
        repository.complete_proposal_generation.assert_called_once_with(
            grant_id=501, agenda_id=2, idea_id=115, target_job_id=110
        )

    def test_post_store_crash_replay_settles_without_another_llm_call(self):
        repository = mock.Mock()
        repository.complete_proposal_generation.return_value = 999
        discover = mock.Mock(side_effect=AssertionError("must not regenerate"))
        store = mock.Mock(side_effect=AssertionError("must not store twice"))
        with (
            mock.patch.object(
                bounded_proposal.db,
                "fetchone",
                return_value=_scope(insight_status="candidate"),
            ),
            mock.patch.object(bounded_proposal, "log_event"),
        ):
            result = bounded_proposal.execute_bounded_proposal(
                _request(),
                actor="ops:controlled-recovery",
                repository=repository,
                discover=discover,
                store=store,
            )
        self.assertEqual(result.tokens_used, 999)
        discover.assert_not_called()
        store.assert_not_called()

    def test_consumed_and_requeued_proposal_is_an_idempotent_noop(self):
        completed = _scope(
            grant_status="consumed",
            grant_live=0,
            job_status="queued",
            job_stage="awaiting_portfolio_decision",
            insight_status="candidate",
            job_resource_grant_id=None,
            grant_tokens_used=321,
        )
        repository = mock.Mock()
        with mock.patch.object(bounded_proposal.db, "fetchone", return_value=completed):
            result = bounded_proposal.execute_bounded_proposal(
                _request(), actor="ops:controlled-recovery", repository=repository
            )
        self.assertEqual(result.status, "already_completed")
        self.assertEqual(result.tokens_used, 321)
        repository.complete_proposal_generation.assert_not_called()

    def test_consumed_proposal_replay_refuses_job_rebound_to_another_grant(self):
        rebound = _scope(
            grant_status="consumed",
            grant_live=0,
            job_status="deferred",
            job_stage="pilot_granted",
            insight_status="candidate",
            job_resource_grant_id=777,
        )
        with mock.patch.object(bounded_proposal.db, "fetchone", return_value=rebound):
            with self.assertRaisesRegex(
                bounded_proposal.BoundedProposalError,
                "not in an executable",
            ):
                bounded_proposal.authorize_bounded_proposal(_request())

    def test_the_bounded_path_loops_on_a_refused_contract(self):
        """The funded path is the one candidates are actually born on.

        The 35 candidates of the 2026-08-25..27 batch all came through here --
        realize_funded_proposals -> execute_bounded_proposal -> exact discovery
        -- so a contract loop wired only into the unfunded discovery loop would
        never run. Two attempts, then a plan that satisfies the contract.
        """
        method = json.dumps(
            {
                "method": {
                    "name": "Bounded Method",
                    "one_line": "A bounded mechanism repair.",
                    "definition": "minimize an exact persisted objective",
                    "why_novel": "This is distinct because it tests the persisted mechanism directly.",
                    "falsification_hook": "Reject when the bounded metric does not improve.",
                }
            }
        )
        experiment = json.dumps({"paper_title": "T", "execution_requirements": {}})
        refused = ContractReview(
            (ContractViolation("metric_contract_unsupported", "pick a real metric"),),
            {},
        )
        reviews = [refused, refused, ContractReview((), {})]
        with (
            mock.patch.object(paper_idea_agent.db, "fetchone", return_value=_exact_problem_scope()),
            mock.patch.object(
                paper_idea_agent,
                "_call_exact_proposal_llm",
                side_effect=[
                    (method, 120, {"model": "method-model"}),
                    (experiment, 10, {"model": "experiment-model"}),
                    (experiment, 10, {"model": "experiment-model"}),
                    (experiment, 10, {"model": "experiment-model"}),
                ],
            ) as exact_call,
            mock.patch.object(
                paper_idea_agent, "configured_role_prompt_version", return_value="v1"
            ),
            mock.patch.object(
                paper_idea_agent, "review_candidate_plan", side_effect=reviews
            ),
        ):
            result = paper_idea_agent.discover_paper_ideas(
                max_problems=1,
                max_papers=1,
                agenda_id=2,
                proposal_job_id=110,
                proposal_candidate_id=115,
                proposal_grant_id=501,
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["llm_calls"], 4)
        self.assertEqual(result[0]["generation_tokens"], 150)
        design_calls = [
            call
            for call in exact_call.call_args_list
            if call.kwargs["operation"].startswith("proposal_experiment_design")
        ]
        self.assertEqual(len(design_calls), 3)
        prompts = [call.kwargs["user_prompt"] for call in design_calls]
        self.assertNotIn("metric_contract_unsupported", prompts[0])
        self.assertIn("metric_contract_unsupported", prompts[1])
        # Each repair is its own operation, so the proposal checkpoint sees a
        # new delivery rather than a changed input to one already delivered --
        # which it refuses, because that is how a crash loop re-bills a step.
        self.assertEqual(
            [call.kwargs["operation"] for call in design_calls],
            [
                "proposal_experiment_design",
                "proposal_experiment_design:repair1",
                "proposal_experiment_design:repair2",
            ],
        )

    def test_an_unsatisfiable_contract_stores_nothing_and_frees_the_grant(self):
        method = json.dumps(
            {
                "method": {
                    "name": "Bounded Method",
                    "one_line": "A bounded mechanism repair.",
                    "definition": "minimize an exact persisted objective",
                    "why_novel": "This is distinct because it tests the persisted mechanism directly.",
                    "falsification_hook": "Reject when the bounded metric does not improve.",
                }
            }
        )
        experiment = json.dumps({"paper_title": "T", "execution_requirements": {}})
        refused = ContractReview(
            (ContractViolation("metric_contract_unsupported", "pick a real metric"),),
            {},
        )
        with (
            mock.patch.object(paper_idea_agent.db, "fetchone", return_value=_exact_problem_scope()),
            mock.patch.object(
                paper_idea_agent,
                "_call_exact_proposal_llm",
                side_effect=[(method, 120, {})] + [(experiment, 10, {})] * 3,
            ),
            mock.patch.object(
                paper_idea_agent, "configured_role_prompt_version", return_value="v1"
            ),
            mock.patch.object(
                paper_idea_agent, "review_candidate_plan", return_value=refused
            ),
            mock.patch.object(
                paper_idea_agent, "_release_abandoned_proposal_grant"
            ) as release,
        ):
            result = paper_idea_agent.discover_paper_ideas(
                max_problems=1,
                max_papers=1,
                agenda_id=2,
                proposal_job_id=110,
                proposal_candidate_id=115,
                proposal_grant_id=501,
            )
        self.assertEqual(result, [])
        release.assert_called_once_with({"id": 501}, 2)

    def test_exact_discovery_uses_only_named_rows_and_no_global_side_path(self):
        method = json.dumps(
            {
                "method": {
                    "name": "Bounded Method",
                    "one_line": "A bounded mechanism repair.",
                    "definition": "minimize an exact persisted objective",
                    "why_novel": "This is distinct because it tests the persisted mechanism directly.",
                    "falsification_hook": "Reject when the bounded metric does not improve.",
                }
            }
        )
        experiment = json.dumps(
            {
                "paper_title": "A Bounded Test of a Persisted Failure Mode",
                "baselines": ["fixed baseline"],
                "datasets": ["materialized fixture"],
                "metrics": {"primary": "accuracy"},
                "ablations": ["remove repair"],
                "expected_results": {"solid": "improvement"},
                "execution_requirements": {"backend": "cpu"},
            }
        )
        global_paths = (
            "get_tier2_signals",
            "signal_refs_from_rows",
            "_recent_tier2_memory",
            "select_problem_first_candidates",
            "discover_research_problems",
            "get_solution_signals",
            "graph_novelty_gate",
            "review_and_refine_tier2_idea",
            "attach_graph_taste_to_insight",
            "enrich_deep_insight",
        )
        with (
            mock.patch.object(paper_idea_agent.db, "fetchone", return_value=_exact_problem_scope()),
            mock.patch.object(
                paper_idea_agent,
                "_call_exact_proposal_llm",
                side_effect=[
                    (method, 120, {"model": "method-model"}),
                    (experiment, 80, {"model": "experiment-model"}),
                ],
            ) as exact_call,
            mock.patch.object(
                paper_idea_agent,
                "configured_role_prompt_version",
                return_value="proposal-v1",
            ),
            # The contract review reaches the hub; this test is about the exact
            # path's row scoping. tests/test_candidate_contract.py covers the
            # review, and test_the_bounded_path_loops_on_a_refused_contract
            # below covers that this path consults it.
            mock.patch.object(
                paper_idea_agent,
                "review_candidate_plan",
                side_effect=lambda plan, **kwargs: ContractReview((), plan),
            ),
        ):
            with ExitStack() as stack:
                for name in global_paths:
                    stack.enter_context(
                        mock.patch.object(
                            paper_idea_agent,
                            name,
                            side_effect=AssertionError(
                                f"exact path called global {name}"
                            ),
                        )
                    )
                result = paper_idea_agent.discover_paper_ideas(
                    max_problems=1,
                    max_papers=1,
                    agenda_id=2,
                    proposal_job_id=110,
                    proposal_candidate_id=115,
                    proposal_grant_id=501,
                )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["proposal_candidate_id"], 115)
        self.assertEqual(result[0]["generation_tokens"], 200)
        self.assertEqual(
            [call.kwargs["operation"] for call in exact_call.call_args_list],
            ["proposal_method_invention", "proposal_experiment_design"],
        )

    def test_exact_discovery_refuses_any_scope_drift_before_llm(self):
        with (
            mock.patch.object(
                paper_idea_agent.db,
                "fetchone",
                return_value=_exact_problem_scope(job_status="queued"),
            ),
            mock.patch.object(paper_idea_agent, "_call_exact_proposal_llm") as call,
        ):
            with self.assertRaisesRegex(ValueError, "not executable"):
                paper_idea_agent.discover_paper_ideas(
                    agenda_id=2,
                    proposal_job_id=110,
                    proposal_candidate_id=115,
                    proposal_grant_id=501,
                )
        call.assert_not_called()

    def test_invalid_checkpointed_method_is_terminal_fail_closed(self):
        checkpoint_repo = mock.Mock()
        checkpoint_repo.recover_or_refuse.return_value = _checkpoint_payload(
            output="not-json",
            cost_usd=0.01,
        )
        with (
            mock.patch.object(
                paper_idea_agent.db,
                "fetchone",
                return_value=_exact_problem_scope(),
            ),
            mock.patch.object(
                paper_idea_agent,
                "configured_role_prompt_version",
                return_value="proposal-v1",
            ),
            mock.patch.object(
                proposal_checkpoint,
                "ProposalCheckpointRepository",
                return_value=checkpoint_repo,
            ),
            mock.patch.object(paper_idea_agent, "call_llm_for_role") as provider,
            mock.patch.object(grant_usage, "GrantUsageLedger") as ledger,
        ):
            for _ in range(2):
                with self.assertRaisesRegex(
                    proposal_checkpoint.ProposalCheckpointError,
                    "automatic retry is forbidden",
                ):
                    paper_idea_agent.discover_paper_ideas(
                        agenda_id=2,
                        proposal_job_id=110,
                        proposal_candidate_id=115,
                        proposal_grant_id=501,
                    )
        provider.assert_not_called()
        ledger.assert_not_called()

    def test_exact_problem_loader_is_read_only_and_scope_bound(self):
        row = {
            "id": 9,
            "agenda_id": 2,
            "problem_statement": "bounded problem",
            "source_signal_ref": "{}",
            "node_ids": "[]",
            "paper_ids": "[]",
            "ruled_out_approaches": "[]",
            "problem_quality_score": 1.0,
        }
        with (
            mock.patch.object(problem_first.db, "fetchone", return_value=row) as fetch,
            mock.patch.object(problem_first, "_load_signal_row", return_value={}),
        ):
            result = problem_first.load_problem_first_candidate(
                agenda_id=2, research_problem_id=9
            )
        self.assertEqual(result["research_problem_id"], 9)
        self.assertEqual(fetch.call_args.args[1], (9, 2, problem_first.MAX_ATTEMPTS))

    def test_scope_escape_refuses_before_store_or_settlement(self):
        repository = mock.Mock()
        discover = mock.Mock(
            return_value=[
                {
                    "proposal_candidate_id": 116,
                    "resource_grant_id": 501,
                    "agenda_id": 2,
                }
            ]
        )
        store = mock.Mock()
        with mock.patch.object(bounded_proposal.db, "fetchone", return_value=_scope()):
            with self.assertRaisesRegex(
                bounded_proposal.BoundedProposalError, "escaped"
            ):
                bounded_proposal.execute_bounded_proposal(
                    _request(),
                    actor="ops:controlled-recovery",
                    repository=repository,
                    discover=discover,
                    store=store,
                )
        store.assert_not_called()
        repository.complete_proposal_generation.assert_not_called()


def _checkpoint_scope(**overrides):
    values = {
        "job_id": 110,
        "agenda_id": 2,
        "idea_id": 115,
        "resource_grant_id": 501,
        "operation": "proposal_method_invention",
        "input_digest": "a" * 64,
    }
    values.update(overrides)
    return proposal_checkpoint.ProposalCheckpointScope(**values)


def _checkpoint_payload(**overrides):
    values = {
        "schema": proposal_checkpoint.CHECKPOINT_SCHEMA,
        "job_id": 110,
        "agenda_id": 2,
        "idea_id": 115,
        "resource_grant_id": 501,
        "operation": "proposal_method_invention",
        "input_digest": "a" * 64,
        "idempotency_key": "bounded-key:t1",
        "reservation_id": 77,
        "tokens_used": 42,
        "cost_usd": 0.01,
        "route": {"provider": "p", "model": "m"},
        "output": '{"method":{"name":"M"}}',
    }
    values.update(overrides)
    return values


class ProposalCheckpointTests(unittest.TestCase):
    def test_reserved_checkpoint_is_settled_and_replayed(self):
        repo = proposal_checkpoint.ProposalCheckpointRepository()
        payload = _checkpoint_payload()
        ledger = mock.Mock()
        with (
            mock.patch.object(repo, "load", return_value=payload),
            mock.patch.object(
                proposal_checkpoint.db,
                "fetchone",
                return_value={
                    "id": 77,
                    "status": "reserved",
                    "operation": "proposal_method_invention",
                    "idempotency_key": "bounded-key:t1",
                    "token_reserved": 100,
                    "tokens_used": None,
                    "cost_usd": None,
                },
            ),
            mock.patch.object(
                proposal_checkpoint,
                "GrantUsageLedger",
                return_value=ledger,
            ),
        ):
            recovered = repo.recover_or_refuse(_checkpoint_scope())
        self.assertEqual(recovered, payload)
        ledger.settle.assert_called_once_with(
            77,
            tokens_used=42,
            cost_usd=0.01,
        )

    def test_open_or_settled_usage_without_checkpoint_is_fail_closed(self):
        repo = proposal_checkpoint.ProposalCheckpointRepository()
        with (
            mock.patch.object(repo, "load", return_value=None),
            mock.patch.object(repo, "_refuse_input_drift"),
            mock.patch.object(
                proposal_checkpoint.db,
                "fetchall",
                return_value=[{"id": 77, "status": "reserved"}],
            ),
        ):
            with self.assertRaisesRegex(
                proposal_checkpoint.ProposalCheckpointError,
                "operator reconciliation",
            ):
                repo.recover_or_refuse(_checkpoint_scope())

    def test_delivered_output_fingerprint_drift_is_refused(self):
        repo = proposal_checkpoint.ProposalCheckpointRepository()
        payload = _checkpoint_payload(input_digest="b" * 64)
        with mock.patch.object(
            proposal_checkpoint.db,
            "fetchall",
            return_value=[{"payload": json.dumps(payload)}],
        ):
            with self.assertRaisesRegex(
                proposal_checkpoint.ProposalCheckpointError,
                "fingerprint changed",
            ):
                repo._refuse_input_drift(_checkpoint_scope())

    def test_exact_llm_replay_never_calls_provider(self):
        repo = mock.Mock()
        repo.recover_or_refuse.return_value = _checkpoint_payload()
        with (
            mock.patch.object(
                proposal_checkpoint,
                "ProposalCheckpointRepository",
                return_value=repo,
            ),
            mock.patch.object(paper_idea_agent, "call_llm_for_role") as provider,
            mock.patch.object(grant_usage, "GrantUsageLedger") as ledger,
        ):
            output, tokens, route = paper_idea_agent._call_exact_proposal_llm(
                job_id=110,
                agenda_id=2,
                idea_id=115,
                grant_id=501,
                operation="proposal_method_invention",
                system_prompt="system",
                user_prompt="user",
                prompt_version="v1",
                token_cap=100,
            )
        self.assertIn("method", output)
        self.assertEqual(tokens, 42)
        self.assertEqual(route["model"], "m")
        provider.assert_not_called()
        ledger.assert_not_called()

    def test_exact_llm_persists_delivery_before_returning(self):
        repo = mock.Mock()
        delivered = _checkpoint_payload()
        repo.recover_or_refuse.side_effect = [None, delivered]
        repo.idempotency_base.return_value = "bounded-key"
        ledger = mock.Mock()
        ledger.next_attempt_key.return_value = "bounded-key:t1"

        def provider(_system, _user, **kwargs):
            kwargs["delivery_sink"](
                {
                    "agenda_id": 2,
                    "idea_id": 115,
                    "resource_grant_id": 501,
                    "operation": "proposal_method_invention",
                    "idempotency_key": "bounded-key:t1",
                    "reservation_id": 77,
                    "tokens_used": 42,
                    "cost_usd": 0.01,
                    "route": {"provider": "p", "model": "m"},
                    "output": delivered["output"],
                }
            )
            return delivered["output"], 42, delivered["route"]

        with (
            mock.patch.object(
                proposal_checkpoint,
                "ProposalCheckpointRepository",
                return_value=repo,
            ),
            mock.patch.object(
                grant_usage,
                "GrantUsageLedger",
                return_value=ledger,
            ),
            mock.patch.object(
                paper_idea_agent,
                "call_llm_for_role",
                side_effect=provider,
            ) as provider_call,
        ):
            output, tokens, _ = paper_idea_agent._call_exact_proposal_llm(
                job_id=110,
                agenda_id=2,
                idea_id=115,
                grant_id=501,
                operation="proposal_method_invention",
                system_prompt="system",
                user_prompt="user",
                prompt_version="v1",
                token_cap=100,
            )
        self.assertEqual(output, delivered["output"])
        self.assertEqual(tokens, 42)
        provider_call.assert_called_once()
        repo.save_delivery.assert_called_once()
        self.assertEqual(repo.recover_or_refuse.call_count, 2)


class _RoutingLedger:
    class Reservation:
        reservation_id = 77

    def __init__(self, events, *, fail_settle=False):
        self.events = events
        self.fail_settle = fail_settle

    def reserve(self, **_kwargs):
        self.events.append("reserve")
        return self.Reservation()

    def settle(self, _reservation_id, **_kwargs):
        self.events.append("settle")
        if self.fail_settle:
            raise RuntimeError("injected settlement failure")

    def release(self, _reservation_id, *, reason):
        self.events.append(f"release:{reason}")


class ProposalDeliveryOrderingTests(unittest.TestCase):
    @staticmethod
    def _route(route_id="p"):
        return ProviderRoute(
            route_id=route_id,
            provider=route_id,
            model=f"{route_id}-model",
            model_family=f"{route_id}-family",
            prompt_version="v1",
            timeout_seconds=30,
        )

    @staticmethod
    def _grant():
        return ResourceGrant(
            agenda_id=2,
            idea_id=115,
            decision_packet_id=4,
            stage="proposal",
            token_cap=100,
            backend_allowlist=["llm"],
            artifact_requirements=["proposal"],
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            grant_reason="test",
            idempotency_key="grant-501",
            grant_id=501,
        )

    @staticmethod
    def _request():
        return RouteRequest(
            agenda_id=2,
            idea_id=115,
            role="proposer",
            stage="proposal",
            resource_grant_id=501,
            token_cap=100,
            operation="proposal_method_invention",
            idempotency_key="bounded-key:t1",
        )

    def test_delivery_is_durable_before_usage_settlement(self):
        events = []
        route = self._route()
        router = LLMRouter(
            {"proposer": [route], "evaluator": [route], "reviewer": [route]},
            ledger=_RoutingLedger(events),
            delivery_sink=lambda _delivery: events.append("delivery"),
            observation_sink=lambda _observation: events.append("observation"),
        )
        result = router.invoke(
            self._request(),
            grant=self._grant(),
            executor=lambda _route, _request: (
                events.append("provider") or "output",
                RouteUsage(10, 5, 0.01),
            ),
        )
        self.assertEqual(result.output, "output")
        self.assertEqual(
            events,
            ["reserve", "provider", "delivery", "settle", "observation"],
        )

    def test_checkpoint_failure_never_falls_back_to_second_provider(self):
        events = []
        first = self._route("p1")
        second = self._route("p2")
        router = LLMRouter(
            {
                "proposer": [first, second],
                "evaluator": [first],
                "reviewer": [first],
            },
            ledger=_RoutingLedger(events),
            delivery_sink=lambda _delivery: (_ for _ in ()).throw(
                RuntimeError("injected checkpoint failure")
            ),
            observation_sink=lambda _observation: None,
        )
        provider_routes = []

        def execute(route, _request):
            provider_routes.append(route.route_id)
            return "output", RouteUsage(10, 5, 0.01)

        with self.assertRaisesRegex(LLMRouteError, "checkpoint_failed"):
            router.invoke(self._request(), grant=self._grant(), executor=execute)
        self.assertEqual(provider_routes, ["p1"])
        self.assertEqual(events.count("settle"), 1)

    def test_observation_failure_after_settlement_never_rebuys_output(self):
        events = []
        first = self._route("p1")
        second = self._route("p2")
        router = LLMRouter(
            {
                "proposer": [first, second],
                "evaluator": [first],
                "reviewer": [first],
            },
            ledger=_RoutingLedger(events),
            delivery_sink=lambda _delivery: events.append("delivery"),
            observation_sink=lambda _observation: (_ for _ in ()).throw(
                RuntimeError("injected observation failure")
            ),
        )
        provider_routes = []

        def execute(route, _request):
            provider_routes.append(route.route_id)
            return "output", RouteUsage(10, 5, 0.01)

        with self.assertRaisesRegex(LLMRouteError, "observation_failed"):
            router.invoke(self._request(), grant=self._grant(), executor=execute)
        self.assertEqual(provider_routes, ["p1"])
        self.assertEqual(events.count("settle"), 1)


if __name__ == "__main__":
    unittest.main()


class TheGrantFundsEveryAttemptNotJustTheFirst(unittest.TestCase):
    """A repair with no budget left is a loop that cannot loop.

    Sized at half the grant, a 32k proposal funded the method call and one
    design call and nothing else, so the first repair failed with
    "ResourceGrant token budget is exhausted" and left the grant spent for the
    candidate behind it (grant 387, agenda 16, 2026-08-27).
    """

    def test_the_default_grant_holds_every_call_the_loop_can_make(self):
        calls = 1 + paper_idea_agent.CONTRACT_ATTEMPTS
        self.assertGreaterEqual(
            paper_idea_agent.PROPOSAL_GRANT_TOKEN_CAP,
            paper_idea_agent.PROPOSAL_CALL_TOKEN_CAP * calls,
        )

    def test_a_call_keeps_the_room_its_prompt_needs(self):
        """Each bounded call reserves prompt bytes plus framing plus output
        against its own ceiling, so shrinking the ceiling to a share of the
        grant made the call impossible rather than cheaper."""
        self.assertEqual(
            paper_idea_agent._proposal_call_token_cap(
                paper_idea_agent.PROPOSAL_GRANT_TOKEN_CAP
            ),
            paper_idea_agent.PROPOSAL_CALL_TOKEN_CAP,
        )

    def test_a_grant_smaller_than_one_call_is_not_overspent(self):
        self.assertEqual(paper_idea_agent._proposal_call_token_cap(5_000), 5_000)

    def test_a_missing_cap_never_yields_zero(self):
        self.assertGreaterEqual(paper_idea_agent._proposal_call_token_cap(0), 1)

    def test_the_lane_default_matches_what_the_loop_needs(self):
        import inspect

        from scripts import auto_advance

        source = inspect.getsource(auto_advance.main)
        self.assertIn("paper_idea_agent.PROPOSAL_GRANT_TOKEN_CAP", source)


class AnUnliftableRefusalIsNotWorthThreeAttempts(unittest.TestCase):
    """The design loop rewrites the experiment, not the problem or the method.

    The agenda's reject rule reads the claim, so a phrase in the invented
    method produces the same refusal on every attempt. Agenda 16 spent three
    design calls on that on 2026-08-27 before abandoning anyway.
    """

    class Agenda:
        reject = {"keywords": ["fine-tuning"]}

    def test_a_refused_claim_is_named_before_any_design_call(self):
        blocked = paper_idea_agent._claim_refused_by_agenda(
            "A method that improves the model by fine-tuning its head.", self.Agenda()
        )
        self.assertIn("fine-tuning", blocked)

    def test_a_clean_claim_costs_nothing(self):
        self.assertEqual(
            paper_idea_agent._claim_refused_by_agenda(
                "A prompt-only intervention with frozen weights.", self.Agenda()
            ),
            "",
        )

    def test_no_agenda_means_no_opinion(self):
        self.assertEqual(
            paper_idea_agent._claim_refused_by_agenda("anything at all", None), ""
        )

    def test_the_bounded_path_asks_before_it_spends(self):
        method = json.dumps(
            {
                "method": {
                    "name": "Fine-Tuned Method",
                    "one_line": "Improve it by fine-tuning the head.",
                    "definition": "minimize an exact persisted objective",
                    "why_novel": "This is distinct because it tests the persisted mechanism directly.",
                    "falsification_hook": "Reject when the bounded metric does not improve.",
                }
            }
        )
        with (
            mock.patch.object(paper_idea_agent.db, "fetchone", return_value=_exact_problem_scope()),
            mock.patch.object(
                paper_idea_agent,
                "_call_exact_proposal_llm",
                side_effect=[(method, 120, {})],
            ) as exact_call,
            mock.patch.object(
                paper_idea_agent, "configured_role_prompt_version", return_value="v1"
            ),
            mock.patch.object(
                paper_idea_agent, "_agenda_scope_rule", return_value=self.Agenda()
            ),
            mock.patch.object(
                paper_idea_agent, "_release_abandoned_proposal_grant"
            ) as release,
        ):
            result = paper_idea_agent.discover_paper_ideas(
                max_problems=1,
                max_papers=1,
                agenda_id=2,
                proposal_job_id=110,
                proposal_candidate_id=115,
                proposal_grant_id=501,
            )
        self.assertEqual(result, [])
        # Only the method call was bought; no design attempt was spent.
        self.assertEqual(
            [call.kwargs["operation"] for call in exact_call.call_args_list],
            ["proposal_method_invention"],
        )
        release.assert_called_once_with({"id": 501}, 2)
