"""Fault-injection contracts for isolated CI; no real providers/backends."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from contracts.meta_harness import ResourceGrant
from meta_harness.compute import (
    ArtifactCollection,
    BackendCapability,
    CPUBackend,
    ComputeBackendError,
    ComputeClaim,
    ComputeJob,
    ComputeScheduler,
    ComputeSubmission,
    LocalGPUBackend,
    UsageAccounting,
)
from meta_harness.llm_routing import (
    LLMExecutionFailure,
    LLMRouteError,
    LLMRouteUnavailableError,
    LLMRouter,
    ProviderRoute,
    RouteRequest,
    RouteUsage,
)
from agents import llm_client


def _grant(*, backends: list[str]) -> ResourceGrant:
    return ResourceGrant(
        agenda_id=2,
        idea_id=3,
        decision_packet_id=4,
        stage="pilot",
        token_cap=1000,
        max_gpu_hours=0.0,
        backend_allowlist=backends,
        artifact_requirements=["raw_metrics"],
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        grant_reason="test",
        idempotency_key="grant-2-3",
        grant_id=5,
    )


class _Ledger:
    class Reservation:
        reservation_id = 17

    def __init__(self):
        self.settled = []
        self.released = []

    def reserve(self, **_kwargs):
        return self.Reservation()

    def settle(self, reservation_id, **kwargs):
        self.settled.append((reservation_id, kwargs))

    def release(self, reservation_id, *, reason):
        self.released.append((reservation_id, reason))


class _CooldownStore:
    def __init__(self):
        self.active = {}
        self.saved = []

    def load_active_cooldowns(self, route_ids, *, now):
        return {
            route_id: until
            for route_id, until in self.active.items()
            if route_id in route_ids and until > now
        }

    def save_cooldown(self, route, *, until, failure_category):
        self.active[route.route_id] = until
        self.saved.append((route.route_id, failure_category, until))


class _Transport:
    def __init__(self, status="submitted", *, kind="cpu", fail_submit=False):
        self.submission_status = status
        self.kind = kind
        self.fail_submit = fail_submit
        self.capability_calls = 0
        self.submit_calls = 0

    def capability(self):
        self.capability_calls += 1
        return BackendCapability(self.kind, True, cpu_cores=2)

    def submit(self, request):
        self.submit_calls += 1
        if self.fail_submit:
            raise RuntimeError("injected submission transport failure")
        return ComputeJob(
            self.kind, "job-1", request.idempotency_key, self.submission_status,
            failure_reason="injected" if self.submission_status == "failed" else None,
        )

    def status(self, backend_job_id):
        return ComputeJob(
            self.kind,
            backend_job_id,
            "key",
            self.submission_status,
            failure_reason=(
                "injected"
                if self.submission_status == "failed"
                else None
            ),
        )

    def heartbeat(self, backend_job_id):
        return self.status(backend_job_id)

    def cancel(self, backend_job_id):
        return ComputeJob(self.kind, backend_job_id, "key", "cancelled")

    def collect_artifacts(self, backend_job_id, requirements):
        return ArtifactCollection({"job": backend_job_id}, True)

    def usage(self, backend_job_id):
        return UsageAccounting(1.0, 0.0, 0.001)


class _ComputeStore:
    def __init__(self, existing: ComputeClaim | None = None):
        self.existing = existing
        self.claims = []
        self.bound = []
        self.unknown = []
        self.states = []
        self.terminals = []
        self.successes = []

    def claim(self, request, *, backend_kind):
        self.claims.append((request, backend_kind))
        if self.existing is not None:
            return self.existing
        return ComputeClaim(
            record_id=41,
            is_new=True,
            backend_kind=backend_kind,
            idempotency_key=request.idempotency_key,
            status="submitting",
        )

    def bind_submitted_job(self, record_id, job):
        self.bound.append((record_id, job))

    def mark_submission_unknown(self, record_id, *, reason):
        self.unknown.append((record_id, reason))

    def record_id_for_job(self, _job):
        return 41

    def record_backend_state(self, job):
        self.states.append(job)
        return "collecting" if job.status == "succeeded" else job.status

    def finalize_terminal(self, job, *, usage):
        self.terminals.append((job, usage))

    def finalize_success(self, record_id, *, artifacts, usage):
        self.successes.append((record_id, artifacts, usage))


class LLMRoutingTests(unittest.TestCase):
    def _route(self, route_id, provider, family):
        return ProviderRoute(
            route_id=route_id,
            provider=provider,
            model=f"{family}-model",
            model_family=family,
            prompt_version="v1",
            timeout_seconds=30,
        )

    def test_evaluator_cannot_reuse_only_proposer_family(self):
        route = self._route("a", "provider-a", "family-a")
        router = LLMRouter(
            {"proposer": [route], "evaluator": [route], "reviewer": [route]},
            ledger=_Ledger(),
            observation_sink=lambda _observation: None,
        )
        request = RouteRequest(
            agenda_id=2,
            idea_id=3,
            role="evaluator",
            stage="pilot",
            resource_grant_id=5,
            token_cap=100,
            operation="evaluate",
            idempotency_key="eval-1",
            proposer_route=route,
        )
        with self.assertRaises(LLMRouteUnavailableError):
            router.invoke(
                request,
                grant=_grant(backends=["llm"]),
                executor=lambda _route, _request: ("unsafe", RouteUsage(1, 1, None)),
            )

    def test_failed_routes_are_observed_and_fail_closed(self):
        proposer = self._route("p", "provider-a", "family-a")
        evaluator = self._route("e", "provider-b", "family-b")
        ledger = _Ledger()
        observations = []
        router = LLMRouter(
            {
                "proposer": [proposer],
                "evaluator": [evaluator],
                "reviewer": [evaluator],
            },
            ledger=ledger,
            observation_sink=observations.append,
        )
        request = RouteRequest(
            agenda_id=2,
            idea_id=3,
            role="evaluator",
            stage="pilot",
            resource_grant_id=5,
            token_cap=100,
            operation="evaluate",
            idempotency_key="eval-2",
            proposer_route=proposer,
        )

        def fail(_route, _request):
            raise LLMExecutionFailure(
                "injected outage",
                category="transient",
                usage=RouteUsage(5, 0, 0.01),
            )

        with self.assertRaises(LLMRouteUnavailableError):
            router.invoke(
                request,
                grant=_grant(backends=["llm"]),
                executor=fail,
            )
        self.assertEqual(observations[0].status, "failed")
        self.assertEqual(ledger.settled[0][1]["tokens_used"], 5)
        self.assertFalse(ledger.released)

    def test_successful_provider_overrun_is_settled_and_fails_closed(self):
        proposer = self._route("p", "provider-a", "family-a")
        ledger = _Ledger()
        observations = []
        router = LLMRouter(
            {
                "proposer": [proposer],
                "evaluator": [proposer],
                "reviewer": [proposer],
            },
            ledger=ledger,
            observation_sink=observations.append,
        )
        request = RouteRequest(
            agenda_id=2,
            idea_id=3,
            role="proposer",
            stage="pilot",
            resource_grant_id=5,
            token_cap=100,
            operation="tagged_repair",
            idempotency_key="tagged-overrun-1",
            max_attempts=1,
        )

        with self.assertRaisesRegex(
            LLMRouteError,
            "provider_usage_exceeded_reserved_cap",
        ):
            router.invoke(
                request,
                grant=_grant(backends=["llm"]),
                executor=lambda _route, _request: (
                    "unexpected",
                    RouteUsage(60, 41, 0.01),
                ),
            )
        self.assertEqual(ledger.settled[0][1]["tokens_used"], 100)
        self.assertNotIn("allow_overrun", ledger.settled[0][1])
        self.assertFalse(ledger.released)
        self.assertEqual(observations[0].status, "failed")
        self.assertEqual(
            observations[0].failure_reason,
            "provider_usage_exceeded_reserved_cap",
        )

    def test_attempt_cap_prevents_transient_retry_or_route_fallback(self):
        first = ProviderRoute(
            route_id="p1",
            provider="provider-a",
            model="family-a-model",
            model_family="family-a",
            prompt_version="v1",
            timeout_seconds=30,
            transient_retries=2,
        )
        second = self._route("p2", "provider-b", "family-b")
        ledger = _Ledger()
        attempts = []
        router = LLMRouter(
            {
                "proposer": [first, second],
                "evaluator": [first],
                "reviewer": [first],
            },
            ledger=ledger,
            observation_sink=lambda _observation: None,
        )
        request = RouteRequest(
            agenda_id=2,
            idea_id=3,
            role="proposer",
            stage="pilot",
            resource_grant_id=5,
            token_cap=100,
            operation="tagged_repair",
            idempotency_key="tagged-attempt-cap-1",
            max_attempts=1,
        )

        def fail_once(route, _request):
            attempts.append(route.route_id)
            raise LLMExecutionFailure(
                "injected outage",
                category="transient",
                usage=RouteUsage(7, 0, None),
            )

        with self.assertRaisesRegex(
            LLMRouteUnavailableError,
            "route_attempt_cap_exhausted",
        ):
            router.invoke(
                request,
                grant=_grant(backends=["llm"]),
                executor=fail_once,
            )
        self.assertEqual(attempts, ["p1"])
        self.assertEqual(ledger.settled[0][1]["tokens_used"], 7)
        self.assertFalse(ledger.released)

    def test_failed_provider_overrun_is_also_fully_settled(self):
        proposer = self._route("p", "provider-a", "family-a")
        ledger = _Ledger()
        observations = []
        router = LLMRouter(
            {
                "proposer": [proposer],
                "evaluator": [proposer],
                "reviewer": [proposer],
            },
            ledger=ledger,
            observation_sink=observations.append,
        )
        request = RouteRequest(
            agenda_id=2,
            idea_id=3,
            role="proposer",
            stage="pilot",
            resource_grant_id=5,
            token_cap=100,
            operation="tagged_repair",
            idempotency_key="tagged-failed-overrun-1",
            max_attempts=1,
        )

        def fail_over_cap(_route, _request):
            raise LLMExecutionFailure(
                "provider reported failure after usage",
                category="provider_error",
                usage=RouteUsage(70, 31, 0.02),
            )

        with self.assertRaisesRegex(
            LLMRouteError,
            "failed_attempt_usage_exceeded_reserved_cap",
        ):
            router.invoke(
                request,
                grant=_grant(backends=["llm"]),
                executor=fail_over_cap,
            )
        self.assertEqual(ledger.settled[0][1]["tokens_used"], 100)
        self.assertNotIn("allow_overrun", ledger.settled[0][1])
        self.assertFalse(ledger.released)
        self.assertEqual(
            observations[0].input_tokens + observations[0].output_tokens,
            101,
        )

    def test_provider_cooldown_survives_router_reconstruction(self):
        proposer = self._route("p", "provider-a", "family-a")
        cooldowns = _CooldownStore()
        first = LLMRouter(
            {
                "proposer": [proposer],
                "evaluator": [proposer],
                "reviewer": [proposer],
            },
            ledger=_Ledger(),
            observation_sink=lambda _observation: None,
            cooldown_store=cooldowns,
        )
        request = RouteRequest(
            agenda_id=2,
            idea_id=3,
            role="proposer",
            stage="pilot",
            resource_grant_id=5,
            token_cap=100,
            operation="propose",
            idempotency_key="propose-cooldown-1",
        )
        with self.assertRaises(LLMRouteUnavailableError):
            first.invoke(
                request,
                grant=_grant(backends=["llm"]),
                executor=lambda _route, _request: (_ for _ in ()).throw(
                    LLMExecutionFailure("injected auth", category="auth")
                ),
            )
        self.assertEqual(cooldowns.saved[0][0:2], ("p", "auth"))

        restarted = LLMRouter(
            {
                "proposer": [proposer],
                "evaluator": [proposer],
                "reviewer": [proposer],
            },
            ledger=_Ledger(),
            observation_sink=lambda _observation: None,
            cooldown_store=cooldowns,
        )
        self.assertEqual(restarted.eligible_routes(request), [])

    def test_role_route_references_resolve_without_secret_material(self):
        routes = {
            "proposer": [
                {
                    "provider_ref": "env:TEST_PROPOSER_PROVIDER",
                    "model_ref": "env:TEST_PROPOSER_MODEL",
                    "model_family_ref": "env:TEST_PROPOSER_FAMILY",
                    "prompt_version": "proposer-v1",
                }
            ],
            "evaluator": [],
            "reviewer": [],
        }
        with (
            mock.patch.object(llm_client, "LLM_ROLE_ROUTES", routes),
            mock.patch.dict(
                "os.environ",
                {
                    "TEST_PROPOSER_PROVIDER": "provider-a",
                    "TEST_PROPOSER_MODEL": "model-a",
                    "TEST_PROPOSER_FAMILY": "family-a",
                },
                clear=False,
            ),
        ):
            policy = llm_client.configured_role_route_policy("proposer")
        self.assertEqual(set(policy), {"provider-a"})
        self.assertEqual(policy["provider-a"]["model"], "model-a")
        self.assertEqual(policy["provider-a"]["model_family"], "family-a")

    def test_unresolved_role_route_fails_closed(self):
        routes = {
            "proposer": [
                {
                    "provider_ref": "env:TEST_MISSING_PROVIDER",
                    "prompt_version": "proposer-v1",
                }
            ],
            "evaluator": [],
            "reviewer": [],
        }
        with (
            mock.patch.object(llm_client, "LLM_ROLE_ROUTES", routes),
            mock.patch.dict("os.environ", {}, clear=True),
            self.assertRaises(llm_client.LLMProviderUnavailableError),
        ):
            llm_client.configured_role_route_policy("proposer")


class ComputeRoutingTests(unittest.TestCase):
    def _submission(self):
        return ComputeSubmission(
            agenda_id=2,
            idea_id=3,
            stage="pilot",
            resource_grant_id=5,
            idempotency_key="job-key",
            command_ref="artifact:commands/pilot.json",
            artifact_namespace="agenda-2/idea-3/job-key",
            timeout_seconds=60,
        )

    def test_backend_failure_does_not_create_completed_job(self):
        scheduler = ComputeScheduler(
            [CPUBackend(_Transport(status="failed"))],
            allow_ephemeral_idempotency=True,
        )
        with self.assertRaises(ComputeBackendError):
            scheduler.submit(
                self._submission(),
                grant=_grant(backends=["cpu"]),
                preferred_backends=["cpu"],
            )

    def test_artifacts_only_certify_successful_job(self):
        scheduler = ComputeScheduler([CPUBackend(_Transport())])
        failed = ComputeJob("cpu", "job-1", "job-key", "failed", failure_reason="x")
        with self.assertRaises(ComputeBackendError):
            scheduler.collect_if_successful(failed, requirements=("raw_metrics",))

    def test_submission_fails_closed_without_durable_store(self):
        transport = _Transport()
        scheduler = ComputeScheduler([CPUBackend(transport)])
        with self.assertRaisesRegex(
            ComputeBackendError,
            "durable_compute_job_store_required",
        ):
            scheduler.submit(
                self._submission(),
                grant=_grant(backends=["cpu"]),
                preferred_backends=["cpu"],
            )
        self.assertEqual(transport.capability_calls, 0)
        self.assertEqual(transport.submit_calls, 0)

    def test_restart_reuses_durable_live_job_without_resubmission(self):
        transport = _Transport()
        store = _ComputeStore(
            ComputeClaim(
                record_id=41,
                is_new=False,
                backend_kind="cpu",
                idempotency_key="job-key",
                status="submitted",
                backend_job_id="persisted-job-1",
            )
        )
        scheduler = ComputeScheduler(
            [CPUBackend(transport)],
            job_store=store,
        )
        job = scheduler.submit(
            self._submission(),
            grant=_grant(backends=["cpu"]),
            preferred_backends=["cpu"],
        )
        self.assertEqual(job.backend_job_id, "persisted-job-1")
        self.assertEqual(transport.submit_calls, 0)
        self.assertFalse(store.bound)

    def test_unknown_submission_does_not_fallback_or_resubmit(self):
        cpu = _Transport(fail_submit=True)
        gpu = _Transport(kind="local_gpu")
        store = _ComputeStore()
        scheduler = ComputeScheduler(
            [CPUBackend(cpu), LocalGPUBackend(gpu)],
            job_store=store,
        )
        with self.assertRaisesRegex(
            ComputeBackendError,
            "manual_reconciliation_required",
        ):
            scheduler.submit(
                self._submission(),
                grant=_grant(backends=["cpu", "local_gpu"]),
                preferred_backends=["cpu", "local_gpu"],
            )
        self.assertEqual(cpu.submit_calls, 1)
        self.assertEqual(gpu.capability_calls, 0)
        self.assertEqual(gpu.submit_calls, 0)
        self.assertEqual(store.unknown, [(41, "cpu:RuntimeError")])
        self.assertFalse(store.bound)

    def test_collecting_claim_requires_reconciliation(self):
        claim = ComputeClaim(
            record_id=41,
            is_new=False,
            backend_kind="cpu",
            idempotency_key="job-key",
            status="collecting",
            backend_job_id="persisted-job-1",
        )
        with self.assertRaisesRegex(
            ComputeBackendError,
            "submission_reconciliation_required",
        ):
            claim.existing_job()

    def test_failed_backend_is_metered_before_terminal_persistence(self):
        transport = _Transport(status="failed")
        store = _ComputeStore()
        scheduler = ComputeScheduler(
            [CPUBackend(transport)],
            job_store=store,
        )
        observed = scheduler.refresh_and_settle(
            ComputeJob("cpu", "backend-job", "job-key", "running"),
            requirements=("raw_metrics",),
        )
        self.assertEqual(observed.status, "failed")
        self.assertFalse(store.states)
        self.assertEqual(len(store.terminals), 1)
        self.assertEqual(store.terminals[0][1].wall_seconds, 1.0)


if __name__ == "__main__":
    unittest.main()
