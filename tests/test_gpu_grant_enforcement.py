"""Every GPU grant limit is enforced at issue time, not by convention."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from contracts.meta_harness import ResourceGrant
from meta_harness.repository import (
    SYSTEM_MAX_GPU_HOURS_PER_GRANT,
    MetaHarnessPersistenceError,
    MetaHarnessRepository,
)


NOW = datetime.now(timezone.utc)


def _grant(**overrides) -> ResourceGrant:
    values = {
        "agenda_id": 11,
        "idea_id": 41,
        "decision_packet_id": 9,
        "stage": "pilot",
        "token_cap": 5_000,
        "gpu_class": "a10",
        "max_gpu_hours": 2.0,
        "backend_allowlist": ["cpu"],
        "artifact_requirements": ["logs", "metrics"],
        "expires_at": (NOW + timedelta(hours=6)).isoformat(),
        "grant_reason": "portfolio_score_selected",
        "idempotency_key": "grant-41-pilot-1",
    }
    values.update(overrides)
    return ResourceGrant(**values)


def _agenda(**overrides) -> dict:
    values = {
        "id": 11,
        "status": "active",
        "max_concurrency": 4,
        "token_budget": 50_000,
        "token_spent": 0,
        "token_reserved": 0,
        "gpu_hours_budget": 32.0,
        "gpu_hours_spent": 0.0,
        "gpu_hours_reserved": 0.0,
        "backend_allowlist_json": '["cpu", "llm", "ssh_gpu"]',
        "prefer_json": "{}",
    }
    values.update(overrides)
    return values


DECISION = {"agenda_id": 11, "idea_id": 41, "decision": "promote"}


def _capability_reports():
    """This host: cpu verified, ssh_gpu configured but never canaried."""
    from meta_harness.backend_capability import evaluate_backends

    return evaluate_backends(
        enabled=["cpu", "ssh_gpu"],
        verified=["cpu"],
        gpu_mode="ssh",
        ssh_target_ref="env:TARGET",
        ssh_credential_ref="env:KEY",
        local_gpu_present=False,
    )


def _issue(grant, *, agenda=None, active_grants=0, reports=None):
    """Run issue_grant far enough to reach every admission check."""
    rows = [agenda or _agenda(), DECISION, None, {"count": active_grants}]
    with mock.patch(
        "meta_harness.backend_capability.reports_from_config",
        return_value=reports or _capability_reports(),
    ), mock.patch(
        "meta_harness.repository.db._use_pg", return_value=False
    ), mock.patch(
        "meta_harness.repository.db.fetchone", side_effect=rows
    ), mock.patch("meta_harness.repository.db.rollback"), mock.patch(
        "meta_harness.repository.db.insert_returning_id", return_value=5
    ), mock.patch("meta_harness.repository.db.execute"), mock.patch(
        "meta_harness.repository.db.commit"
    ):
        return MetaHarnessRepository().issue_grant(grant)


class GpuGrantAdmissionTests(unittest.TestCase):
    def test_a_bounded_grant_is_issued(self):
        self.assertEqual(_issue(_grant()), 5)

    def test_per_grant_gpu_cap_defaults_to_eight_hours(self):
        # The Agenda declares no gpu_policy: the system ceiling applies rather
        # than an unlimited grant.
        with self.assertRaisesRegex(
            MetaHarnessPersistenceError, "per-grant GPU-hour cap"
        ):
            _issue(_grant(max_gpu_hours=SYSTEM_MAX_GPU_HOURS_PER_GRANT + 0.01))

    def test_agenda_policy_can_tighten_but_not_widen_the_cap(self):
        tight = _agenda(prefer_json='{"gpu_policy":{"max_gpu_hours_per_grant":1}}')
        with self.assertRaisesRegex(
            MetaHarnessPersistenceError, "per-grant GPU-hour cap"
        ):
            _issue(_grant(max_gpu_hours=2.0), agenda=tight)

        wide = _agenda(prefer_json='{"gpu_policy":{"max_gpu_hours_per_grant":64}}')
        with self.assertRaisesRegex(
            MetaHarnessPersistenceError, "per-grant GPU-hour cap"
        ):
            _issue(_grant(max_gpu_hours=9.0), agenda=wide)

    def test_aggregate_gpu_budget_is_a_hard_cap(self):
        nearly_spent = _agenda(gpu_hours_spent=31.0, gpu_hours_reserved=0.5)
        with self.assertRaisesRegex(
            MetaHarnessPersistenceError, "GPU-hour hard cap"
        ):
            _issue(_grant(max_gpu_hours=2.0), agenda=nearly_spent)

    def test_aggregate_token_budget_is_a_hard_cap(self):
        nearly_spent = _agenda(token_spent=48_000, token_reserved=1_000)
        with self.assertRaisesRegex(MetaHarnessPersistenceError, "token hard cap"):
            _issue(_grant(token_cap=5_000), agenda=nearly_spent)

    def test_max_concurrency_is_enforced(self):
        with self.assertRaisesRegex(MetaHarnessPersistenceError, "max_concurrency"):
            _issue(_grant(), active_grants=4)

    def test_backend_allowlist_cannot_exceed_the_agenda(self):
        # Agenda scope is reported before system capability, so an operator
        # sees the specific reason the request was out of bounds.
        restricted = _agenda(backend_allowlist_json='["cpu", "llm"]')
        with self.assertRaisesRegex(
            MetaHarnessPersistenceError, "backend exceeds agenda allowlist"
        ):
            _issue(_grant(backend_allowlist=["ssh_gpu"]), agenda=restricted)

    def test_ttl_must_be_short_and_in_the_future(self):
        with self.assertRaisesRegex(MetaHarnessPersistenceError, "TTL exceeds"):
            _issue(_grant(expires_at=(NOW + timedelta(days=30)).isoformat()))
        with self.assertRaisesRegex(MetaHarnessPersistenceError, "already expired"):
            _issue(_grant(expires_at=(NOW - timedelta(minutes=1)).isoformat()))

    def test_llm_only_grants_get_the_longer_but_still_bounded_ttl(self):
        llm_only = _grant(
            max_gpu_hours=0.0,
            gpu_class="none",
            backend_allowlist=["llm"],
            expires_at=(NOW + timedelta(hours=48)).isoformat(),
        )
        self.assertEqual(_issue(llm_only), 5)

        too_long = _grant(
            max_gpu_hours=0.0,
            gpu_class="none",
            backend_allowlist=["llm"],
            expires_at=(NOW + timedelta(hours=100)).isoformat(),
        )
        with self.assertRaisesRegex(MetaHarnessPersistenceError, "TTL exceeds"):
            _issue(too_long)

    def test_inactive_agenda_and_non_promoted_decision_are_refused(self):
        with self.assertRaisesRegex(MetaHarnessPersistenceError, "agenda is not active"):
            _issue(_grant(), agenda=_agenda(status="paused_budget"))

        rows = [_agenda(), {"agenda_id": 11, "idea_id": 41, "decision": "park"}]
        with mock.patch(
            "meta_harness.repository.db._use_pg", return_value=False
        ), mock.patch(
            "meta_harness.repository.db.fetchone", side_effect=rows
        ), mock.patch("meta_harness.repository.db.rollback"):
            with self.assertRaisesRegex(
                MetaHarnessPersistenceError, "promote/revisit decision"
            ):
                MetaHarnessRepository().issue_grant(_grant())

    def test_artifacts_are_required_by_the_contract(self):
        from contracts.base import ContractValidationError

        with self.assertRaises(ContractValidationError):
            _grant(artifact_requirements=[]).validate()

    def test_a_grant_must_bound_at_least_one_resource(self):
        from contracts.base import ContractValidationError

        with self.assertRaises(ContractValidationError):
            _grant(token_cap=0, max_gpu_hours=0.0).validate()


class BackendCapabilityAdmissionTests(unittest.TestCase):
    """A grant may not authorize a backend that could never be scheduled."""

    def test_unverified_gpu_backend_is_refused_at_issue_time(self):
        # Found in production: the capability model was only consulted when a
        # job was submitted, so this grant was issued and reserved GPU hours
        # plus a concurrency slot for work that can never run.
        with self.assertRaisesRegex(
            MetaHarnessPersistenceError, "cannot be scheduled"
        ):
            _issue(_grant(backend_allowlist=["ssh_gpu"]))

    def test_a_verified_backend_is_accepted(self):
        from meta_harness.backend_capability import evaluate_backends

        verified = evaluate_backends(
            enabled=["cpu", "ssh_gpu"],
            verified=["cpu", "ssh_gpu"],
            gpu_mode="ssh",
            ssh_target_ref="env:TARGET",
            ssh_credential_ref="env:KEY",
            local_gpu_present=False,
        )
        self.assertEqual(
            _issue(_grant(backend_allowlist=["ssh_gpu"]), reports=verified), 5
        )

    def test_llm_only_grants_do_not_need_a_compute_backend(self):
        llm_only = _grant(
            backend_allowlist=["llm"],
            max_gpu_hours=0.0,
            gpu_class="none",
        )
        self.assertEqual(_issue(llm_only), 5)


class GrantRevocationTests(unittest.TestCase):
    """Withdrawing an unused grant refunds it; a used one cannot be erased."""

    def test_revoking_an_unused_grant_refunds_the_reservation(self):
        rows = [
            {"count": 0},
            {
                "id": 2,
                "agenda_id": 11,
                "reservation_id": 7,
                "token_reserved": 1000,
                "gpu_hours_reserved": 4.0,
                "reservation_status": "reserved",
            },
        ]
        with mock.patch(
            "meta_harness.repository.db.fetchone", side_effect=rows
        ), mock.patch("meta_harness.repository.db.execute") as execute, mock.patch(
            "meta_harness.repository.db.commit"
        ), mock.patch("meta_harness.repository.db.rollback"):
            revoked = MetaHarnessRepository().revoke_grant(
                2, agenda_id=11, reason="backend cannot be scheduled"
            )

        self.assertTrue(revoked)
        statements = " ".join(str(call.args[0]) for call in execute.call_args_list)
        self.assertIn("token_reserved=token_reserved-?", statements)
        self.assertIn("gpu_hours_reserved=gpu_hours_reserved-?", statements)
        self.assertIn("status='released'", statements)
        self.assertIn("status='revoked'", statements)
        # Withdrawal is not completion.
        self.assertIn("resource_grant_revoked", statements)
        self.assertNotIn("outcome_records", statements)

    def test_a_grant_that_metered_usage_cannot_be_revoked(self):
        with mock.patch(
            "meta_harness.repository.db.fetchone", return_value={"count": 1}
        ), mock.patch("meta_harness.repository.db.rollback"):
            with self.assertRaisesRegex(
                MetaHarnessPersistenceError, "already metered usage"
            ):
                MetaHarnessRepository().revoke_grant(2, agenda_id=11, reason="oops")

    def test_a_reason_is_required(self):
        with mock.patch("meta_harness.repository.db.rollback"):
            with self.assertRaises(MetaHarnessPersistenceError):
                MetaHarnessRepository().revoke_grant(2, agenda_id=11, reason="  ")


if __name__ == "__main__":
    unittest.main()
