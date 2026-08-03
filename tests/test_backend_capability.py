"""Unverified backends are never usable, and nothing falls back silently."""

from __future__ import annotations

import unittest
from unittest import mock

from meta_harness.backend_capability import (
    STATE_DISABLED,
    STATE_ENABLED,
    STATE_UNKNOWN,
    BackendCapabilityError,
    evaluate_backends,
    require_schedulable,
    selected_canary_backend,
)


def _production_shaped(**overrides):
    """The configuration actually observed on the host at session start."""
    values = {
        "enabled": ["cpu", "ssh_gpu"],
        "verified": ["cpu"],
        "gpu_mode": "ssh",
        "ssh_target_ref": "env:DEEPGRAPH_SSH_TARGET",
        "ssh_credential_ref": "env:DEEPGRAPH_SSH_KEY",
        "colab_manifest_ref": "",
        "colab_binary": "",
        "local_gpu_present": False,
        "legacy_gpu_backend": "colab",
    }
    values.update(overrides)
    return evaluate_backends(**values)


class ObservedConfigurationTests(unittest.TestCase):
    def test_cpu_is_the_only_schedulable_backend(self):
        reports = _production_shaped()

        self.assertEqual(reports["cpu"].state, STATE_ENABLED)
        self.assertEqual(reports["ssh_gpu"].state, STATE_UNKNOWN)
        self.assertEqual(reports["colab_gpu"].state, STATE_DISABLED)
        self.assertEqual(reports["local_gpu"].state, STATE_DISABLED)

    def test_ssh_gpu_is_unknown_until_a_canary_verifies_it(self):
        reports = _production_shaped()

        self.assertIn(
            "configured_but_never_verified_by_a_canary", reports["ssh_gpu"].reasons
        )
        self.assertIn(
            "ssh_reachability_and_gpu_presence_canary",
            reports["ssh_gpu"].verification_required,
        )
        self.assertFalse(reports["ssh_gpu"].usable_for_scheduling)
        self.assertTrue(reports["ssh_gpu"].usable_for_canary)

    def test_colab_without_its_manifest_is_disabled(self):
        reports = _production_shaped()

        self.assertIn("colab_accounts_manifest_absent", reports["colab_gpu"].reasons)
        self.assertFalse(reports["colab_gpu"].usable_for_canary)

    def test_legacy_gpu_backend_field_never_enables_anything(self):
        reports = _production_shaped()

        self.assertIn(
            "legacy_gpu_backend_field_conflicts_with_enabled_list",
            reports["colab_gpu"].reasons,
        )
        self.assertEqual(reports["colab_gpu"].state, STATE_DISABLED)

    def test_no_local_gpu_on_this_host(self):
        reports = _production_shaped(enabled=["cpu", "local_gpu"])

        self.assertEqual(reports["local_gpu"].state, STATE_DISABLED)
        self.assertIn("no_local_gpu_detected_on_host", reports["local_gpu"].reasons)

    def test_only_reference_names_are_reported_never_values(self):
        reports = _production_shaped()

        for report in reports.values():
            for ref in report.secret_refs:
                self.assertNotIn("env:", ref)
                self.assertTrue(ref.startswith("compute_backends."))


class FailClosedTests(unittest.TestCase):
    def test_unknown_backend_kind_is_rejected(self):
        with self.assertRaises(BackendCapabilityError):
            evaluate_backends(enabled=["cpu", "quantum"])
        with self.assertRaises(BackendCapabilityError):
            require_schedulable("quantum", _production_shaped())

    def test_scheduling_refuses_unknown_and_disabled_backends(self):
        reports = _production_shaped()

        require_schedulable("cpu", reports)
        for kind in ("ssh_gpu", "colab_gpu", "local_gpu"):
            with self.subTest(kind=kind), self.assertRaises(BackendCapabilityError):
                require_schedulable(kind, reports)

    def test_verification_promotes_exactly_the_verified_backend(self):
        reports = _production_shaped(verified=["cpu", "ssh_gpu"])

        self.assertEqual(reports["ssh_gpu"].state, STATE_ENABLED)
        require_schedulable("ssh_gpu", reports)
        self.assertEqual(reports["colab_gpu"].state, STATE_DISABLED)

    def test_verifying_a_backend_that_is_not_configured_does_not_enable_it(self):
        reports = _production_shaped(
            verified=["cpu", "colab_gpu"], enabled=["cpu", "colab_gpu"]
        )

        self.assertEqual(reports["colab_gpu"].state, STATE_DISABLED)


class CanarySelectionTests(unittest.TestCase):
    def test_exactly_one_gpu_backend_may_be_selected(self):
        reports = _production_shaped()

        self.assertEqual(selected_canary_backend(reports, requested="ssh_gpu").kind, "ssh_gpu")
        with self.assertRaisesRegex(BackendCapabilityError, "is not the eligible one"):
            selected_canary_backend(reports, requested="colab_gpu")

    def test_ambiguous_configuration_is_refused_not_resolved(self):
        reports = _production_shaped(
            enabled=["cpu", "ssh_gpu", "colab_gpu"],
            colab_manifest_ref="secretmanager:colab-accounts",
            colab_binary="/usr/local/bin/colab",
        )

        with self.assertRaisesRegex(BackendCapabilityError, "enable exactly one"):
            selected_canary_backend(reports, requested="ssh_gpu")

    def test_no_eligible_gpu_backend_is_an_explicit_error(self):
        reports = _production_shaped(enabled=["cpu"])

        with self.assertRaisesRegex(BackendCapabilityError, "no GPU backend"):
            selected_canary_backend(reports, requested="ssh_gpu")

    def test_cpu_is_never_a_canary_target(self):
        with self.assertRaises(BackendCapabilityError):
            selected_canary_backend(_production_shaped(), requested="cpu")


class RuntimeRoutingTests(unittest.TestCase):
    def test_submit_refuses_an_unverified_gpu_backend(self):
        from orchestrator import meta_compute_runtime

        grant_row = {
            "id": 3,
            "agenda_id": 5,
            "idea_id": 41,
            "decision_packet_id": 9,
            "stage": "pilot",
            "token_cap": 1000,
            "gpu_class": "a10",
            "max_gpu_hours": 1.0,
            "backend_allowlist_json": '["ssh_gpu"]',
            "artifact_requirements_json": '["logs"]',
            "expires_at": "2030-01-01T00:00:00+00:00",
            "grant_reason": "promote",
            "idempotency_key": "k",
            "status": "active",
            "reservation_id": 11,
        }
        with mock.patch.object(
            meta_compute_runtime.db, "fetchone", return_value=grant_row
        ), mock.patch.object(
            meta_compute_runtime, "_backend_kind", return_value="ssh_gpu"
        ), mock.patch.object(
            meta_compute_runtime, "_enabled_backend_kinds", return_value={"cpu", "ssh_gpu"}
        ), mock.patch.object(
            meta_compute_runtime, "reports_from_config", return_value=_production_shaped()
        ):
            with self.assertRaisesRegex(
                meta_compute_runtime.ComputeBackendError, "not schedulable"
            ):
                meta_compute_runtime.submit_experiment_run(
                    agenda_id=5,
                    idea_id=41,
                    experiment_run_id=17,
                    resource_grant_id=3,
                    timeout_seconds=60,
                    backend_kind="ssh_gpu",
                )

    def test_scheduler_only_builds_verified_backends(self):
        from orchestrator import meta_compute_runtime

        with mock.patch.object(
            meta_compute_runtime, "_enabled_backend_kinds", return_value={"cpu", "ssh_gpu"}
        ), mock.patch.object(
            meta_compute_runtime, "reports_from_config", return_value=_production_shaped()
        ), mock.patch.object(
            meta_compute_runtime, "ComputeJobRepository", return_value=object()
        ), mock.patch.object(
            meta_compute_runtime, "ComputeScheduler"
        ) as scheduler:
            meta_compute_runtime.build_scheduler()

        built = scheduler.call_args.args[0]
        self.assertEqual([type(backend).__name__ for backend in built], ["CPUBackend"])


if __name__ == "__main__":
    unittest.main()
