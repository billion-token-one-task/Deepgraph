from __future__ import annotations

import unittest
from unittest import mock

from meta_harness import outcome_finalizer


class SettledEvidenceStateTests(unittest.TestCase):
    """One rule for every backend, because a GPU is a GPU.

    The transition existed four times -- colab_worker twice, meta_compute_runtime
    for ssh, bounded_execution gated on resource_class=="cpu" whose else-branch
    literally recorded "non_cpu_run". Every run that ever produced supported or
    refuted went through Colab, and each new rented accelerator would have
    needed a fifth copy.
    """

    def _advance(self, rows, artifacts=("digest", 5, ()), contract="chash"):
        calls = []

        class Repo:
            def advance_experiment_state(self, **kwargs):
                calls.append(kwargs)

        def fetchall(sql, params=()):
            return rows

        def fetchone(sql, params=()):
            # The upper rung reads the grant first, then the locked contract.
            if "resource_grants" in sql:
                return {"stage": getattr(self, "_stage", "pilot"),
                        "preflight_result_id": 313}
            return {"requirements_hash": contract} if contract else None

        with mock.patch.object(outcome_finalizer.db, "fetchall", fetchall), \
             mock.patch.object(outcome_finalizer.db, "fetchone", fetchone), \
             mock.patch.object(outcome_finalizer.db, "rollback", lambda: None), \
             mock.patch.dict(
                 "sys.modules",
                 {"orchestrator.bounded_execution": mock.Mock(
                     raw_artifacts_hash=lambda **k: artifacts)},
             ), \
             mock.patch("meta_harness.repository.MetaHarnessRepository", Repo):
            counts = outcome_finalizer._advance_settled_evidence_state()
        return counts, calls

    def _row(self, state="planned", stage="pilot", verdict="refuted"):
        self._stage = stage
        return {"run_id": 272, "agenda_id": 18, "state": state,
                "grant_id": 362, "verdict": verdict}

    def test_a_pilot_advances_one_rung_whatever_ran_it(self):
        counts, calls = self._advance([self._row()])
        self.assertEqual(counts["sanity_passed"], 1)
        self.assertEqual(calls[0]["target"], "sanity_passed")
        self.assertTrue(calls[0]["context"].pilot_only)

    def test_a_colab_shaped_run_is_not_excluded(self):
        """The first draft keyed on compute_jobs_v1.command_ref.

        Colab keys its jobs colab-work-request:N, ssh keys them
        experiment-run:N, and a cpu run creates none -- that draft would have
        cut Colab out of its own ladder while its own transition was being
        deleted. The selection must not mention compute jobs at all.
        """
        source = outcome_finalizer._advance_settled_evidence_state.__doc__ or ""
        import inspect
        body = inspect.getsource(outcome_finalizer._advance_settled_evidence_state)
        self.assertNotIn("compute_jobs_v1", body)
        self.assertIn("hypothesis_verdict", body)

    def test_the_upper_rung_needs_a_full_benchmark_grant_and_a_contract(self):
        counts, calls = self._advance([self._row(state="sanity_passed",
                                                 stage="full_benchmark")])
        self.assertEqual(counts["full_benchmark_complete"], 1)
        self.assertTrue(calls[0]["context"].full_benchmark_complete)

    def test_a_pilot_grant_cannot_buy_the_upper_rung(self):
        counts, calls = self._advance([self._row(state="sanity_passed",
                                                 stage="pilot")])
        self.assertEqual(counts["full_benchmark_complete"], 0)
        self.assertEqual(calls, [])

    def test_missing_artifacts_refuse_the_transition(self):
        counts, calls = self._advance([self._row()],
                                      artifacts=("digest", 0, ("final_results",)))
        self.assertEqual(counts["refused"], 1)
        self.assertEqual(calls, [])

    def test_a_missing_contract_hash_refuses_the_upper_rung(self):
        counts, calls = self._advance([self._row(state="sanity_passed",
                                                 stage="full_benchmark")],
                                      contract="")
        self.assertEqual(counts["refused"], 1)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()


class StateOrderingTests(unittest.TestCase):
    """The forward-only guard, exercised so a bad import cannot pass CI.

    EVIDENCE_STATES lives in contracts.meta_harness; importing it from
    meta_harness.repository raises only when the branch actually runs, which
    no test reached.
    """

    def test_a_record_behind_the_run_moves_forward(self):
        self.assertTrue(
            outcome_finalizer._state_is_behind("planned", "sanity_passed")
        )
        self.assertTrue(
            outcome_finalizer._state_is_behind(
                "sanity_passed", "full_benchmark_complete"
            )
        )

    def test_a_record_never_moves_backward(self):
        self.assertFalse(
            outcome_finalizer._state_is_behind("sanity_passed", "planned")
        )

    def test_a_retraction_is_a_decision_not_a_stale_view(self):
        self.assertFalse(
            outcome_finalizer._state_is_behind(
                "unmeasurable_retracted", "sanity_passed"
            )
        )


class ValidationGrantStageTests(unittest.TestCase):
    """A full_benchmark grant is compute authority, not a different currency.

    run_validation_loop accepted only pilot and validation, so idea 237's
    benchmark run -- funded by the full_benchmark grant issued precisely to
    run it -- was refused as having no authority at all.
    """

    def test_the_loop_accepts_the_grant_that_funds_a_benchmark(self):
        import inspect
        from agents import validation_loop

        body = inspect.getsource(validation_loop.run_validation_loop)
        self.assertIn("'full_benchmark'", body)
        self.assertIn("'pilot'", body)
        self.assertIn("'validation'", body)


class HoldoutAdmissionLeaseTests(unittest.TestCase):
    """An admission that never produced a compute job is not a flight.

    Requests 155 and 156 sat at 'admitting' with compute_job_id NULL while both
    audits reported holdout_pending on every pass, so two candidates waited on
    a provision that had already failed silently. Nothing bounded that state.
    """

    def test_the_stale_check_can_see_whether_a_job_exists(self):
        import inspect
        from meta_harness import evidence_audit

        body = inspect.getsource(evidence_audit)
        # the guard reads compute_job_id, so the query must fetch it
        self.assertIn("SELECT id, status, failure_reason, compute_job_id", body)
        self.assertIn("transport:admission_abandoned", body)

    def test_the_abandoned_reason_is_transport_class(self):
        from meta_harness.failure_policy import measured_nothing

        self.assertTrue(measured_nothing("transport:admission_abandoned"))


class HoldoutBackendIndependenceTests(unittest.TestCase):
    """A holdout is a different flight on purpose.

    The preflight records where the candidate ran. For a pilot, requiring the
    submission to match it is a real check. For an evidence_audit holdout --
    which reruns the locked contract on unseen data, on whatever accelerator
    the audit chooses -- equating them refused two candidates measured on
    ssh_gpu with passed_candidate_preflight_required.
    """

    def test_the_audit_stage_validates_against_the_grants_own_allowlist(self):
        import inspect
        from orchestrator import meta_compute_runtime as mcr

        body = inspect.getsource(mcr._require_grant_preflight)
        self.assertIn('"evidence_audit"', body)
        self.assertIn("backend_allowlist_json", body)

    def test_the_grant_row_actually_carries_what_the_branch_reads(self):
        """SELECT * is what makes the branch reachable; keep it that way."""
        import inspect
        from orchestrator import meta_compute_runtime as mcr

        for fn in (mcr.submit_colab_work, mcr.submit_experiment_run):
            with self.subTest(fn=fn.__name__):
                self.assertIn("SELECT * FROM resource_grants", inspect.getsource(fn))
