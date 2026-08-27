from __future__ import annotations

import unittest
from unittest import mock

from orchestrator import meta_compute_runtime as mcr


class SshPilotEvidenceStateTests(unittest.TestCase):
    """A pilot that measured on ssh_gpu must reach the evidence ladder.

    colab_worker advances its run to sanity_passed; nothing on the legacy/ssh
    path did, and advance_to_full_benchmark selects on that state. So an
    ssh_gpu run could measure a real two-arm result and never be funded for
    the full benchmark, never audited, and never reach a directional verdict.
    """

    def _run(self, row, artifacts=("hash", 5, ())):
        advanced = {}

        class Repo:
            def advance_experiment_state(self, **kwargs):
                advanced.update(kwargs)

        with mock.patch.object(mcr.db, "fetchone", lambda *a, **k: row), \
             mock.patch.object(mcr.db, "rollback", lambda: None), \
             mock.patch.dict(
                 "sys.modules",
                 {"orchestrator.bounded_execution": mock.Mock(
                     raw_artifacts_hash=lambda **k: artifacts)},
             ), \
             mock.patch("meta_harness.repository.MetaHarnessRepository", Repo):
            result = mcr._advance_pilot_evidence_state("legacy-gpu-job:121")
        return result, advanced

    def test_a_measured_pilot_advances_one_rung(self):
        row = {"agenda_id": 18, "resource_grant_id": 362, "run_id": 272,
               "command_ref": "experiment-run:272",
               "scientific_evidence_state": "planned"}
        result, advanced = self._run(row)
        self.assertEqual(result, "advanced_to_sanity_passed")
        self.assertEqual(advanced["target"], "sanity_passed")
        self.assertTrue(advanced["context"].pilot_only)

    def test_incomplete_artifacts_refuse_the_transition(self):
        row = {"agenda_id": 18, "resource_grant_id": 362, "run_id": 272,
               "command_ref": "experiment-run:272",
               "scientific_evidence_state": "planned"}
        result, advanced = self._run(row, artifacts=("hash", 0, ("final_results",)))
        self.assertEqual(result, "runner_artifact_registration_incomplete")
        self.assertEqual(advanced, {})

    def test_a_run_already_past_planned_is_left_alone(self):
        row = {"agenda_id": 18, "resource_grant_id": 362, "run_id": 272,
               "command_ref": "experiment-run:272",
               "scientific_evidence_state": "full_benchmark_complete"}
        result, advanced = self._run(row)
        self.assertEqual(result, "already_advanced")
        self.assertEqual(advanced, {})

    def test_a_job_with_no_run_is_reported_not_raised(self):
        result, advanced = self._run(None)
        self.assertEqual(result, "no_run_for_job")
        self.assertEqual(advanced, {})


if __name__ == "__main__":
    unittest.main()
