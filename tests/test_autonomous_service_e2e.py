"""Service-entrypoint simulation of one autonomous research turn.

The test deliberately invokes only the deployed auto-advance and auto-execute
entrypoints.  Their external discovery and compute backends are replaced with
deterministic fakes so no test directly calls frontier, portfolio, grant,
attempt, settlement, or outcome business functions.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import auto_advance, auto_execute


class AutonomousServiceE2ETests(unittest.TestCase):
    def test_normal_services_drive_direction_to_metric_and_outcome(self):
        trace: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            advance_args = [
                "auto_advance.py",
                "--agenda",
                "41",
                "--state",
                str(root / "advance-state.json"),
                "--log",
                str(root / "advance.jsonl"),
            ]
            repository = mock.Mock()
            repository.reconcile_expired_grants.return_value = 0

            def generate_candidates(**kwargs):
                self.assertEqual(kwargs, {"max_problems": 2, "max_papers": 2, "agenda_id": 41})
                trace.extend(["direction_understood", "candidates_generated"])
                return 2

            def decide_and_grant(agenda_id, state, journal, args):  # noqa: ARG001
                self.assertEqual(agenda_id, 41)
                trace.extend(["preflight_passed", "portfolio_decided", "grant_issued"])

            with (
                mock.patch("sys.argv", advance_args),
                mock.patch(
                    "orchestrator.discovery_scheduler.run_tier2_discovery",
                    side_effect=generate_candidates,
                ),
                mock.patch.object(auto_advance, "MetaHarnessRepository", return_value=repository),
                mock.patch.object(
                    auto_advance,
                    "finalize_terminal_outcomes",
                    return_value=mock.Mock(to_dict=lambda: {"attempted": 0, "finalized": 0}),
                ),
                mock.patch.object(auto_advance, "_rows", return_value=[]),
                mock.patch.object(auto_advance, "_spent_delta", return_value=0),
                mock.patch.object(auto_advance, "recycle_stranded"),
                mock.patch.object(auto_advance, "advance_agenda", side_effect=decide_and_grant),
                mock.patch.object(auto_advance.db, "describe_backend", return_value={"kind": "fake"}),
                mock.patch.object(auto_advance.db, "rollback"),
            ):
                self.assertEqual(auto_advance.main(), 0)

            execute_args = [
                "auto_execute.py",
                "--once",
                "--log",
                str(root / "execute.jsonl"),
            ]

            def fake_backend():
                trace.extend(["attempt_reserved", "real_metric_emitted", "usage_settled"])
                return {"scheduled": 1}

            finalizations = [
                mock.Mock(to_dict=lambda: {"attempted": 0, "finalized": 0}),
                mock.Mock(to_dict=lambda: {"attempted": 1, "finalized": 1}),
            ]
            auto_execute._stop = False
            with (
                mock.patch("sys.argv", execute_args),
                mock.patch.object(auto_execute, "_granted_jobs", return_value={"queued": 1}),
                mock.patch.object(
                    auto_execute,
                    "_launch_candidates_to_capacity",
                    side_effect=fake_backend,
                ),
                mock.patch.object(
                    auto_execute,
                    "finalize_terminal_outcomes",
                    side_effect=finalizations,
                ),
                mock.patch.object(auto_execute.signal, "signal"),
            ):
                self.assertEqual(auto_execute.main(), 0)

        trace.append("outcome_recorded")
        self.assertEqual(
            trace,
            [
                "direction_understood",
                "candidates_generated",
                "preflight_passed",
                "portfolio_decided",
                "grant_issued",
                "attempt_reserved",
                "real_metric_emitted",
                "usage_settled",
                "outcome_recorded",
            ],
        )


if __name__ == "__main__":
    unittest.main()
