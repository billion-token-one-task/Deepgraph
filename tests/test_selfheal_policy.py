"""The watchdog must not restart a deliberately paused system."""

import unittest
from unittest import mock
from unittest.mock import patch

from orchestrator.selfheal_policy import (
    ACTION_HOLD,
    REASON_MAINTENANCE,
    REASON_STARTUP_GRACE,
    ACTION_RESTART,
    HEALTH_FAILED,
    HEALTH_OK,
    HEALTH_UNKNOWN,
    REASON_AUTONOMY_DISABLED,
    REASON_AWAITING_AUTHORITY,
    REASON_COOLDOWN,
    REASON_HEALTH_FLAPPING,
    REASON_HEALTH_OK,
    REASON_NO_WORK_EXPECTED,
    REASON_OUTPUT_AGE_UNKNOWN,
    REASON_OUTPUT_FRESH,
    REASON_PROCESS_NOT_RUNNING,
    REASON_PROVIDER_ISSUE,
    REASON_RESTART_DB_TRANSACTION_STALLED,
    REASON_RESTART_HEALTH,
    REASON_RESTART_OUTPUT_STALLED,
    SelfHealPolicy,
    SelfHealPolicyError,
    SelfHealSignals,
    decide,
    next_consecutive_failures,
)


def _working_signals(**overrides) -> SelfHealSignals:
    """A system that really is supposed to be producing output."""
    base = {
        "web_process_running": True,
        "health_status": HEALTH_OK,
        "auto_research_enabled": True,
        "auto_pipeline_enabled": True,
        "active_resource_grants": 1,
        "running_jobs": 1,
        "awaiting_authority": False,
        "output_age_seconds": 10,
        "provider_issue": False,
        "seconds_since_last_restart": None,
    }
    base.update(overrides)
    return SelfHealSignals(**base)


class PausedSystemTests(unittest.TestCase):
    def test_disabled_autonomy_never_restarts_on_stale_output(self):
        signals = _working_signals(
            auto_research_enabled=False,
            auto_pipeline_enabled=False,
            output_age_seconds=86_400,
        )

        decision = decide(signals)

        self.assertEqual(decision.action, ACTION_HOLD)
        self.assertEqual(decision.reason_code, REASON_AUTONOMY_DISABLED)

    def test_seventy_minutes_of_ticks_on_a_paused_system_never_restart(self):
        # Ten-minute timer, so 70 minutes is 7 consecutive decisions.
        for tick in range(7):
            signals = _working_signals(
                auto_research_enabled=False,
                auto_pipeline_enabled=False,
                output_age_seconds=45 * 60 + tick * 600,
                seconds_since_last_restart=None,
            )
            decision = decide(signals)
            self.assertEqual(decision.action, ACTION_HOLD, f"tick {tick}")
            self.assertEqual(decision.reason_code, REASON_AUTONOMY_DISABLED)

    def test_awaiting_authority_holds_even_with_autonomy_enabled(self):
        signals = _working_signals(
            awaiting_authority=True,
            awaiting_authority_reasons=("portfolio_or_grant_decision_pending",),
            output_age_seconds=86_400,
        )

        decision = decide(signals)

        self.assertEqual(decision.action, ACTION_HOLD)
        self.assertEqual(decision.reason_code, REASON_AWAITING_AUTHORITY)
        self.assertEqual(
            decision.details["awaiting"], ["portfolio_or_grant_decision_pending"]
        )

    def test_no_admitted_work_holds(self):
        signals = _working_signals(
            active_resource_grants=0,
            running_jobs=0,
            output_age_seconds=86_400,
        )

        decision = decide(signals)

        self.assertEqual(decision.reason_code, REASON_NO_WORK_EXPECTED)

    def test_provider_credit_issue_is_not_restartable(self):
        signals = _working_signals(provider_issue=True, output_age_seconds=86_400)

        decision = decide(signals)

        self.assertEqual(decision.action, ACTION_HOLD)
        self.assertEqual(decision.reason_code, REASON_PROVIDER_ISSUE)


class RealFailureTests(unittest.TestCase):
    def test_stalled_admitted_work_restarts(self):
        signals = _working_signals(output_age_seconds=46 * 60)

        decision = decide(signals)

        self.assertEqual(decision.action, ACTION_RESTART)
        self.assertEqual(decision.reason_code, REASON_RESTART_OUTPUT_STALLED)

    def test_repeated_health_failure_restarts_even_when_paused(self):
        signals = _working_signals(
            health_status=HEALTH_FAILED,
            health_consecutive_failures=3,
            auto_research_enabled=False,
            auto_pipeline_enabled=False,
        )

        decision = decide(signals)

        self.assertEqual(decision.action, ACTION_RESTART)
        self.assertEqual(decision.reason_code, REASON_RESTART_HEALTH)

    def test_single_health_failure_does_not_restart(self):
        signals = _working_signals(
            health_status=HEALTH_FAILED,
            health_consecutive_failures=1,
        )

        decision = decide(signals)

        self.assertEqual(decision.action, ACTION_HOLD)
        self.assertEqual(decision.reason_code, REASON_HEALTH_FLAPPING)

    def test_dead_process_is_left_to_systemd(self):
        signals = _working_signals(web_process_running=False, output_age_seconds=86_400)

        decision = decide(signals)

        self.assertEqual(decision.action, ACTION_HOLD)
        self.assertEqual(decision.reason_code, REASON_PROCESS_NOT_RUNNING)


class FailSafeTests(unittest.TestCase):
    def test_unknown_output_age_never_restarts(self):
        signals = _working_signals(output_age_seconds=None)

        decision = decide(signals)

        self.assertEqual(decision.reason_code, REASON_OUTPUT_AGE_UNKNOWN)

    def test_unknown_health_falls_through_to_output_rules(self):
        signals = _working_signals(health_status=HEALTH_UNKNOWN, output_age_seconds=5)

        decision = decide(signals)

        self.assertEqual(decision.reason_code, REASON_OUTPUT_FRESH)

    def test_fresh_output_holds(self):
        self.assertEqual(decide(_working_signals()).reason_code, REASON_OUTPUT_FRESH)

    def test_cooldown_suppresses_every_restart_path(self):
        for signals in (
            _working_signals(output_age_seconds=86_400, seconds_since_last_restart=60),
            _working_signals(
                health_status=HEALTH_FAILED,
                health_consecutive_failures=9,
                seconds_since_last_restart=60,
            ),
        ):
            decision = decide(signals)
            self.assertEqual(decision.action, ACTION_HOLD)
            self.assertEqual(decision.reason_code, REASON_COOLDOWN)
            self.assertIn("suppressed_reason", decision.details)

    def test_restart_allowed_after_cooldown_expires(self):
        signals = _working_signals(
            output_age_seconds=86_400,
            seconds_since_last_restart=31 * 60,
        )

        self.assertEqual(decide(signals).action, ACTION_RESTART)

    def test_invalid_signals_are_rejected(self):
        with self.assertRaises(SelfHealPolicyError):
            decide(_working_signals(health_status="weird"))
        with self.assertRaises(SelfHealPolicyError):
            decide(_working_signals(output_age_seconds=-5))
        with self.assertRaises(SelfHealPolicyError):
            decide(_working_signals(), policy=SelfHealPolicy(stall_seconds=0))

    def test_consecutive_failure_counter(self):
        self.assertEqual(next_consecutive_failures(0, health_status=HEALTH_FAILED), 1)
        self.assertEqual(next_consecutive_failures(2, health_status=HEALTH_FAILED), 3)
        self.assertEqual(next_consecutive_failures(4, health_status=HEALTH_OK), 0)
        # An unobservable probe must neither escalate nor forgive.
        self.assertEqual(next_consecutive_failures(2, health_status=HEALTH_UNKNOWN), 2)

    def test_health_ok_reason_code_is_reachable_only_through_output_rules(self):
        # REASON_HEALTH_OK stays exported for the runbook even though the
        # healthy path is expressed through the output rules.
        self.assertTrue(REASON_HEALTH_OK.startswith("hold_"))


class RunnerSignalParsingTests(unittest.TestCase):
    """The runner's pure helpers; no systemd, database, or network."""

    def setUp(self):
        from scripts import deepgraph_selfheal

        self.runner = deepgraph_selfheal

    def test_parse_counts_reads_one_aggregate_row(self):
        parsed = self.runner.parse_counts(" 1234|2|1|3 \n")

        self.assertEqual(
            parsed,
            {
                "output_age_seconds": 1234,
                "active_grants": 2,
                "running_jobs": 1,
                "awaiting_jobs": 3,
            },
        )

    def test_parse_counts_rejects_unusable_output(self):
        self.assertIsNone(self.runner.parse_counts(""))
        self.assertIsNone(self.runner.parse_counts("ERROR:  relation does not exist"))
        self.assertIsNone(self.runner.parse_counts("1|2|3"))

    def test_parse_counts_maps_empty_database_to_unknown_age(self):
        self.assertEqual(
            self.runner.parse_counts("-1|0|0|0")["output_age_seconds"], -1
        )

    def test_env_flag_defaults_when_absent(self):
        values = {"DEEPGRAPH_AUTO_RESEARCH_ENABLED": "false"}

        self.assertFalse(
            self.runner.env_flag(values, "DEEPGRAPH_AUTO_RESEARCH_ENABLED", True)
        )
        self.assertFalse(
            self.runner.env_flag(values, "DEEPGRAPH_AUTO_PIPELINE_ENABLED", False)
        )
        self.assertTrue(self.runner.env_flag({"X": "1"}, "X", False))

    def test_provider_issue_markers(self):
        self.assertTrue(self.runner.provider_issue_in_log("... Cooling down 900s ..."))
        self.assertTrue(self.runner.provider_issue_in_log("auth failed (401)"))
        self.assertFalse(self.runner.provider_issue_in_log("normal operation"))

    def test_health_probe_maps_transport_errors_to_unknown(self):
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("temporary name resolution failure"),
        ):
            self.assertEqual(
                self.runner.probe_health("http://127.0.0.1:8080/api/meta"),
                HEALTH_UNKNOWN,
            )

    def test_health_probe_maps_connection_refused_to_failed(self):
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("[Errno 111] Connection refused"),
        ):
            self.assertEqual(
                self.runner.probe_health("http://127.0.0.1:8080/api/meta"),
                HEALTH_FAILED,
            )

    def test_health_probe_maps_server_error_to_failed(self):
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                "http://127.0.0.1:8080/api/meta", 503, "unavailable", {}, None
            ),
        ):
            self.assertEqual(
                self.runner.probe_health("http://127.0.0.1:8080/api/meta"),
                HEALTH_FAILED,
            )

    def test_unreadable_database_signal_becomes_unknown_not_zero(self):
        with patch.object(self.runner, "_run", side_effect=OSError("no psql")):
            self.assertIsNone(
                self.runner.read_counts(
                    "postgresql://user:pw@127.0.0.1:5433/deepgraph", psql="psql"
                )
            )
        self.assertIsNone(self.runner.read_counts("", psql="psql"))

    def test_collect_signals_marks_unreadable_database_as_unknown(self):
        with patch.object(self.runner, "parse_env_file", return_value={}), patch.object(
            self.runner, "process_running", return_value=True
        ), patch.object(
            self.runner, "probe_health", return_value=HEALTH_OK
        ), patch.object(
            self.runner, "read_counts", return_value=None
        ), patch.object(
            self.runner, "tail_text", return_value=""
        ):
            signals = self.runner.collect_signals(
                runtime_env="/nonexistent/.env",
                process_pattern="Deepgraph/main.py",
                health_url="http://127.0.0.1:8080/api/meta",
                web_log="/nonexistent/web.log",
                psql="psql",
                previous_state={},
                now=1_000_000.0,
            )

        self.assertIsNone(signals.output_age_seconds)
        self.assertEqual(signals.active_resource_grants, 0)
        self.assertFalse(signals.awaiting_authority)
        self.assertEqual(decide(signals).action, ACTION_HOLD)

    def test_library_root_resolution_prefers_an_explicit_deployment_path(self):
        import os
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory(prefix="selfheal-lib-") as directory:
            module = os.path.join(directory, "orchestrator", "selfheal_policy.py")
            os.makedirs(os.path.dirname(module))
            open(module, "w").close()

            with patch.dict(os.environ, {"DEEPGRAPH_SELFHEAL_LIB": directory}):
                self.assertEqual(self.runner._resolve_library_root(), directory)

            # An env pointing at a directory without the module falls through to
            # the next candidate instead of poisoning sys.path.
            with patch.dict(os.environ, {"DEEPGRAPH_SELFHEAL_LIB": "/nonexistent"}):
                self.assertNotEqual(self.runner._resolve_library_root(), "/nonexistent")


class StartupAndMaintenanceTests(unittest.TestCase):
    """The two rules that would have prevented tonight's restart loop."""

    def test_a_service_that_is_still_starting_is_never_restarted(self):
        # The external one-minute healthcheck killed this host's web service
        # mid-startup, once a minute, because startup takes longer than a
        # minute. A probe faster than startup is a guaranteed restart loop
        # unless the policy knows what "still starting" looks like.
        signals = _working_signals(
            health_status=HEALTH_FAILED,
            health_consecutive_failures=9,
            service_uptime_seconds=30,
        )

        decision = decide(signals)

        self.assertEqual(decision.action, ACTION_HOLD)
        self.assertEqual(decision.reason_code, REASON_STARTUP_GRACE)

    def test_the_same_failure_restarts_once_startup_has_finished(self):
        signals = _working_signals(
            health_status=HEALTH_FAILED,
            health_consecutive_failures=3,
            service_uptime_seconds=600,
        )

        self.assertEqual(decide(signals).action, ACTION_RESTART)

    def test_unknown_uptime_does_not_block_a_real_restart(self):
        signals = _working_signals(
            health_status=HEALTH_FAILED,
            health_consecutive_failures=3,
            service_uptime_seconds=None,
        )

        self.assertEqual(decide(signals).action, ACTION_RESTART)

    def test_maintenance_mode_outranks_every_signal(self):
        signals = _working_signals(
            maintenance_mode=True,
            web_process_running=False,
            health_status=HEALTH_FAILED,
            health_consecutive_failures=99,
            output_age_seconds=86_400,
        )

        decision = decide(signals)

        self.assertEqual(decision.action, ACTION_HOLD)
        self.assertEqual(decision.reason_code, REASON_MAINTENANCE)

    def test_negative_uptime_is_rejected(self):
        with self.assertRaises(SelfHealPolicyError):
            decide(_working_signals(service_uptime_seconds=-1))
        with self.assertRaises(SelfHealPolicyError):
            decide(_working_signals(), policy=SelfHealPolicy(startup_grace_seconds=-1))

    def test_a_one_minute_tick_on_a_paused_system_skips_the_expensive_query(self):
        from scripts import deepgraph_selfheal

        with mock.patch.object(
            deepgraph_selfheal, "parse_env_file",
            return_value={"DEEPGRAPH_AUTO_RESEARCH_ENABLED": "false",
                          "DEEPGRAPH_AUTO_PIPELINE_ENABLED": "0"},
        ), mock.patch.object(
            deepgraph_selfheal, "process_running", return_value=True
        ), mock.patch.object(
            deepgraph_selfheal, "probe_health", return_value=HEALTH_OK
        ), mock.patch.object(
            deepgraph_selfheal, "read_counts"
        ) as read_counts, mock.patch.object(
            deepgraph_selfheal, "tail_text", return_value=""
        ):
            signals = deepgraph_selfheal.collect_signals(
                runtime_env="/nonexistent/.env",
                process_pattern="x",
                health_url="http://127.0.0.1:8080/api/meta",
                web_log="/nonexistent/web.log",
                psql="psql",
                previous_state={},
                now=1_000_000.0,
            )

        read_counts.assert_not_called()
        self.assertIsNone(signals.output_age_seconds)
        self.assertEqual(decide(signals).reason_code, REASON_AUTONOMY_DISABLED)


class StuckTransactionSignalTests(unittest.TestCase):
    """The half-dead state: HTTP 200, no output, and locks held forever."""

    def test_abandoned_transaction_past_the_window_restarts(self):
        signals = _working_signals(
            output_age_seconds=10,
            max_idle_transaction_seconds=20 * 60,
            idle_transaction_sessions=11,
        )
        decision = decide(signals, policy=SelfHealPolicy(idle_transaction_seconds=15 * 60))

        self.assertEqual(decision.action, ACTION_RESTART)
        self.assertEqual(decision.reason_code, REASON_RESTART_DB_TRANSACTION_STALLED)
        self.assertEqual(decision.details["idle_transaction_sessions"], 11)

    def test_stuck_transaction_is_caught_even_with_autonomy_off(self):
        """Output freshness cannot see this, so the autonomy hold must not mask it."""
        signals = _working_signals(
            auto_research_enabled=False,
            auto_pipeline_enabled=False,
            output_age_seconds=None,
            max_idle_transaction_seconds=20 * 60,
            idle_transaction_sessions=4,
        )
        decision = decide(signals, policy=SelfHealPolicy(idle_transaction_seconds=15 * 60))

        self.assertEqual(decision.reason_code, REASON_RESTART_DB_TRANSACTION_STALLED)

    def test_short_lived_transaction_is_left_alone(self):
        """Normal write traffic is briefly idle in transaction all the time."""
        signals = _working_signals(
            output_age_seconds=10,
            max_idle_transaction_seconds=30,
            idle_transaction_sessions=3,
        )
        decision = decide(signals, policy=SelfHealPolicy(idle_transaction_seconds=15 * 60))

        self.assertEqual(decision.action, ACTION_HOLD)
        self.assertEqual(decision.reason_code, REASON_OUTPUT_FRESH)

    def test_unavailable_transaction_signal_never_restarts_and_never_masks(self):
        """Fail safe: unknown falls through to the existing output-stall path."""
        self.assertEqual(
            decide(_working_signals(output_age_seconds=10)).reason_code,
            REASON_OUTPUT_FRESH,
        )
        stalled = decide(
            _working_signals(output_age_seconds=99 * 60, max_idle_transaction_seconds=None)
        )
        self.assertEqual(stalled.reason_code, REASON_RESTART_OUTPUT_STALLED)

    def test_stuck_transaction_restart_still_obeys_the_cooldown(self):
        decision = decide(
            _working_signals(
                output_age_seconds=10,
                max_idle_transaction_seconds=20 * 60,
                seconds_since_last_restart=60,
            )
        )

        self.assertEqual(decision.action, ACTION_HOLD)
        self.assertEqual(decision.reason_code, REASON_COOLDOWN)
        self.assertEqual(
            decision.details["suppressed_reason"],
            REASON_RESTART_DB_TRANSACTION_STALLED,
        )

    def test_maintenance_and_startup_grace_still_outrank_the_new_signal(self):
        stuck = {"max_idle_transaction_seconds": 20 * 60, "output_age_seconds": 10}
        self.assertEqual(
            decide(_working_signals(maintenance_mode=True, **stuck)).reason_code,
            REASON_MAINTENANCE,
        )
        self.assertEqual(
            decide(_working_signals(service_uptime_seconds=5, **stuck)).reason_code,
            REASON_STARTUP_GRACE,
        )

    def test_negative_transaction_counters_are_rejected(self):
        with self.assertRaises(SelfHealPolicyError):
            decide(_working_signals(max_idle_transaction_seconds=-1))
        with self.assertRaises(SelfHealPolicyError):
            decide(_working_signals(idle_transaction_sessions=-1))
        with self.assertRaises(SelfHealPolicyError):
            decide(_working_signals(), policy=SelfHealPolicy(idle_transaction_seconds=0))


class StuckTransactionCollectionTests(unittest.TestCase):
    def setUp(self):
        from scripts import deepgraph_selfheal

        self.runner = deepgraph_selfheal

    def test_parse_transaction_health_reads_one_row(self):
        parsed = self.runner.parse_transaction_health(" 660|11 \n")

        self.assertEqual(parsed["max_idle_transaction_seconds"], 660)
        self.assertEqual(parsed["idle_transaction_sessions"], 11)

    def test_parse_transaction_health_rejects_unusable_output(self):
        self.assertIsNone(self.runner.parse_transaction_health(""))
        self.assertIsNone(self.runner.parse_transaction_health("ERROR:  denied"))
        self.assertIsNone(self.runner.parse_transaction_health("1|2|3"))

    def test_empty_database_maps_to_unknown_not_zero(self):
        self.assertEqual(
            self.runner.parse_transaction_health("-1|0")["max_idle_transaction_seconds"],
            -1,
        )
        self.assertIsNone(self.runner.read_transaction_health("", psql="psql"))
        with patch.object(self.runner, "_run", side_effect=OSError("no psql")):
            self.assertIsNone(
                self.runner.read_transaction_health(
                    "postgresql://user:pw@127.0.0.1:5433/deepgraph", psql="psql"
                )
            )

    def test_transaction_query_runs_even_when_autonomy_is_off(self):
        """The expensive freshness scan is skipped; this cheap one is not."""
        with mock.patch.object(
            self.runner,
            "parse_env_file",
            return_value={
                "DEEPGRAPH_AUTO_RESEARCH_ENABLED": "false",
                "DEEPGRAPH_AUTO_PIPELINE_ENABLED": "0",
                "DEEPGRAPH_DATABASE_URL": "postgresql://user:pw@127.0.0.1:5433/deepgraph",
            },
        ), mock.patch.object(
            self.runner, "process_running", return_value=True
        ), mock.patch.object(
            self.runner, "probe_health", return_value=HEALTH_OK
        ), mock.patch.object(
            self.runner, "read_counts"
        ) as read_counts, mock.patch.object(
            self.runner,
            "read_transaction_health",
            return_value={
                "max_idle_transaction_seconds": 1200,
                "idle_transaction_sessions": 11,
            },
        ) as read_txn, mock.patch.object(
            self.runner, "tail_text", return_value=""
        ):
            signals = self.runner.collect_signals(
                runtime_env="/nonexistent/.env",
                process_pattern="x",
                health_url="http://127.0.0.1:8080/api/meta",
                web_log="/nonexistent/web.log",
                psql="psql",
                previous_state={},
                now=1_000_000.0,
            )

        read_counts.assert_not_called()
        read_txn.assert_called_once()
        self.assertEqual(signals.max_idle_transaction_seconds, 1200)
        self.assertEqual(signals.idle_transaction_sessions, 11)
        self.assertEqual(
            decide(signals).reason_code, REASON_RESTART_DB_TRANSACTION_STALLED
        )

    def test_transaction_query_selects_no_row_content(self):
        """The watchdog must never read query text or user data out of the DB."""
        sql = self.runner.IDLE_TRANSACTION_SQL.lower()

        self.assertIn("pg_stat_activity", sql)
        self.assertNotIn("query", sql)
        self.assertIn("pid <> pg_backend_pid()", sql)


if __name__ == "__main__":
    unittest.main()
