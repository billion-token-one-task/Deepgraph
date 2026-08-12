import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import auto_advance
from meta_harness.attempt_gpu_usage import GrantGPUUsage
from meta_harness.frontier_bootstrap import FrontierBootstrapError


class AutoAdvanceGuardTests(unittest.TestCase):
    def test_gpu_failure_is_recyclable(self):
        self.assertIn(("failed", "gpu_failed"), auto_advance.DEAD_END)

    def test_recycle_reuses_live_grant_without_reserving_another_cap(self):
        job = {
            "id": 99,
            "deep_insight_id": 105,
            "status": "failed",
            "stage": "gpu_failed",
            "resource_grant_id": 17,
            "last_error": "reproduction failure",
            "token_cap": 40000,
            "grant_status": "active",
            "grant_live": True,
        }
        args = mock.Mock(
            agenda=[10, 11],
            grant_token_cap=40000,
            spend_limit=120000,
        )
        journal = mock.Mock()
        state = {"recycles": {"105": 1}}

        with (
            mock.patch.object(auto_advance, "_rows", return_value=[job]),
            mock.patch.object(
                auto_advance,
                "_spent_delta",
                side_effect=AssertionError("must not reserve a second grant"),
            ),
            mock.patch.object(auto_advance, "_requeue_for_consumer") as requeue,
        ):
            auto_advance.recycle_stranded(11, state, journal, args)

        requeue.assert_called_once_with(11, 105, 17, journal, args, 2, token_cap=40000)
        self.assertEqual(state["recycles"]["105"], 2)

    def test_expired_gpu_job_regrant_uses_structured_preflight(self):
        job = {
            "id": 99,
            "deep_insight_id": 105,
            "status": "failed",
            "stage": "gpu_failed",
            "resource_grant_id": 17,
            "last_error": "reproduction failure",
            "token_cap": 40000,
            "grant_status": "expired",
            "grant_live": False,
        }
        args = mock.Mock(
            agenda=[10, 11],
            grant_token_cap=40000,
            grant_gpu_hours=2.0,
            gpu_class="a100",
            spend_limit=120000,
        )
        journal = mock.Mock()
        state = {"recycles": {}}
        issued = mock.Mock(id=18)

        def fetchone(sql, params=()):
            if "FROM idea_decision_packets" in sql:
                return {"id": 7}
            if "backend_allowlist_json" in sql:
                return {"backend_allowlist_json": '["cpu","llm","ssh_gpu"]'}
            raise AssertionError(sql)

        with (
            mock.patch.object(auto_advance, "_rows", return_value=[job]),
            mock.patch.object(auto_advance, "_spent_delta", return_value=0),
            mock.patch.object(auto_advance.db, "fetchone", side_effect=fetchone),
            mock.patch.object(auto_advance, "_rebuild_decision", return_value=mock.Mock()),
            mock.patch.object(
                auto_advance.CandidatePreflightRepository,
                "run_candidate",
                return_value=mock.Mock(
                    passed=True,
                    selected_backend="ssh_gpu",
                    preflight_result_id=9,
                ),
            ),
            mock.patch.object(auto_advance, "_grant_key", return_value="recovery-key"),
            mock.patch.object(auto_advance, "issue_resource_grant", return_value=issued) as issue,
            mock.patch.object(auto_advance.MetaHarnessRepository, "issue_grant", return_value=18),
            mock.patch.object(auto_advance, "_requeue_for_consumer") as requeue,
        ):
            auto_advance.recycle_stranded(11, state, journal, args)

        self.assertEqual(issue.call_args.kwargs["token_cap"], 40000)
        self.assertEqual(issue.call_args.kwargs["preflight_result_id"], 9)
        self.assertEqual(
            issue.call_args.kwargs["backend_allowlist"],
            ["ssh_gpu", "llm"],
        )
        requeue.assert_called_once_with(
            11, 105, 18, journal, args, 1, token_cap=40000
        )
        self.assertEqual(state["recycles"]["105"], 1)

    def test_preflight_deferral_does_not_consume_recycle(self):
        job = {
            "id": 99,
            "deep_insight_id": 105,
            "status": "failed",
            "stage": "gpu_failed",
            "resource_grant_id": 17,
            "last_error": "reproduction failure",
            "token_cap": 40000,
            "grant_status": "expired",
            "grant_live": False,
        }
        args = mock.Mock(
            agenda=[11],
            grant_token_cap=40000,
            grant_gpu_hours=2.0,
            gpu_class="a100",
            spend_limit=40000,
            process_spend_baseline={"11": 205452},
        )
        journal = mock.Mock()
        state = {"recycles": {}}

        def fetchone(sql, params=()):
            if "FROM idea_decision_packets" in sql:
                return {"id": 7}
            if "backend_allowlist_json" in sql:
                return {"backend_allowlist_json": '["llm","ssh_gpu"]'}
            raise AssertionError(sql)

        with (
            mock.patch.object(auto_advance, "_rows", return_value=[job]),
            mock.patch.object(auto_advance, "_guard_spent_delta", return_value=0),
            mock.patch.object(auto_advance.db, "fetchone", side_effect=fetchone),
            mock.patch.object(auto_advance, "_rebuild_decision", return_value=mock.Mock()),
            mock.patch.object(
                auto_advance.CandidatePreflightRepository,
                "run_candidate",
                return_value=mock.Mock(
                    passed=False,
                    selected_backend=None,
                    status="deferred",
                    reason_codes=["model_task_mismatch"],
                ),
            ),
            mock.patch.object(auto_advance, "_requeue_for_consumer") as requeue,
        ):
            auto_advance.recycle_stranded(11, state, journal, args)

        requeue.assert_not_called()
        self.assertEqual(state["recycles"], {})

    def test_live_exhausted_gpu_grant_is_not_recycled(self):
        job = {
            "id": 99,
            "deep_insight_id": 105,
            "status": "failed",
            "stage": "gpu_failed",
            "resource_grant_id": 18,
            "last_error": "reproduction failure",
            "token_cap": 40000,
            "max_gpu_hours": 2.0,
            "grant_status": "active",
            "grant_live": True,
        }
        args = mock.Mock(
            agenda=[10, 11],
            grant_token_cap=40000,
            spend_limit=120000,
        )
        journal = mock.Mock()
        state = {"recycles": {}}

        with (
            mock.patch.object(auto_advance, "_rows", return_value=[job]),
            mock.patch.object(
                auto_advance,
                "_grant_gpu_usage",
                return_value=GrantGPUUsage(
                    resource_grant_id=18,
                    cap_gpu_seconds=7200.0,
                    settled_gpu_seconds=7201.0,
                    active_reserved_gpu_seconds=0.0,
                    active_reservations=0,
                    grant_status="active",
                ),
            ),
            mock.patch.object(auto_advance, "_requeue_for_consumer") as requeue,
        ):
            auto_advance.recycle_stranded(11, state, journal, args)

        requeue.assert_not_called()
        journal.log.assert_called_once()
        self.assertEqual(journal.log.call_args.args[0], "gpu_budget_exhausted")
        self.assertEqual(state["recycles"], {})

    def test_deployed_recycle_epoch_resets_old_operational_retry_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            path.write_text(
                json.dumps(
                    {
                        "spend_baseline": {"11": 65879},
                        "frontier_packets": {"11": 3},
                        "recycles": {"105": 3},
                        "recycle_epoch": "old-code",
                    }
                ),
                encoding="utf-8",
            )

            state = auto_advance._load_state(path)

        self.assertEqual(state["recycles"], {})
        self.assertEqual(state["recycle_epoch"], auto_advance.RECYCLE_EPOCH)
        self.assertEqual(state["spend_baseline"], {"11": 65879})

    def test_spend_guard_counts_expired_metered_usage_and_live_grant_cap(self):
        state = {"spend_baseline": {"11": 65879}}

        def fetchone(sql, params=()):
            if "FROM research_agendas" in sql:
                return {"s": 31817}
            if "resource_grant_usage_reservations" in sql:
                self.assertIn("NOT (g.status='active'", sql)
                return {"s": 113047}
            if "FROM resource_grants" in sql:
                return {"s": 40000}
            raise AssertionError(sql)

        with mock.patch.object(auto_advance.db, "fetchone", side_effect=fetchone):
            spent = auto_advance._spent_delta(state, [11])

        self.assertEqual(spent, 118985)

    def test_explicit_spend_limit_uses_process_not_stale_durable_baseline(self):
        state = {"spend_baseline": {"11": 65879}}
        args = mock.Mock(
            agenda=[11],
            process_spend_baseline={"11": 205452},
        )

        with mock.patch.object(
            auto_advance, "_agenda_committed_spend", return_value=225452
        ):
            spent = auto_advance._guard_spent_delta(state, args)

        self.assertEqual(spent, 20000)
        self.assertEqual(state["spend_baseline"], {"11": 65879})

    def test_waiting_candidates_are_decided_as_one_portfolio(self):
        waiting = [
            {"id": 1, "deep_insight_id": 101, "insight_status": "candidate"},
            {"id": 2, "deep_insight_id": 102, "insight_status": "candidate"},
        ]
        packets = [mock.Mock(idea_id=101), mock.Mock(idea_id=102)]
        decisions = [
            mock.Mock(idea_id=101, decision="park", reason_codes=["correlated"], decision_packet_id=11),
            mock.Mock(idea_id=102, decision="park", reason_codes=["lower_value"], decision_packet_id=12),
        ]
        args = mock.Mock(
            max_new_grants=2,
            proposal_token_cap=32000,
            grant_token_cap=40000,
            spend_limit=0,
            agenda=[9],
        )
        journal = mock.Mock()
        repository = mock.Mock()

        with (
            mock.patch.object(auto_advance, "_rows", side_effect=[[], waiting]),
            mock.patch.object(auto_advance, "select_next", return_value=None),
            mock.patch.object(auto_advance, "ensure_frontier_packet", return_value=7),
            mock.patch.object(auto_advance, "build_packet", side_effect=packets),
            mock.patch.object(auto_advance, "decide_portfolio", return_value=decisions) as decide,
            mock.patch.object(auto_advance, "MetaHarnessRepository", return_value=repository),
        ):
            auto_advance.advance_agenda(9, {}, journal, args)

        self.assertEqual(len(decide.call_args.args[0]), 2)
        self.assertEqual(repository.save_decision.call_count, 2)


class FrontierRationTests(unittest.TestCase):
    """An agenda must not lose its frontier ring to three spent problems or to
    a provider outage.

    Measured 2026-08-10: agendas 1,2,3,4,5,7,10 could no longer obtain a
    frontier packet - and so never reached portfolio or grant despite 14
    waiting candidates - because the pool stopped at the top 3 problems and a
    permanent 4-try counter had been burned on them. On agendas 1 and 10 every
    one of those tries died on `provider_unavailable:HTTPStatusError`, a
    transient evaluator outage that says nothing about the problem.
    """

    def _args(self):
        return mock.Mock(
            authority_token_cap=20000,
            evaluator_provider="p", evaluator_model="m", evaluator_family="f",
            proposer_provider="pp", proposer_family="pf",
        )

    def _problems(self, count):
        return [{"id": 100 + i, "problem_statement": f"s{i}"} for i in range(count)]

    def test_pool_rotates_past_the_problems_whose_ration_is_spent(self):
        state = {
            "frontier_packets": {},
            "frontier_attempts": {f"9:{100 + i}": 4 for i in range(3)},
        }
        journal = mock.Mock()
        with (
            mock.patch.object(auto_advance, "_rows", return_value=self._problems(6)),
            mock.patch.object(auto_advance, "FrontierAuthorityRepository") as repo,
            mock.patch.object(auto_advance, "run_bootstrap_evaluation",
                              return_value={"frontier_packet_id": 42, "gate_allowed": True}),
            mock.patch.object(auto_advance.db, "fetchone", return_value=None),
        ):
            repo.return_value.issue.return_value = 5
            packet = auto_advance.ensure_frontier_packet(9, state, journal, self._args())

        self.assertEqual(packet, 42, "an untried problem was available and unused")
        self.assertEqual(state["frontier_attempts"]["9:103"], 1)

    def test_pool_query_is_not_capped_at_three(self):
        state = {"frontier_packets": {}, "frontier_attempts": {}}
        with (
            mock.patch.object(auto_advance, "_rows", return_value=[]) as rows,
            mock.patch.object(auto_advance, "FrontierAuthorityRepository"),
        ):
            auto_advance.ensure_frontier_packet(9, state, mock.Mock(), self._args())
        self.assertEqual(rows.call_args.args[1][-1], auto_advance.FRONTIER_PROBLEM_POOL)
        self.assertGreater(auto_advance.FRONTIER_PROBLEM_POOL, 3)

    def test_transient_provider_failure_refunds_the_ration_and_stops_the_pass(self):
        state = {"frontier_packets": {}, "frontier_attempts": {}}
        journal = mock.Mock()
        outage = FrontierBootstrapError("evaluator route unavailable", transient=True)

        with (
            mock.patch.object(auto_advance, "_rows", return_value=self._problems(4)),
            mock.patch.object(auto_advance, "FrontierAuthorityRepository") as repo,
            mock.patch.object(auto_advance, "run_bootstrap_evaluation", side_effect=outage) as run,
            mock.patch.object(auto_advance.db, "rollback"),
        ):
            repo.return_value.issue.return_value = 5
            packet = auto_advance.ensure_frontier_packet(9, state, journal, self._args())

        self.assertIsNone(packet)
        self.assertEqual(state["frontier_attempts"]["9:100"], 0,
                         "an outage consumed a problem's non-renewable ration")
        self.assertEqual(run.call_count, 1,
                         "the pass kept calling a route it already knew was down")
        steps = [call.args[0] for call in journal.log.call_args_list]
        self.assertIn("frontier_bootstrap_transient", steps)

    def test_a_structural_refusal_still_consumes_the_ration(self):
        """Refunding must be limited to transient failures."""

        state = {"frontier_packets": {}, "frontier_attempts": {}}
        refusal = FrontierBootstrapError("linked evidence is unusable")

        with (
            mock.patch.object(auto_advance, "_rows", return_value=self._problems(1)),
            mock.patch.object(auto_advance, "FrontierAuthorityRepository") as repo,
            mock.patch.object(auto_advance, "run_bootstrap_evaluation", side_effect=refusal),
            mock.patch.object(auto_advance.db, "rollback"),
        ):
            repo.return_value.issue.return_value = 5
            auto_advance.ensure_frontier_packet(9, state, mock.Mock(), self._args())

        self.assertEqual(state["frontier_attempts"]["9:100"], 1)

    def test_retry_after_a_refund_uses_a_fresh_authority_key(self):
        """A failed authority burns its idempotency key, so the serial that
        builds the key must advance even when the ration is refunded."""

        state = {"frontier_packets": {}, "frontier_attempts": {}, "frontier_issues": {}}
        outage = FrontierBootstrapError("evaluator route unavailable", transient=True)
        keys = []

        def issue(authority):
            keys.append(authority.idempotency_key)
            return 5

        for _ in range(2):
            with (
                mock.patch.object(auto_advance, "_rows", return_value=self._problems(1)),
                mock.patch.object(auto_advance, "FrontierAuthorityRepository") as repo,
                mock.patch.object(auto_advance, "run_bootstrap_evaluation", side_effect=outage),
                mock.patch.object(auto_advance.db, "rollback"),
            ):
                repo.return_value.issue.side_effect = issue
                auto_advance.ensure_frontier_packet(9, state, mock.Mock(), self._args())

        self.assertEqual(len(keys), 2)
        self.assertNotEqual(keys[0], keys[1])

    def test_attempts_per_pass_stay_below_the_previous_behaviour(self):
        """Widening the pool must not widen the spend: each attempt is one real
        evaluator call."""

        self.assertLessEqual(auto_advance.FRONTIER_ATTEMPTS_PER_PASS, 3)
        state = {"frontier_packets": {}, "frontier_attempts": {}}
        refusal = FrontierBootstrapError("linked evidence is unusable")

        with (
            mock.patch.object(auto_advance, "_rows", return_value=self._problems(10)),
            mock.patch.object(auto_advance, "FrontierAuthorityRepository") as repo,
            mock.patch.object(auto_advance, "run_bootstrap_evaluation", side_effect=refusal) as run,
            mock.patch.object(auto_advance.db, "rollback"),
        ):
            repo.return_value.issue.return_value = 5
            auto_advance.ensure_frontier_packet(9, state, mock.Mock(), self._args())

        self.assertEqual(run.call_count, auto_advance.FRONTIER_ATTEMPTS_PER_PASS)


if __name__ == "__main__":
    unittest.main()
