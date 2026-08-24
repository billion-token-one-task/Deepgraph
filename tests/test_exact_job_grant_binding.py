"""Transaction and replay invariants for one-job controlled recovery."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from contracts.meta_harness import ResourceGrant
from meta_harness.repository import (
    MetaHarnessPersistenceError,
    MetaHarnessRepository,
)


def _grant() -> ResourceGrant:
    return ResourceGrant(
        agenda_id=2,
        idea_id=115,
        decision_packet_id=41,
        stage="proposal",
        token_cap=1_000,
        gpu_class="none",
        max_gpu_hours=0.0,
        backend_allowlist=["llm"],
        artifact_requirements=["candidate_stage_gate_record"],
        expires_at=(
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat(),
        grant_reason="controlled job110 proposal",
        idempotency_key="controlled-recovery:job110:proposal:v1",
    )


def _agenda() -> dict:
    return {
        "id": 2,
        "status": "active",
        "max_concurrency": 2,
        "token_budget": 50_000,
        "token_spent": 0,
        "token_reserved": 0,
        "gpu_hours_budget": 0.0,
        "gpu_hours_spent": 0.0,
        "gpu_hours_reserved": 0.0,
        "backend_allowlist_json": '["llm"]',
        "prefer_json": "{}",
    }


def _decision() -> dict:
    return {"agenda_id": 2, "idea_id": 115, "decision": "promote"}


def _target(*, bound: bool = False) -> dict:
    return {
        "id": 110,
        "agenda_id": 2,
        "deep_insight_id": 115,
        "status": "deferred" if bound else "queued",
        "stage": (
            "proposal_generation_granted"
            if bound
            else "awaiting_portfolio_decision"
        ),
        "resource_grant_id": 193 if bound else None,
    }


def _existing(grant: ResourceGrant, *, live: bool = True) -> dict:
    return {
        "id": 193,
        "agenda_id": grant.agenda_id,
        "idea_id": grant.idea_id,
        "decision_packet_id": grant.decision_packet_id,
        "stage": grant.stage,
        "token_cap": grant.token_cap,
        "gpu_class": grant.gpu_class,
        "max_gpu_hours": grant.max_gpu_hours,
        "backend_allowlist_json": json.dumps(grant.backend_allowlist),
        "artifact_requirements_json": json.dumps(grant.artifact_requirements),
        "expires_at": datetime.fromisoformat(grant.expires_at),
        "grant_reason": grant.grant_reason,
        "reservation_id": 71,
        "preflight_result_id": grant.preflight_result_id,
        "status": "active" if live else "expired",
        "grant_live": 1 if live else 0,
    }


class ExactJobGrantBindingTests(unittest.TestCase):
    def _guards(self):
        return (
            mock.patch(
                "meta_harness.repository._require_proposal_funding_headroom"
            ),
            mock.patch("meta_harness.repository._require_short_ttl"),
            mock.patch("meta_harness.repository._require_schedulable_backends"),
            mock.patch("meta_harness.repository._require_execution_preflight"),
        )

    def test_new_grant_and_exact_job_binding_commit_as_one_transaction(self):
        grant = _grant()
        guards = self._guards()
        with (
            mock.patch("meta_harness.repository.db") as database,
            guards[0],
            guards[1],
            guards[2],
            guards[3],
        ):
            database._use_pg.return_value = True
            database.fetchone.side_effect = [
                _agenda(),
                _decision(),
                _target(),
                None,
                {"count": 0},
            ]
            database.insert_returning_id.side_effect = [71, 193]
            database.execute.return_value = SimpleNamespace(rowcount=1)

            grant_id = MetaHarnessRepository().issue_grant(
                grant, target_job_id=110
            )

        self.assertEqual(grant_id, 193)
        bind_sql = database.execute.call_args_list[-1].args[0]
        bind_params = database.execute.call_args_list[-1].args[1]
        self.assertIn("WHERE id=?", bind_sql)
        self.assertIn("resource_grant_id IS NULL", bind_sql)
        self.assertEqual(bind_params, (193, 110, 2, 115))
        database.commit.assert_called_once_with()
        database.rollback.assert_not_called()

    def test_binding_compare_and_set_failure_rolls_back_grant_and_ledger(self):
        grant = _grant()
        guards = self._guards()
        with (
            mock.patch("meta_harness.repository.db") as database,
            guards[0],
            guards[1],
            guards[2],
            guards[3],
        ):
            database._use_pg.return_value = True
            database.fetchone.side_effect = [
                _agenda(),
                _decision(),
                _target(),
                None,
                {"count": 0},
            ]
            database.insert_returning_id.side_effect = [71, 193]
            database.execute.side_effect = [
                SimpleNamespace(rowcount=1),
                SimpleNamespace(rowcount=0),
            ]

            with self.assertRaisesRegex(
                MetaHarnessPersistenceError, "bind exact proposal job"
            ):
                MetaHarnessRepository().issue_grant(grant, target_job_id=110)

        database.commit.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_exact_idempotent_replay_requires_same_bound_job_and_live_grant(self):
        for live, error in ((True, None), (False, "not live")):
            with self.subTest(error=error):
                grant = _grant()
                existing = _existing(grant, live=live)
                target = _target(bound=True)
                with mock.patch("meta_harness.repository.db") as database:
                    database._use_pg.return_value = True
                    database.fetchone.side_effect = [
                        _agenda(),
                        _decision(),
                        target,
                        existing,
                    ]
                    if error:
                        with self.assertRaisesRegex(
                            MetaHarnessPersistenceError, error
                        ):
                            MetaHarnessRepository().issue_grant(
                                grant, target_job_id=110
                            )
                        database.commit.assert_not_called()
                        database.rollback.assert_called_once_with()
                    else:
                        self.assertEqual(
                            MetaHarnessRepository().issue_grant(
                                grant, target_job_id=110
                            ),
                            193,
                        )
                        database.commit.assert_called_once_with()
                        database.rollback.assert_not_called()
                    database.insert_returning_id.assert_not_called()
                    database.execute.assert_not_called()

    def test_idempotency_key_cannot_be_reused_for_different_authority(self):
        grant = _grant()
        conflicting = _existing(grant)
        conflicting["token_cap"] = grant.token_cap + 1
        with mock.patch("meta_harness.repository.db") as database:
            database._use_pg.return_value = True
            database.fetchone.side_effect = [
                _agenda(),
                _decision(),
                _target(bound=True),
                conflicting,
            ]
            with self.assertRaisesRegex(
                MetaHarnessPersistenceError, "conflicts with a different"
            ):
                MetaHarnessRepository().issue_grant(grant, target_job_id=110)

        database.insert_returning_id.assert_not_called()
        database.execute.assert_not_called()
        database.commit.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_exact_proposal_completion_cas_closes_grant_and_named_job(self):
        grant = {
            "id": 193,
            "agenda_id": 2,
            "idea_id": 115,
            "stage": "proposal",
            "status": "active",
            "reservation_id": 71,
            "token_cap": 1_000,
        }
        target = _target(bound=True)
        usage = {"tokens_used": 600, "open_reservations": 0}
        ledger = {
            "id": 71,
            "status": "reserved",
            "token_reserved": 1_000,
        }
        with mock.patch("meta_harness.repository.db") as database:
            database._use_pg.return_value = True
            database.fetchone.side_effect = [
                {**_agenda(), "token_reserved": 1_000},
                target,
                grant,
                usage,
                ledger,
            ]
            database.execute.return_value = SimpleNamespace(rowcount=1)

            tokens = MetaHarnessRepository().complete_proposal_generation(
                grant_id=193,
                agenda_id=2,
                idea_id=115,
                target_job_id=110,
            )

        self.assertEqual(tokens, 600)
        statements = [call.args[0] for call in database.execute.call_args_list]
        self.assertEqual(len(statements), 5)
        job_sql = statements[3]
        job_params = database.execute.call_args_list[3].args[1]
        self.assertIn("WHERE id=?", job_sql)
        self.assertIn("status='deferred'", job_sql)
        self.assertEqual(job_params, (110, 2, 115, 193))
        database.commit.assert_called_once_with()
        database.rollback.assert_not_called()

    def test_exact_proposal_completion_job_race_rolls_back_all_settlement(self):
        with mock.patch("meta_harness.repository.db") as database:
            database._use_pg.return_value = True
            database.fetchone.side_effect = [
                {**_agenda(), "token_reserved": 1_000},
                _target(bound=True),
                {
                    "id": 193,
                    "agenda_id": 2,
                    "idea_id": 115,
                    "stage": "proposal",
                    "status": "active",
                    "reservation_id": 71,
                    "token_cap": 1_000,
                },
                {"tokens_used": 600, "open_reservations": 0},
                {
                    "id": 71,
                    "status": "reserved",
                    "token_reserved": 1_000,
                },
            ]
            database.execute.side_effect = [
                SimpleNamespace(rowcount=1),
                SimpleNamespace(rowcount=1),
                SimpleNamespace(rowcount=1),
                SimpleNamespace(rowcount=0),
            ]

            with self.assertRaisesRegex(
                MetaHarnessPersistenceError, "complete exact proposal job"
            ):
                MetaHarnessRepository().complete_proposal_generation(
                    grant_id=193,
                    agenda_id=2,
                    idea_id=115,
                    target_job_id=110,
                )

        database.commit.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_consumed_exact_proposal_replay_requires_job_reset_truth(self):
        for target, error in (
            (_target(bound=False), None),
            (_target(bound=True), "not reflected"),
        ):
            with self.subTest(error=error), mock.patch(
                "meta_harness.repository.db"
            ) as database:
                database._use_pg.return_value = True
                database.fetchone.side_effect = [
                    {"id": 2},
                    target,
                    {
                        "id": 193,
                        "agenda_id": 2,
                        "idea_id": 115,
                        "stage": "proposal",
                        "status": "consumed",
                        "reservation_id": 71,
                        "token_cap": 1_000,
                    },
                    {"tokens_used": 600, "open_reservations": 0},
                    {
                        "id": 71,
                        "status": "settled",
                        "tokens_used": 600,
                        "gpu_hours_used": 0.0,
                    },
                ]
                if error:
                    with self.assertRaisesRegex(
                        MetaHarnessPersistenceError, error
                    ):
                        MetaHarnessRepository().complete_proposal_generation(
                            grant_id=193,
                            agenda_id=2,
                            idea_id=115,
                            target_job_id=110,
                        )
                    database.commit.assert_not_called()
                    database.rollback.assert_called_once_with()
                else:
                    self.assertEqual(
                        MetaHarnessRepository().complete_proposal_generation(
                            grant_id=193,
                            agenda_id=2,
                            idea_id=115,
                            target_job_id=110,
                        ),
                        600,
                    )
                    database.commit.assert_called_once_with()
                    database.rollback.assert_not_called()
                database.execute.assert_not_called()

    def test_consumed_proposal_replay_requires_matching_settled_ledger(self):
        with mock.patch("meta_harness.repository.db") as database:
            database._use_pg.return_value = True
            database.fetchone.side_effect = [
                {"id": 2},
                _target(bound=False),
                {
                    "id": 193,
                    "agenda_id": 2,
                    "idea_id": 115,
                    "stage": "proposal",
                    "status": "consumed",
                    "reservation_id": 71,
                    "token_cap": 1_000,
                },
                {"tokens_used": 600, "open_reservations": 0},
                {
                    "id": 71,
                    "status": "settled",
                    "tokens_used": 599,
                    "gpu_hours_used": 0.0,
                },
            ]

            with self.assertRaisesRegex(
                MetaHarnessPersistenceError, "inconsistent settlement"
            ):
                MetaHarnessRepository().complete_proposal_generation(
                    grant_id=193,
                    agenda_id=2,
                    idea_id=115,
                    target_job_id=110,
                )

        database.execute.assert_not_called()
        database.commit.assert_not_called()
        database.rollback.assert_called_once_with()

    def test_active_proposal_refuses_inconsistent_agenda_reservation(self):
        with mock.patch("meta_harness.repository.db") as database:
            database._use_pg.return_value = True
            database.fetchone.side_effect = [
                {**_agenda(), "token_reserved": 999},
                _target(bound=True),
                {
                    "id": 193,
                    "agenda_id": 2,
                    "idea_id": 115,
                    "stage": "proposal",
                    "status": "active",
                    "reservation_id": 71,
                    "token_cap": 1_000,
                },
                {"tokens_used": 600, "open_reservations": 0},
                {"id": 71, "status": "reserved", "token_reserved": 1_000},
            ]

            with self.assertRaisesRegex(
                MetaHarnessPersistenceError, "agenda accounting"
            ):
                MetaHarnessRepository().complete_proposal_generation(
                    grant_id=193,
                    agenda_id=2,
                    idea_id=115,
                    target_job_id=110,
                )

        database.execute.assert_not_called()
        database.commit.assert_not_called()
        database.rollback.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
