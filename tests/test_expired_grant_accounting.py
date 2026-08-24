"""Backend-neutral accounting checks for expired ResourceGrants."""

from __future__ import annotations

import unittest
from unittest import mock

from meta_harness.repository import MetaHarnessPersistenceError, MetaHarnessRepository


class ExpiredGrantAccountingTests(unittest.TestCase):
    def test_expired_ingestion_grant_with_open_usage_is_left_untouched(self):
        grant = {
            "id": 9,
            "agenda_id": 14,
            "reservation_id": 21,
            "stage": "ingestion_backfill_canary",
            "token_reserved": 100,
            "gpu_hours_reserved": 0.0,
            "gpu_hours_used": 0.0,
            "reservation_status": "reserved",
        }
        with mock.patch(
            "meta_harness.repository.db._use_pg", return_value=False
        ), mock.patch(
            "meta_harness.repository.db.fetchall",
            side_effect=[[grant], [{"id": 91, "status": "reserved"}]],
        ), mock.patch(
            "meta_harness.repository.db.execute"
        ) as execute, mock.patch(
            "meta_harness.repository.db.commit"
        ) as commit, mock.patch("meta_harness.repository.db.rollback"):
            count = MetaHarnessRepository().reconcile_expired_grants(agenda_id=14)

        self.assertEqual(count, 0)
        execute.assert_not_called()
        commit.assert_called_once_with()

    def test_early_expiry_refuses_open_ingestion_usage(self):
        with mock.patch(
            "meta_harness.repository.db._use_pg", return_value=False
        ), mock.patch(
            "meta_harness.repository.db.fetchone",
            side_effect=[
                {"id": 9, "stage": "ingestion_backfill_canary"},
                {"id": 91},
            ],
        ), mock.patch(
            "meta_harness.repository.db.execute"
        ) as execute, mock.patch(
            "meta_harness.repository.db.rollback"
        ) as rollback:
            with self.assertRaisesRegex(
                MetaHarnessPersistenceError, "exact usage disposition"
            ):
                MetaHarnessRepository().expire_grant_now(
                    9, agenda_id=14, reason="operator recovery"
                )

        execute.assert_not_called()
        rollback.assert_called_once_with()

    def test_sqlite_path_clamps_aggregate_drift_and_records_both_shortfalls(self):
        grant = {
            "id": 9,
            "agenda_id": 14,
            "reservation_id": 21,
            "token_reserved": 40000,
            "gpu_hours_reserved": 4.0,
            "gpu_hours_used": 0.0,
            "reservation_status": "reserved",
        }
        child_usage = [{"status": "settled", "tokens_used": 30000}]
        agenda = {"token_reserved": 25000, "gpu_hours_reserved": 3.5}

        with mock.patch(
            "meta_harness.repository.db._use_pg", return_value=False
        ), mock.patch(
            "meta_harness.repository.db.fetchall",
            side_effect=[[grant], child_usage],
        ), mock.patch(
            "meta_harness.repository.db.fetchone", return_value=agenda
        ), mock.patch(
            "meta_harness.repository.db.execute"
        ) as execute, mock.patch(
            "meta_harness.repository.db.commit"
        ), mock.patch(
            "meta_harness.repository.db.rollback"
        ):
            count = MetaHarnessRepository().reconcile_expired_grants(agenda_id=14)

        self.assertEqual(count, 1)
        agenda_update = next(
            call
            for call in execute.call_args_list
            if "UPDATE research_agendas" in str(call.args[0])
        )
        self.assertNotIn("GREATEST", str(agenda_update.args[0]))
        self.assertEqual(agenda_update.args[1], (0, 30000, 0.0, 14))

        ledger_update = next(
            call
            for call in execute.call_args_list
            if "UPDATE agenda_resource_ledger" in str(call.args[0])
        )
        self.assertEqual(ledger_update.args[1][0], 30000)
        self.assertEqual(ledger_update.args[1][1], "settled")
        self.assertIn(
            "agenda_token_reservation_shortfall=15000",
            ledger_update.args[1][2],
        )
        self.assertIn(
            "agenda_gpu_reservation_shortfall_hours=0.5",
            ledger_update.args[1][2],
        )
        # The ledger UPDATE does not write gpu_hours_used: drift is disclosed,
        # never disguised as measured accelerator usage.
        self.assertNotIn("gpu_hours_used=", str(ledger_update.args[0]))


if __name__ == "__main__":
    unittest.main()
