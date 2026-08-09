from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from meta_harness.attempt_gpu_usage import (
    AttemptGPUUsageError,
    GrantGPUUsageControl,
)


NOW = datetime.now(timezone.utc)


def _grant(**overrides):
    row = {
        "id": 18,
        "agenda_id": 11,
        "idea_id": 105,
        "reservation_id": 25,
        "status": "active",
        "max_gpu_hours": 2.0,
        "backend_allowlist_json": '["ssh_gpu"]',
        "expires_at": NOW + timedelta(hours=4),
    }
    row.update(overrides)
    return row


def _attempt(**overrides):
    row = {
        "id": 44,
        "agenda_id": 11,
        "idea_id": 105,
        "resource_grant_id": 18,
        "attempt_key": "run-133",
        "backend_kind": "ssh_gpu",
        "gpu_count": 1,
        "reserved_gpu_seconds": 1800.0,
        "timeout_seconds": 1800,
        "status": "reserved",
        "started_at": None,
        "completed_at": None,
        "actual_gpu_seconds": None,
        "reason_code": None,
    }
    row.update(overrides)
    return row


class AttemptGPUUsageTests(unittest.TestCase):
    def test_admission_subtracts_settled_and_active_reservations(self):
        def fetchone(sql, _params=()):
            if "FROM resource_grants WHERE id=" in sql:
                return _grant()
            if "WHERE resource_grant_id=? AND attempt_key=?" in sql:
                return None
            if "SUM(CASE WHEN status='settled'" in sql:
                return {
                    "settled_gpu_seconds": 3600.0,
                    "active_reserved_gpu_seconds": 1800.0,
                    "active_reservations": 1,
                }
            if "WHERE id=?" in sql and "experiment_attempt" in sql:
                return _attempt(timeout_seconds=1795)
            self.fail(sql)

        with (
            mock.patch("meta_harness.attempt_gpu_usage.db._use_pg", return_value=True),
            mock.patch("meta_harness.attempt_gpu_usage.db.fetchone", side_effect=fetchone),
            mock.patch(
                "meta_harness.attempt_gpu_usage.db.insert_returning_id",
                return_value=44,
            ) as insert,
            mock.patch("meta_harness.attempt_gpu_usage.db.commit"),
            mock.patch("meta_harness.attempt_gpu_usage.db.rollback"),
        ):
            reservation = GrantGPUUsageControl().reserve_attempt(
                agenda_id=11,
                idea_id=105,
                resource_grant_id=18,
                attempt_key="run-133",
                backend_kind="ssh_gpu",
                requested_timeout_seconds=3600,
            )

        self.assertEqual(reservation.timeout_seconds, 1795)
        inserted_params = insert.call_args.args[1]
        self.assertEqual(inserted_params[7], 1800.0)
        self.assertEqual(inserted_params[8], 1795)

    def test_concurrent_reservation_cannot_overbook_the_grant(self):
        def fetchone(sql, _params=()):
            if "FROM resource_grants WHERE id=" in sql:
                return _grant()
            if "WHERE resource_grant_id=? AND attempt_key=?" in sql:
                return None
            if "SUM(CASE WHEN status='settled'" in sql:
                return {
                    "settled_gpu_seconds": 3600.0,
                    "active_reserved_gpu_seconds": 3600.0,
                    "active_reservations": 1,
                }
            self.fail(sql)

        with (
            mock.patch("meta_harness.attempt_gpu_usage.db._use_pg", return_value=True),
            mock.patch("meta_harness.attempt_gpu_usage.db.fetchone", side_effect=fetchone),
            mock.patch(
                "meta_harness.attempt_gpu_usage.db.insert_returning_id"
            ) as insert,
            mock.patch("meta_harness.attempt_gpu_usage.db.rollback"),
        ):
            with self.assertRaisesRegex(
                AttemptGPUUsageError, "grant_gpu_hours_exhausted"
            ):
                GrantGPUUsageControl().reserve_attempt(
                    agenda_id=11,
                    idea_id=105,
                    resource_grant_id=18,
                    attempt_key="run-134",
                    backend_kind="ssh_gpu",
                    requested_timeout_seconds=60,
                )

        insert.assert_not_called()

    def test_duplicate_settlement_is_idempotent(self):
        settled = _attempt(
            status="settled",
            started_at=NOW - timedelta(minutes=10),
            completed_at=NOW,
            actual_gpu_seconds=600.0,
        )
        responses = [
            {"resource_grant_id": 18},
            _grant(status="consumed"),
            settled,
        ]
        with (
            mock.patch("meta_harness.attempt_gpu_usage.db._use_pg", return_value=True),
            mock.patch(
                "meta_harness.attempt_gpu_usage.db.fetchone",
                side_effect=responses,
            ),
            mock.patch("meta_harness.attempt_gpu_usage.db.execute") as execute,
            mock.patch("meta_harness.attempt_gpu_usage.db.commit") as commit,
            mock.patch("meta_harness.attempt_gpu_usage.db.rollback"),
        ):
            result = GrantGPUUsageControl().settle_attempt(
                44,
                completed_at=NOW + timedelta(minutes=5),
                reason_code="controller_lost",
            )

        self.assertEqual(result.actual_gpu_seconds, 600.0)
        execute.assert_not_called()
        commit.assert_called_once()

    def test_running_attempt_uses_persisted_start_for_remaining_timeout(self):
        row = _attempt(
            status="running",
            timeout_seconds=1200,
            started_at=NOW - timedelta(minutes=5),
        )
        with (
            mock.patch(
                "meta_harness.attempt_gpu_usage.db.fetchone", return_value=row
            ),
            mock.patch(
                "meta_harness.attempt_gpu_usage._now", return_value=NOW
            ),
            mock.patch("meta_harness.attempt_gpu_usage.db.commit"),
            mock.patch("meta_harness.attempt_gpu_usage.db.rollback"),
        ):
            remaining = GrantGPUUsageControl().remaining_attempt_wall_seconds(44)

        self.assertEqual(remaining, 900.0)


if __name__ == "__main__":
    unittest.main()
