from __future__ import annotations

import unittest
from unittest import mock

from meta_harness import outcome_finalizer


class StrandedTokenReservationTests(unittest.TestCase):
    """Token budget held by a dead call must come back.

    GrantUsage.release is only called on the normal completion path, and the
    sweep beside it covers GPU-hour reservations, not token ones. Idea 241's
    validation_code_iteration held 17,275 of a 40,000 token grant after the
    staleness sweep reclaimed its run twenty seconds later, and every
    subsequent forge failed with "ResourceGrant token budget is exhausted".
    """

    def _sweep(self, rows, release_raises=False):
        released: list[tuple[int, str]] = []

        class Ledger:
            def __init__(self, grant_id):
                self.grant_id = grant_id

            def release(self, reservation_id, *, reason):
                if release_raises:
                    raise RuntimeError("stuck row")
                released.append((reservation_id, reason))

        with mock.patch.object(outcome_finalizer.db, "fetchall", lambda *a, **k: rows), \
             mock.patch.object(outcome_finalizer, "GrantUsageLedger", Ledger), \
             mock.patch.object(outcome_finalizer.db, "rollback", lambda: None):
            count = outcome_finalizer._release_stranded_token_reservations()
        return count, released

    def test_an_expired_reservation_with_no_live_work_is_released(self):
        rows = [{"id": 2872, "resource_grant_id": 362,
                 "operation": "validation_code_iteration", "token_reserved": 17275}]
        count, released = self._sweep(rows)
        self.assertEqual(count, 1)
        self.assertEqual(released[0][0], 2872)
        self.assertIn("lease_expired_no_live_work", released[0][1])

    def test_nothing_to_reclaim_is_not_an_error(self):
        count, released = self._sweep([])
        self.assertEqual(count, 0)
        self.assertEqual(released, [])

    def test_one_stuck_row_does_not_stop_the_sweep(self):
        rows = [{"id": 1, "resource_grant_id": 9, "operation": "x", "token_reserved": 1}]
        count, _ = self._sweep(rows, release_raises=True)
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
