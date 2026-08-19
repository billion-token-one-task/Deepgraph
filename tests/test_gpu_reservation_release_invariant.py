"""A GPU reservation may only be released once, across all terminal paths.

Attempt-level settlement (attempt_gpu_usage.settle_attempt) releases hours as
they are burned but deliberately leaves the ledger row 'reserved' so later
attempts on the same grant can still draw against it. Every terminal path --
revocation, expiry reconciliation, ledger release, ledger settle -- therefore
owes the agenda only the hours nobody has released yet.

Three of those paths subtracted the full cap instead. By 2026-08-19 agenda 7
sat at gpu_hours_reserved = -1.644 and agenda 10 at -4.608; a negative
reservation fails ResearchAgenda.validate(), so agenda 7's selector died with
"agenda GPU accounting cannot be negative" and it could not select any work.
"""

import unittest

from agents.agenda_repository import _outstanding_gpu_hours as agenda_outstanding
from meta_harness.repository import _outstanding_gpu_hours as grant_outstanding


class GpuReservationReleaseInvariantTests(unittest.TestCase):
    def test_partially_burned_reservation_releases_only_the_remainder(self):
        for outstanding in (agenda_outstanding, grant_outstanding):
            row = {"gpu_hours_reserved": 4.0, "gpu_hours_used": 0.5894074591666667}
            self.assertAlmostEqual(
                outstanding(row), 3.4105925408333333, places=12
            )

    def test_untouched_reservation_releases_its_whole_cap(self):
        for outstanding in (agenda_outstanding, grant_outstanding):
            self.assertEqual(
                outstanding({"gpu_hours_reserved": 2.0, "gpu_hours_used": None}), 2.0
            )

    def test_fully_burned_reservation_releases_nothing(self):
        # grant 206's shape: the cap was consumed exactly, plus an overrun
        for outstanding in (agenda_outstanding, grant_outstanding):
            self.assertEqual(
                outstanding({"gpu_hours_reserved": 2.0, "gpu_hours_used": 2.0}), 0.0
            )

    def test_overrun_never_produces_a_negative_release(self):
        for outstanding in (agenda_outstanding, grant_outstanding):
            self.assertEqual(
                outstanding({"gpu_hours_reserved": 2.0, "gpu_hours_used": 2.9}), 0.0
            )


if __name__ == "__main__":
    unittest.main()
