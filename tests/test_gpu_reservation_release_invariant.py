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


class LedgerSettlementDirectionTests(unittest.TestCase):
    """The settlement gate must distinguish unmetered spend from phantom charge.

    Grant 135's ledger held 2.6193 GPU-hours while its four holdout attempts
    metered 2.4900 -- 466 phantom seconds left behind by a reopened attempt.
    The gate compared abs(gap) and refused, so an audited run that had already
    reached scientifically_decided could never settle its grant (2026-08-19).

    Under-charge is the dangerous direction: hours burned that nobody billed.
    Over-charge is a bookkeeping error against a canonical record that exists.
    """

    @staticmethod
    def _tolerance(actual):
        return max(150.0 / 3600.0, 0.05 * actual)

    def test_grant_135_over_charge_is_outside_the_old_symmetric_tolerance(self):
        ledger, metered = 2.6193461413888883, 2.4899507938888883
        gap = ledger - metered
        self.assertGreater(gap, self._tolerance(metered))
        self.assertGreater(gap, 0, "this is over-charge, not unmetered spend")

    def test_source_refuses_only_the_under_charge_direction(self):
        import inspect

        from meta_harness import repository

        source = inspect.getsource(repository.MetaHarnessRepository.record_outcome)
        self.assertIn("if gpu_gap < -tolerance:", source)
        self.assertNotIn("gpu_gap = abs(", source)


if __name__ == "__main__":
    unittest.main()
