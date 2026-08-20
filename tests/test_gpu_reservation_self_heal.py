"""A drifted reservation cache must not be able to stop an agenda dead.

research_agendas.gpu_hours_reserved is a running cache of what the ledger
already records. When it drifts negative, ResearchAgenda.validate() refuses
the row, and an agenda that cannot validate selects no work at all -- so a
bookkeeping error becomes a total stop.

It happened twice. Agenda 10 sat at -1.644 on 2026-08-19, and at -0.693 on
2026-08-20 while the M2 window was three runs from finishing and all three
GPU lanes were idle with candidates queued behind them.

Four over-releasing paths were fixed at the source (floors73). The recurrence
proves at least one more exists. Until it is found, every auto-advance pass
re-derives the cache from the ledger and logs each correction, so the
underlying defect stays visible rather than being silently absorbed.
"""

import inspect
import unittest

import scripts.auto_advance as auto_advance


class GpuReservationSelfHealTests(unittest.TestCase):
    def _source(self):
        return inspect.getsource(auto_advance._reconcile_gpu_reservation_drift)

    def test_it_runs_before_any_agenda_is_read(self):
        main = inspect.getsource(auto_advance.main)
        recon = main.index("_reconcile_gpu_reservation_drift")
        # nothing that selects work may precede the reconcile
        self.assertLess(recon, main.index("try:"))

    def test_the_ledger_is_the_source_of_truth(self):
        source = self._source()
        self.assertIn("FROM agenda_resource_ledger", source)
        self.assertIn("status='reserved'", source)
        self.assertIn("GREATEST(", source)

    def test_every_correction_is_logged_with_the_drift(self):
        # silently fixing it would hide the defect that keeps causing it
        source = self._source()
        self.assertIn("gpu_reservation_drift_reconciled", source)
        self.assertIn("drift=stored - expected", source)

    def test_an_exact_match_writes_nothing(self):
        source = self._source()
        self.assertIn("if abs(stored - expected) <= 1e-9:", source)
        self.assertIn("continue", source)

    def test_a_reconcile_failure_never_aborts_the_pass(self):
        source = self._source()
        self.assertIn("except Exception as exc:", source)
        self.assertIn("db.rollback()", source)
        self.assertIn("gpu_reservation_reconcile_failed", source)


if __name__ == "__main__":
    unittest.main()
