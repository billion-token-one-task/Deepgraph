from __future__ import annotations

import unittest
from unittest import mock

from meta_harness import outcome_finalizer


class _Repository:
    def __init__(self, outcome_id: int = 41, error: Exception | None = None):
        self.outcome_id = outcome_id
        self.error = error
        self.calls: list[tuple[int, int]] = []

    def assemble_and_record_outcome(
        self, *, resource_grant_id: int, experiment_run_id: int
    ) -> int:
        self.calls.append((resource_grant_id, experiment_run_id))
        if self.error:
            raise self.error
        return self.outcome_id


class OutcomeFinalizerTests(unittest.TestCase):
    def _row(self, **updates):
        row = {
            "agenda_id": 3,
            "deep_insight_id": 7,
            "experiment_run_id": 11,
            "resource_grant_id": 13,
            "outcome_record_id": None,
        }
        row.update(updates)
        return row

    def test_terminal_run_is_assembled_and_closed_once(self):
        repository = _Repository()
        with (
            mock.patch.object(outcome_finalizer, "_recover_terminal_usage", return_value={}),
            mock.patch.object(outcome_finalizer, "_candidate_rows", return_value=[self._row()]),
            mock.patch.object(outcome_finalizer, "MetaHarnessRepository", return_value=repository),
            mock.patch.object(outcome_finalizer.db, "fetchone", return_value={"verdict": "refuted"}),
            mock.patch.object(outcome_finalizer, "_mark_closed") as mark_closed,
        ):
            report = outcome_finalizer.finalize_terminal_outcomes()

        self.assertEqual(repository.calls, [(13, 11)])
        self.assertEqual(report.finalized, [41])
        mark_closed.assert_called_once_with(self._row(), 41, "refuted")

    def test_existing_outcome_repairs_snapshot_without_duplicate_assembly(self):
        repository = _Repository()
        row = self._row(outcome_record_id=41)
        with (
            mock.patch.object(outcome_finalizer, "_recover_terminal_usage", return_value={}),
            mock.patch.object(outcome_finalizer, "_candidate_rows", return_value=[row]),
            mock.patch.object(outcome_finalizer, "MetaHarnessRepository", return_value=repository),
            mock.patch.object(outcome_finalizer.db, "fetchone", return_value={"verdict": "inconclusive"}),
            mock.patch.object(outcome_finalizer, "_mark_closed") as mark_closed,
        ):
            report = outcome_finalizer.finalize_terminal_outcomes()

        self.assertEqual(repository.calls, [])
        self.assertEqual(report.already_finalized, [41])
        mark_closed.assert_called_once_with(row, 41, "inconclusive")

    def test_open_usage_defers_without_fabricating_outcome(self):
        repository = _Repository(error=RuntimeError("open GPU attempts"))
        with (
            mock.patch.object(outcome_finalizer, "_recover_terminal_usage", return_value={}),
            mock.patch.object(outcome_finalizer, "_candidate_rows", return_value=[self._row()]),
            mock.patch.object(outcome_finalizer, "MetaHarnessRepository", return_value=repository),
            mock.patch.object(outcome_finalizer.db, "rollback") as rollback,
            mock.patch.object(outcome_finalizer, "_mark_closed") as mark_closed,
        ):
            report = outcome_finalizer.finalize_terminal_outcomes()

        self.assertIn(13, report.deferred)
        self.assertEqual(report.finalized, [])
        rollback.assert_called_once()
        mark_closed.assert_not_called()

    def test_mark_closed_emits_deduplicated_outcome_event(self):
        row = self._row()
        with (
            mock.patch.object(outcome_finalizer.db, "execute") as execute,
            mock.patch.object(outcome_finalizer.db, "commit"),
            mock.patch.object(outcome_finalizer.db, "emit_pipeline_event") as emit,
            mock.patch.object(outcome_finalizer, "apply_experiment_finished_deep") as finish,
        ):
            outcome_finalizer._mark_closed(row, 41, "refuted")

        self.assertIn("stage='outcome_recorded'", execute.call_args.args[0])
        finish.assert_called_once_with(
            7, verdict="refuted", success=False, inconclusive=False
        )
        self.assertEqual(emit.call_args.kwargs["dedupe_key"], "outcome_recorded:41")


if __name__ == "__main__":
    unittest.main()
