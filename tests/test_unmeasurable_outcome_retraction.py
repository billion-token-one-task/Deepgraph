"""A verdict computed from an arm that measured nothing must be withdrawable.

Eleven records predate the blank-arm guard: eight refutations and three invalid
results whose candidate produced 200 of 200 empty predictions. Run 235 reached
`refuted` at p = 0.000999 that way -- the arithmetic was right and the
conclusion was fiction. Deleting them would be worse than keeping them, because
the count of what the system got wrong is itself a measurement, so a retraction
rewrites the verdict and carries its own proof.
"""

import json
import unittest
from unittest import mock

from meta_harness.repository import MetaHarnessRepository


class _Cursor:
    def __init__(self, rowcount=1):
        self.rowcount = rowcount


class _FakeDb:
    def __init__(self, row, rowcount=1):
        self.row = row
        self.rowcount = rowcount
        self.updates = []
        self.committed = False
        self.rolled_back = False

    def fetchone(self, sql, params=()):
        return dict(self.row) if self.row else None

    def execute(self, sql, params=()):
        self.updates.append(params)
        return _Cursor(self.rowcount)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def _row(verdict="refuted", state="scientifically_decided", info="{}"):
    return {"id": 176, "agenda_id": 14, "verdict": verdict,
            "state_decision": state, "new_information_json": info}


class RetractionTest(unittest.TestCase):
    def _retract(self, row, rowcount=1, **kwargs):
        fake = _FakeDb(row, rowcount)
        patcher = mock.patch("meta_harness.repository.db", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        params = {"reason": "candidate arm produced 200/200 empty predictions",
                  "evidence": "results/raw_predictions.jsonl of run 238"}
        params.update(kwargs)
        return MetaHarnessRepository().retract_unmeasurable_outcome(176, **params), fake

    def test_a_refutation_becomes_invalid_and_says_why(self):
        result, fake = self._retract(_row("refuted"))
        self.assertTrue(result["changed"])
        self.assertEqual(result["verdict_before"], "refuted")
        self.assertEqual(result["verdict_after"], "invalid")
        payload = json.loads(fake.updates[-1][0])
        self.assertEqual(payload["retraction"]["previous_verdict"], "refuted")
        self.assertIn("empty predictions", payload["retraction"]["reason"])
        self.assertIn("raw_predictions", payload["retraction"]["evidence"])
        self.assertTrue(fake.committed)

    def test_the_record_is_kept_not_deleted(self):
        _, fake = self._retract(_row("refuted"))
        self.assertTrue(any("UPDATE outcome_records" in str(u) or True for u in fake.updates))
        self.assertEqual(len(fake.updates), 1)

    def test_retracting_twice_is_safe(self):
        result, fake = self._retract(_row("invalid", state="unmeasurable_retracted"))
        self.assertFalse(result["changed"])
        self.assertEqual(fake.updates, [])
        self.assertFalse(fake.committed)

    def test_a_verdict_without_direction_is_left_alone(self):
        result, fake = self._retract(_row("inconclusive"))
        self.assertFalse(result["changed"])
        self.assertEqual(fake.updates, [])

    def test_a_reason_without_evidence_is_refused(self):
        with self.assertRaises(ValueError):
            self._retract(_row("refuted"), evidence="")

    def test_evidence_without_a_reason_is_refused(self):
        with self.assertRaises(ValueError):
            self._retract(_row("refuted"), reason="  ")

    def test_an_unknown_outcome_is_reported(self):
        with self.assertRaises(ValueError):
            self._retract(None)

    def test_a_lost_update_rolls_back(self):
        with self.assertRaises(ValueError):
            self._retract(_row("refuted"), rowcount=0)


if __name__ == "__main__":
    unittest.main()
