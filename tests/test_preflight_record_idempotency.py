from __future__ import annotations

import unittest
from unittest import mock

from meta_harness import preflight_repository
from meta_harness.runner_capability import PreflightEnvironment, PreflightResult


ENVIRONMENT = PreflightEnvironment(
    enabled_backends=("ssh_gpu",),
    backend_vram_gb={"ssh_gpu": 22.0},
    network_available=True,
    disk_free_gb=100.0,
)


class RecordIdempotencyTests(unittest.TestCase):
    """The hourly key must not freeze a provisional answer for an hour.

    ``deferred: backend_unavailable`` is provisional by design -- the code
    that writes it says it heals when hardware appears. Returning that row
    for a later run that passed handed grant authority a preflight which had
    not passed, and the grant was refused against the candidate's own earlier
    deferral.
    """

    def _record(self, *, stored, new_status):
        rows = list(stored)
        inserted: list[str] = []

        def fetchone(sql, params=()):
            key = params[1]
            for row in rows:
                if row["idempotency_key"] == key:
                    return dict(row)
            return None

        def insert_returning_id(sql, params):
            inserted.append(params[-1])
            return 900 + len(inserted)

        result = PreflightResult(new_status, (), {})
        with mock.patch.object(preflight_repository.db, "fetchone", fetchone), \
             mock.patch.object(
                 preflight_repository.db, "insert_returning_id", insert_returning_id
             ), \
             mock.patch.object(preflight_repository.db, "commit", lambda: None):
            result_id = preflight_repository.CandidatePreflightRepository().record(
                agenda_id=16,
                idea_id=237,
                requirement_id=7,
                result=result,
                environment=ENVIRONMENT,
                idempotency_key="preflight:7:2026082610",
            )
        return result_id, inserted

    def test_the_same_answer_in_the_same_hour_is_a_no_op(self):
        stored = [{"id": 277, "status": "deferred",
                   "idempotency_key": "preflight:7:2026082610"}]
        result_id, inserted = self._record(stored=stored, new_status="deferred")
        self.assertEqual(result_id, 277)
        self.assertEqual(inserted, [])

    def test_a_changed_answer_in_the_same_hour_is_recorded(self):
        stored = [{"id": 277, "status": "deferred",
                   "idempotency_key": "preflight:7:2026082610"}]
        result_id, inserted = self._record(stored=stored, new_status="passed")
        self.assertNotEqual(result_id, 277)
        self.assertEqual(inserted, ["preflight:7:2026082610:passed"])

    def test_the_changed_answer_is_itself_idempotent(self):
        stored = [
            {"id": 277, "status": "deferred",
             "idempotency_key": "preflight:7:2026082610"},
            {"id": 280, "status": "passed",
             "idempotency_key": "preflight:7:2026082610:passed"},
        ]
        result_id, inserted = self._record(stored=stored, new_status="passed")
        self.assertEqual(result_id, 280)
        self.assertEqual(inserted, [])


if __name__ == "__main__":
    unittest.main()
