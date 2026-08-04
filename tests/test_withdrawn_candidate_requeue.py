"""A withdrawn grant must not strand its candidate forever.

Authority is withdrawn two ways: expiry parks the candidate at
``resource_grant_expired``, revocation at ``resource_grant_revoked``. Neither
could move again, because ``issue_grant`` only re-points a job sitting at
``awaiting_portfolio_decision``. These tests pin the narrow return path and the
guards that stop it from becoming a way to re-spend settled work.
"""

from __future__ import annotations

import unittest
from unittest import mock

from meta_harness.repository import MetaHarnessPersistenceError, MetaHarnessRepository


AGENDA_ID = 5
IDEA_ID = 97
GRANT_ID = 1
JOB_ID = 42


class FakeDb:
    def __init__(self, *, job=None, grant=None, outcome=None):
        self.job = job
        self.grant = grant
        self.outcome = outcome
        self.statements: list[tuple[str, tuple]] = []
        self.commits = 0
        self.rollbacks = 0

    def _use_pg(self):
        return False

    def fetchone(self, sql, params=()):
        text = " ".join(sql.split()).lower()
        if "from auto_research_jobs" in text:
            return dict(self.job) if self.job else None
        if "from resource_grants" in text:
            return dict(self.grant) if self.grant else None
        if "from outcome_records" in text:
            return dict(self.outcome) if self.outcome else None
        raise AssertionError(f"unexpected fetchone: {text}")

    def execute(self, sql, params=()):
        self.statements.append((" ".join(sql.split()), params))
        return mock.Mock(rowcount=1)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _job(**overrides) -> dict:
    values = {
        "id": JOB_ID,
        "stage": "resource_grant_expired",
        "status": "blocked",
        "resource_grant_id": GRANT_ID,
    }
    values.update(overrides)
    return values


def _requeue(fake_db, **overrides):
    kwargs = {"agenda_id": AGENDA_ID, "idea_id": IDEA_ID, "reason": "pilot clock ran out"}
    kwargs.update(overrides)
    with mock.patch("meta_harness.repository.db", fake_db):
        return MetaHarnessRepository().requeue_withdrawn_candidate(**kwargs)


class RequeueWithdrawnCandidateTests(unittest.TestCase):
    def test_an_expired_candidate_returns_to_the_portfolio_queue(self):
        fake_db = FakeDb(job=_job(), grant={"status": "expired"})

        self.assertTrue(_requeue(fake_db))

        sql, params = fake_db.statements[0]
        self.assertIn("stage='awaiting_portfolio_decision'", sql)
        self.assertIn("WHERE id=? AND agenda_id=?", sql)
        self.assertIn("resource_grant_expired", sql)
        # The stale pointer must go, or the dead grant gets picked up again.
        self.assertIn("resource_grant_id=NULL", sql)
        self.assertIn("pilot clock ran out", params[0])
        self.assertEqual(fake_db.commits, 1)

    def test_a_revoked_and_refunded_candidate_also_returns(self):
        """Revocation refunds and refuses once usage is metered, same as expiry."""
        fake_db = FakeDb(
            job=_job(stage="resource_grant_revoked"), grant={"status": "revoked"}
        )

        self.assertTrue(_requeue(fake_db))

        sql, params = fake_db.statements[0]
        self.assertIn("resource_grant_revoked", sql)
        self.assertIn("resource_grant_revoked", params[0])

    def test_a_job_at_any_other_stage_is_left_alone(self):
        for stage in ("portfolio_granted", "awaiting_portfolio_decision", "pilot_running"):
            fake_db = FakeDb(job=_job(stage=stage), grant={"status": "expired"})

            self.assertFalse(_requeue(fake_db), stage)
            self.assertEqual(fake_db.statements, [])

    def test_a_missing_job_is_not_an_error_and_changes_nothing(self):
        fake_db = FakeDb(job=None)

        self.assertFalse(_requeue(fake_db))
        self.assertEqual(fake_db.statements, [])

    def test_a_live_or_settled_grant_is_never_requeued_over(self):
        for status in ("active", "consumed"):
            fake_db = FakeDb(job=_job(), grant={"status": status})

            with self.assertRaisesRegex(
                MetaHarnessPersistenceError, "expired or revoked"
            ):
                _requeue(fake_db)
            self.assertEqual(fake_db.statements, [])
            self.assertEqual(fake_db.rollbacks, 1)

    def test_settled_work_cannot_be_requeued_and_respent(self):
        fake_db = FakeDb(job=_job(), grant={"status": "expired"}, outcome={"id": 9})

        with self.assertRaisesRegex(MetaHarnessPersistenceError, "OutcomeRecord"):
            _requeue(fake_db)
        self.assertEqual(fake_db.statements, [])

    def test_scope_and_reason_are_required(self):
        fake_db = FakeDb(job=_job(), grant={"status": "expired"})

        for override in ({"agenda_id": 0}, {"idea_id": 0}, {"reason": "  "}):
            with self.assertRaises(MetaHarnessPersistenceError):
                _requeue(fake_db, **override)
        self.assertEqual(fake_db.statements, [])


if __name__ == "__main__":
    unittest.main()
