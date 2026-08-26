from __future__ import annotations

import unittest
from unittest import mock

from agents.agenda_repository import AgendaRepository


class RequeuedCandidateVisibilityTests(unittest.TestCase):
    """A withdrawn grant must not retire its candidate.

    requeue_withdrawn_candidate parks the job at awaiting_portfolio_decision
    with the stale grant pointer cleared, so a fresh grant can bind. The
    selector that issues that grant hid every candidate which had ever owned a
    job, so the recovery path handed the candidate to a queue nothing could
    read.
    """

    def _captured_sql(self):
        seen = {}

        def fetchall(sql, params=()):
            seen["sql"] = sql
            seen["params"] = params
            return []

        with mock.patch("agents.agenda_repository.db.fetchall", fetchall):
            AgendaRepository().candidates(16, limit=10)
        return seen["sql"]

    def test_work_in_flight_still_hides_its_candidate(self):
        sql = " ".join(self._captured_sql().split())
        self.assertIn("NOT EXISTS", sql)
        self.assertIn("auto_research_jobs", sql)

    def test_a_job_awaiting_a_decision_without_authority_does_not_hide_it(self):
        sql = " ".join(self._captured_sql().split())
        self.assertIn("arj.stage='awaiting_portfolio_decision'", sql)
        self.assertIn("arj.resource_grant_id IS NULL", sql)


if __name__ == "__main__":
    unittest.main()
