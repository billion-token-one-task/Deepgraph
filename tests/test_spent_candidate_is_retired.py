"""A spent proposal candidate must be retired, not merely reported.

deep_insights carries a unique key on (agenda_id, research_problem_id) for
rows at status='proposal_pending'. A candidate that reached a terminal outcome
but was never moved off that status keeps the key while being unusable, so its
research problem can never seed another candidate.

Nothing performed that transition. The terminal outcome was written and no
code read it -- the same write-only shape that has produced most of this
system's stalls. Idea 110 held research problem 9 for a full day
(outcome=experiment_failed_run) while the portfolio kept issuing it fresh
proposal grants, and because a funded proposal preempts discovery rotation,
one dead candidate starved every other agenda. Clearing it by hand took five
operator grant expiries in nine hours (grants 142, 146, 147, 169, 175) and it
re-pinned within forty minutes each time.

Discovery already detected the condition precisely. It now acts on it: the
placeholder is archived and its job closed, which frees the key and releases
the preemption slot. The experiment run and its evidence are untouched --
only the pre-idea placeholder is retired.
"""

import inspect
import unittest

from agents import paper_idea_agent


class SpentCandidateRetirementTests(unittest.TestCase):
    def _source(self):
        return inspect.getsource(paper_idea_agent)

    def test_the_spent_holder_is_archived(self):
        source = self._source()
        self.assertIn("UPDATE deep_insights SET status='archived'", source)

    def test_archiving_is_guarded_on_the_pending_status(self):
        # never archive a candidate that has moved on under us
        source = self._source()
        self.assertIn("WHERE id=? AND status='proposal_pending'", source)

    def test_the_job_is_closed_so_it_stops_preempting_discovery(self):
        source = self._source()
        self.assertIn("stage='proposal_unrealized'", source)
        self.assertIn("WHERE deep_insight_id=? AND status='deferred'", source)

    def test_the_pass_still_reports_the_problem_as_unavailable(self):
        # this pass cannot use the problem; the next one can
        source = self._source()
        self.assertIn("the problem is free for the next pass", source)

    def test_retirement_failure_never_aborts_the_discovery_pass(self):
        source = self._source()
        window = source.split("spent_id = int(holder")[1][:1400]
        self.assertIn("except Exception:", window)
        self.assertIn("db.rollback()", window)

    def test_only_the_placeholder_is_touched(self):
        # evidence lives in experiment_runs; retirement must not reach it
        source = self._source()
        window = source.split("spent_id = int(holder")[1][:1400]
        self.assertNotIn("experiment_runs", window)


if __name__ == "__main__":
    unittest.main()
