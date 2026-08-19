"""Delivering a proposal must earn a problem a clean slate.

The undelivered-proposal ceiling stops the system pouring tokens into a
research problem that never produces a candidate. It measured GROSS LIFETIME
spend: the figure never decays, and delivering a realized proposal never
credited it. Under that rule every problem crosses the ceiling eventually, so
the candidate funnel is guaranteed to dry up given enough time -- which is what
it did on 2026-08-19, leaving the M2 acceptance window unable to forge a single
new run.

Problem 49 had delivered SIX realized proposals, the most recent five hours
earlier, and was locked out by 264,151 tokens of historical wastage against a
250,000 ceiling. Only 32,000 had been spent since that last delivery.

The rule now measures from the problem's most recent delivery. A problem that
has never delivered still accumulates from the beginning and is still stopped:
problems 8 and 28 (76,867 and 71,857 against a 50,000 ceiling, no delivery
ever) stayed blocked when this shipped.

A second defect sat next to the first. proposal_problem_is_over_budget carried
its own copy of the arithmetic, under a docstring promising "One rule, two
callers -- a second copy of the arithmetic would drift". It had drifted: the
spend helper answered 32,000 while the predicate still answered 264,151 and
kept the problem blocked. Both now call one function.
"""

import inspect
import unittest

from meta_harness import repository


class UndeliveredProposalWindowTests(unittest.TestCase):
    def test_both_callers_share_one_query(self):
        for func in (
            repository._undelivered_proposal_spend,
            repository.proposal_problem_is_over_budget,
        ):
            self.assertIn(
                "_undelivered_proposal_spend_for_problem", inspect.getsource(func)
            )
        # the predicate must not carry its own copy again -- that copy is what
        # drifted and kept problem 49 blocked after the window was fixed
        self.assertNotIn(
            "SUM(u.tokens_used)",
            inspect.getsource(repository.proposal_problem_is_over_budget),
        )

    def test_the_no_problem_fallback_is_a_known_asymmetry(self):
        # A candidate with no research_problem_id still sums its own lifetime
        # spend, idea-scoped, with no delivery window. Left as-is deliberately:
        # every candidate on the agendas in play carries a problem id, so
        # changing it would be an untested edit to an unused path. Pinned here
        # so the asymmetry is recorded rather than forgotten.
        source = inspect.getsource(repository._undelivered_proposal_spend)
        self.assertIn("g.idea_id=?", source)

    def test_the_window_starts_at_the_last_delivered_proposal(self):
        source = inspect.getsource(
            repository._undelivered_proposal_spend_for_problem
        )
        self.assertIn("dg.status = 'consumed'", source)
        self.assertIn("MAX(dg.created_at)", source)
        self.assertIn("g.created_at >", source)

    def test_a_problem_that_never_delivered_accumulates_from_the_start(self):
        # COALESCE to the epoch: with no delivered grant the window is all time
        source = inspect.getsource(
            repository._undelivered_proposal_spend_for_problem
        )
        self.assertIn("1970-01-01", source)

    def test_the_ceiling_itself_is_unchanged(self):
        # the fix must not quietly raise the ceiling as well
        self.assertEqual(repository._undelivered_proposal_ceiling(500_000), 50_000)
        self.assertEqual(
            repository._undelivered_proposal_ceiling(1_600_000_000), 250_000
        )


if __name__ == "__main__":
    unittest.main()
