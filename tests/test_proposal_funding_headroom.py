"""Proposal generation that keeps delivering nothing must stop being funded.

Separating attempt identity from operation identity stopped one unusable
provider response from retiring a candidate for good. But the attempt bound
that came with it counts per grant, and when a grant expired the candidate was
requeued and handed an identical new one, so the bound reset and the cycle
started over.

Measured 2026-08-10: grants 20-28 consumed 205393 tokens across nine grants for
four candidates and produced no idea at all. A count bounds attempts; the thing
that needed bounding was money, so the ceiling is a share of the agenda's own
budget instead.
"""

from __future__ import annotations

import unittest
from unittest import mock

from contracts.meta_harness import ResourceGrant
from meta_harness import repository
from meta_harness.repository import (
    UNDELIVERED_PROPOSAL_BUDGET_SHARE,
    MetaHarnessPersistenceError,
    _require_proposal_funding_headroom,
)


def _grant(stage="proposal", agenda_id=10, idea_id=125):
    return ResourceGrant(
        agenda_id=agenda_id,
        idea_id=idea_id,
        decision_packet_id=1,
        stage=stage,
        token_cap=32000,
        gpu_class="none",
        max_gpu_hours=0.0,
        backend_allowlist=["llm"],
        artifact_requirements=["candidate_design"],
        expires_at="2026-08-10T23:00:00+00:00",
        grant_reason="test",
        idempotency_key="k",
    )


class ProposalFundingHeadroomTests(unittest.TestCase):
    BUDGET = 500_000

    def _run(self, undelivered, *, stage="proposal", budget=None):
        with mock.patch.object(
            repository, "_undelivered_proposal_spend", return_value=undelivered
        ):
            _require_proposal_funding_headroom(
                _grant(stage=stage), self.BUDGET if budget is None else budget
            )

    def test_a_first_funding_is_allowed(self):
        self._run(0)

    def test_funding_continues_below_the_ceiling(self):
        self._run(int(self.BUDGET * UNDELIVERED_PROPOSAL_BUDGET_SHARE) - 1)

    def test_funding_stops_at_the_ceiling(self):
        ceiling = int(self.BUDGET * UNDELIVERED_PROPOSAL_BUDGET_SHARE)
        with self.assertRaises(MetaHarnessPersistenceError) as caught:
            self._run(ceiling)
        message = str(caught.exception)
        self.assertIn("without delivering a candidate", message)
        self.assertIn(str(ceiling), message)

    def test_the_observed_burn_would_have_been_refused(self):
        """205393 tokens across nine grants must not have been reachable."""

        with self.assertRaises(MetaHarnessPersistenceError):
            self._run(205_393)

    def test_compute_grants_are_untouched(self):
        """The ceiling is about proposal generation, not experiments."""

        self._run(10**9, stage="pilot")

    def test_a_zero_budget_agenda_is_left_to_the_hard_cap(self):
        self._run(10**9, budget=0)


class UndeliveredSpendQueryTests(unittest.TestCase):
    def test_only_counts_proposal_grants_that_never_delivered(self):
        captured = {}

        def fetchone(sql, params=()):
            captured["sql"] = " ".join(str(sql).split())
            captured["params"] = params
            return {"spent": 1234}

        with mock.patch.object(repository.db, "fetchone", side_effect=fetchone):
            spent = repository._undelivered_proposal_spend(10, 125)

        self.assertEqual(spent, 1234)
        sql = captured["sql"]
        # A realized proposal settles its grant to 'consumed'; that spend bought
        # something and must not count against the ceiling.
        self.assertIn("g.status <> 'consumed'", sql)
        self.assertIn("g.stage='proposal'", sql)
        self.assertIn("u.status='settled'", sql)
        self.assertEqual(captured["params"], (10, 125))


class RetirementWiringTests(unittest.TestCase):
    """A refused candidate must leave the queue, not be refused every pass."""

    def test_advancer_retires_on_this_refusal(self):
        import inspect

        from scripts import auto_advance

        source = inspect.getsource(auto_advance.advance_agenda)
        self.assertIn("without delivering a candidate", source)
        self.assertIn("OUTCOME_PROPOSAL_UNREALIZED", source)
        self.assertIn("proposal_candidate_retired", source)

    def test_the_retirement_outcome_is_declared(self):
        from db.insight_outcomes import ALL_OUTCOMES, OUTCOME_PROPOSAL_UNREALIZED

        self.assertIn(OUTCOME_PROPOSAL_UNREALIZED, ALL_OUTCOMES)

    def test_a_retired_candidate_stops_holding_its_research_problem(self):
        """Retirement has to be visible to the pre-idea identity lookup, which
        selects only candidates whose outcome is still pending."""

        from db.insight_outcomes import OUTCOME_PROPOSAL_UNREALIZED

        self.assertNotEqual(OUTCOME_PROPOSAL_UNREALIZED, "pending")


if __name__ == "__main__":
    unittest.main()
