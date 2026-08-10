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
    def _spend(self, rows):
        """Run the query with ``rows`` as successive fetchone results."""

        calls = []

        def fetchone(sql, params=()):
            calls.append({"sql": " ".join(str(sql).split()), "params": params})
            return rows[len(calls) - 1]

        with mock.patch.object(repository.db, "fetchone", side_effect=fetchone):
            spent = repository._undelivered_proposal_spend(10, 125)
        return spent, calls

    def test_only_counts_proposal_grants_that_never_delivered(self):
        spent, calls = self._spend([{"research_problem_id": 49}, {"spent": 1234}])

        self.assertEqual(spent, 1234)
        sql = calls[-1]["sql"]
        # A realized proposal settles its grant to 'consumed'; that spend bought
        # something and must not count against the ceiling.
        self.assertIn("g.status <> 'consumed'", sql)
        self.assertIn("g.stage='proposal'", sql)
        self.assertIn("u.status='settled'", sql)

    def test_the_bill_follows_the_research_problem_not_the_row(self):
        """Archiving a spent candidate must not hand its problem a fresh budget.

        Retirement frees the problem to be seeded again under a new row id. If
        the ceiling counted per idea_id, that new row would start at zero and
        the burn the ceiling exists to stop would simply repeat, one row id at
        a time.
        """

        spent, calls = self._spend([{"research_problem_id": 49}, {"spent": 58590}])

        self.assertEqual(spent, 58590)
        self.assertIn("d.research_problem_id=?", calls[-1]["sql"])
        self.assertEqual(calls[-1]["params"], (10, 49))
        self.assertNotIn("g.idea_id=?", calls[-1]["sql"])

    def test_a_candidate_with_no_problem_still_bills_itself(self):
        spent, calls = self._spend([{"research_problem_id": None}, {"spent": 77}])

        self.assertEqual(spent, 77)
        self.assertIn("g.idea_id=?", calls[-1]["sql"])
        self.assertEqual(calls[-1]["params"], (10, 125))


class ProblemOverBudgetTests(unittest.TestCase):
    """The seeding path and the grant gate must agree on one rule."""

    def _over(self, *, budget, spent, problem_id=49):
        rows = [{"token_budget": budget}, {"spent": spent}]
        calls = []

        def fetchone(sql, params=()):
            calls.append(params)
            return rows[len(calls) - 1]

        with mock.patch.object(repository.db, "fetchone", side_effect=fetchone):
            return repository.proposal_problem_is_over_budget(10, problem_id)

    def test_a_problem_that_burned_its_share_is_over_budget(self):
        # The real numbers: problem 49 burned 58590 against a 50000 ceiling.
        self.assertTrue(self._over(budget=500_000, spent=58_590))

    def test_a_problem_with_headroom_is_not(self):
        self.assertFalse(self._over(budget=500_000, spent=43_847))

    def test_no_problem_id_is_never_over_budget(self):
        with mock.patch.object(repository.db, "fetchone", side_effect=AssertionError):
            self.assertFalse(repository.proposal_problem_is_over_budget(10, 0))

    def test_a_zero_budget_agenda_is_left_to_the_hard_cap(self):
        self.assertFalse(self._over(budget=0, spent=10**9))

    def test_it_uses_the_same_share_as_the_grant_gate(self):
        budget = 500_000
        ceiling = int(budget * UNDELIVERED_PROPOSAL_BUDGET_SHARE)
        self.assertFalse(self._over(budget=budget, spent=ceiling - 1))
        self.assertTrue(self._over(budget=budget, spent=ceiling))


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
        """Retirement must move status, not just outcome.

        This test used to assert only that the outcome enum differs from
        'pending', which it does -- and production was broken the whole time it
        passed. Selection keys on status, so writing the outcome alone left
        idea 125 to be promoted, refused and "retired" once per pass (observed
        2026-08-10 at 21:10:45, 21:22:17 and 21:51:24), while the row went on
        owning idx_deep_insights_pending_proposal and research problem 49 could
        never be worked again.
        """

        import inspect

        from scripts import auto_advance

        source = inspect.getsource(auto_advance.advance_agenda)
        retire = source.split("without delivering a candidate", 1)[1]
        retire = retire.split("continue", 1)[0]

        statement = " ".join(retire.split())
        self.assertIn("UPDATE deep_insights SET status='archived'", statement)
        # The unique index is partial on status='proposal_pending'; leaving the
        # row in that status is what held the problem.
        self.assertIn("status='proposal_pending'", statement)
        # Agenda-owned tables are only ever mutated with agenda_id in the WHERE.
        self.assertIn("WHERE id=? AND agenda_id=?", statement)

    def test_archived_is_a_status_the_candidate_queries_already_exclude(self):
        """'archived' must be read, not merely written.

        A retirement status nobody selects on would be one more write-only
        state -- the defect class this repository has hit four times.
        """

        import pathlib

        root = pathlib.Path(repository.__file__).resolve().parent.parent
        readers = [
            root / "agents" / "paper_idea_agent.py",
            root / "agents" / "agenda_repository.py",
            root / "scripts" / "auto_advance.py",
        ]
        for path in readers:
            text = path.read_text()
            self.assertIn(
                "'archived'",
                text,
                f"{path.name} no longer excludes archived candidates",
            )


if __name__ == "__main__":
    unittest.main()
