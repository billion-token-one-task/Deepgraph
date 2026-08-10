"""A paid proposal grant must be able to produce an idea.

Two defects in the ideation ring made that impossible, and together they kept
every agenda at zero generated candidates while grants were being issued and
spent.

1. The pre-idea identity row is seeded from the research problem, so it carries
   the problem's ``source_node_ids`` and ``mechanism_type``. When the LLM
   realized the idea under the grant, the duplicate check compared the result
   against that placeholder - itself - scored node_overlap 1.0, and discarded
   the idea. Because the LLM reservations were already settled, the retry then
   failed on the burned idempotency key, so the candidate could never be
   realized and the grant never settled. Observed on agenda 11 / idea 128:
   10278 tokens spent, nothing produced.

2. A candidate left at ``status='proposal_pending'`` with a terminal outcome
   owns the partial unique key on (agenda_id, research_problem_id) without
   being reusable. The insert collided with it, the recovery lookup filtered on
   ``outcome='pending'`` and missed it, and the resulting RuntimeError escaped
   ``discover_paper_ideas`` and killed the whole Tier-2 pass - so one stale
   candidate silently disabled ideation for its entire agenda.
"""

from __future__ import annotations

import unittest
from unittest import mock

from agents import paper_idea_agent
from agents.paper_idea_agent import (
    ProposalProblemUnavailable,
    _find_existing_tier2_duplicate,
    _proposal_candidate_and_grant,
)


PLACEHOLDER = {
    "id": 128,
    "title": "Test the missing mechanism behind strong claims in ml.theory",
    "source_node_ids": '["node-a", "node-b", "node-c"]',
    "mechanism_type": "mechanism_mismatch",
    "status": "proposal_pending",
    "novelty_status": "unchecked",
    "outcome": "pending",
}

REALIZED = {
    "title": "Non-Gaussian Spectral Cumulant Alignment",
    "source_node_ids": '["node-a", "node-b", "node-c"]',
    "mechanism_type": "mechanism_mismatch",
}


class DuplicateGateTests(unittest.TestCase):
    def test_candidate_is_not_a_duplicate_of_its_own_placeholder(self):
        with mock.patch.object(paper_idea_agent.db, "fetchall", return_value=[PLACEHOLDER]):
            self.assertIsNone(
                _find_existing_tier2_duplicate(REALIZED, exclude_id=128),
                "the realized idea was rejected as a duplicate of its own "
                "pre-idea identity row",
            )

    def test_a_genuine_duplicate_is_still_rejected(self):
        """The exclusion must be exactly one row, not a disabled gate."""

        other = dict(PLACEHOLDER, id=99)
        with mock.patch.object(paper_idea_agent.db, "fetchall", return_value=[other]):
            duplicate = _find_existing_tier2_duplicate(REALIZED, exclude_id=128)
        self.assertIsNotNone(duplicate)
        self.assertEqual(duplicate["id"], 99)

    def test_gate_is_unchanged_when_no_placeholder_is_excluded(self):
        with mock.patch.object(paper_idea_agent.db, "fetchall", return_value=[PLACEHOLDER]):
            duplicate = _find_existing_tier2_duplicate(REALIZED)
        self.assertIsNotNone(duplicate)
        self.assertEqual(duplicate["id"], 128)


class ProposalIdentityTests(unittest.TestCase):
    problem = {"research_problem_id": 9, "title": "p", "problem_statement": "s"}

    def _run(self, *, existing, insert_result, holder):
        """Drive _proposal_candidate_and_grant through its conflict branch."""

        calls = {"n": 0}

        def fetchone(sql, params=()):
            calls["n"] += 1
            text = " ".join(str(sql).split())
            if "INSERT INTO deep_insights" in text:
                return insert_result
            if "status='proposal_pending'" in text and "SELECT id, outcome" in text:
                return holder
            if "FROM deep_insights" in text:
                return existing
            if "FROM resource_grants" in text:
                return {"id": 21, "token_cap": 32000}
            return None

        with mock.patch.object(paper_idea_agent.db, "fetchone", side_effect=fetchone), \
                mock.patch.object(paper_idea_agent.db, "commit"), \
                mock.patch.object(paper_idea_agent.db, "rollback"):
            return _proposal_candidate_and_grant(agenda_id=11, problem=self.problem)

    def test_spent_key_holder_skips_the_problem_instead_of_killing_the_pass(self):
        with self.assertRaises(ProposalProblemUnavailable) as caught:
            self._run(
                existing=None,
                insert_result=None,
                holder={"id": 110, "outcome": "experiment_failed_setup"},
            )
        self.assertIn("110", str(caught.exception))

    def test_a_concurrent_pending_holder_is_reused(self):
        candidate_id, grant = self._run(
            existing=None,
            insert_result=None,
            holder={"id": 131, "outcome": "pending"},
        )
        self.assertEqual(candidate_id, 131)
        self.assertEqual(grant["id"], 21)

    def test_a_real_race_with_no_holder_still_raises(self):
        with self.assertRaises(RuntimeError) as caught:
            self._run(existing=None, insert_result=None, holder=None)
        self.assertIn("identity race", str(caught.exception))
        self.assertNotIsInstance(caught.exception, ProposalProblemUnavailable)


class DiscoveryResilienceTests(unittest.TestCase):
    def test_unavailable_problem_does_not_abort_the_remaining_problems(self):
        """One spent problem must cost one problem, not the whole pass."""

        import inspect

        source = inspect.getsource(paper_idea_agent.discover_paper_ideas)
        self.assertIn("except ProposalProblemUnavailable", source)
        skip = source.index("except ProposalProblemUnavailable")
        self.assertIn("continue", source[skip:skip + 300])

class SiblingProvenanceTests(unittest.TestCase):
    """Provenance shared by construction is not evidence of duplication.

    Ideas raised from the same research problem inherit that problem's
    source_node_ids verbatim, so their node overlap is 1.0 whatever the ideas
    say. Scoring it meant a research problem could yield exactly one idea, ever:
    agenda 11's second idea on problem 8 was rejected as a duplicate of its
    first at node_overlap 1.0, title_sim 0.095.
    """

    SIBLING = {
        "id": 105,
        "title": "Test the missing mechanism behind strong claims in ml.theory",
        "source_node_ids": '["ml.theory.interpretability"]',
        "mechanism_type": "mechanism_mismatch",
        "research_problem_id": 8,
    }

    def _candidate(self, title="Polyadic Cumulant Spectral Alignment"):
        return {
            "title": title,
            "source_node_ids": '["ml.theory.interpretability"]',
            "mechanism_type": "mechanism_mismatch",
            "research_problem_id": 8,
        }

    def test_a_second_idea_on_the_same_problem_is_allowed(self):
        with mock.patch.object(paper_idea_agent.db, "fetchall", return_value=[self.SIBLING]):
            self.assertIsNone(_find_existing_tier2_duplicate(self._candidate()))

    def test_a_genuinely_similar_title_is_still_caught_between_siblings(self):
        """Only provenance is discounted; content signals still apply."""

        with mock.patch.object(paper_idea_agent.db, "fetchall", return_value=[self.SIBLING]):
            duplicate = _find_existing_tier2_duplicate(
                self._candidate(title=self.SIBLING["title"])
            )
        self.assertIsNotNone(duplicate)
        self.assertEqual(duplicate["id"], 105)

    def test_node_overlap_still_counts_across_different_problems(self):
        other = dict(self.SIBLING, id=27, research_problem_id=41)
        with mock.patch.object(paper_idea_agent.db, "fetchall", return_value=[other]):
            duplicate = _find_existing_tier2_duplicate(self._candidate())
        self.assertIsNotNone(duplicate, "cross-problem duplicate detection was weakened")
        self.assertEqual(duplicate["id"], 27)

if __name__ == "__main__":
    unittest.main()
