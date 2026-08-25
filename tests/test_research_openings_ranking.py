"""Research leads must outrank bookkeeping about the graph.

`value_score` measures how much a node lacks, not how good a question is. A
`benchmark_diversification` row scores 4.8 for having few `evaluated_on` links,
while an open question an author explicitly called unresolved scores 2.5. Of
the top 4,000 openings by value_score, every agent-related hit was a
completeness statistic, so the graph's 25,652 openings never produced a single
research direction and the directions were written by hand instead.
"""

import unittest
from unittest import mock

from db import opportunity_engine as opp


def _row(kind, confidence=0.7, value=2.5, title="t", description="d"):
    return {
        "node_id": "ml.agents.tool_use",
        "opportunity_type": kind,
        "title": title,
        "description": description,
        "why_now": "",
        "value_score": value,
        "confidence": confidence,
        "signal_counts": None,
        "evidence_paper_ids": None,
    }


class OpeningWeightTest(unittest.TestCase):
    def test_an_author_stated_question_outranks_a_completeness_statistic(self):
        self.assertGreater(
            opp.opening_research_weight("open_question"),
            opp.opening_research_weight("benchmark_diversification"))

    def test_a_contradiction_is_a_lead_not_a_statistic(self):
        self.assertGreaterEqual(opp.opening_research_weight("contradiction_resolution"), 90)

    def test_an_unknown_kind_lands_between_the_two(self):
        weight = opp.opening_research_weight("something_new")
        self.assertLess(weight, opp.opening_research_weight("open_question"))
        self.assertGreater(weight, opp.opening_research_weight("benchmark_diversification"))


class RankResearchOpeningsTest(unittest.TestCase):
    def _rank(self, rows, **kwargs):
        fake = mock.Mock()
        fake.fetchall.return_value = rows
        fake._load_json = lambda value, default: default
        with mock.patch.object(opp, "db", fake):
            return opp.rank_research_openings(**kwargs)

    def test_the_high_scoring_statistic_loses_to_the_low_scoring_question(self):
        rows = [_row("benchmark_diversification", confidence=0.79, value=4.8),
                _row("open_question", confidence=0.74, value=2.5)]
        ranked = self._rank(rows)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["opportunity_type"], "open_question")

    def test_completeness_statistics_can_be_asked_for_explicitly(self):
        rows = [_row("benchmark_diversification", value=4.8)]
        self.assertEqual(len(self._rank(rows, min_weight=0)), 1)

    def test_terms_narrow_to_an_area_over_title_and_description(self):
        rows = [_row("open_question", title="agent scaffolding", description="x"),
                _row("open_question", title="protein folding", description="y")]
        ranked = self._rank(rows, terms=["scaffold"])
        self.assertEqual(len(ranked), 1)
        self.assertIn("scaffold", ranked[0]["title"])

    def test_confidence_breaks_ties_within_one_kind(self):
        rows = [_row("open_question", confidence=0.5, title="low"),
                _row("open_question", confidence=0.9, title="high")]
        self.assertEqual(self._rank(rows)[0]["title"], "high")

    def test_the_limit_is_respected(self):
        rows = [_row("open_question", title=str(i)) for i in range(10)]
        self.assertEqual(len(self._rank(rows, limit=3)), 3)

    def test_every_row_reports_the_weight_it_was_ranked_by(self):
        ranked = self._rank([_row("open_question")])
        self.assertEqual(ranked[0]["research_weight"],
                         opp.opening_research_weight("open_question"))


if __name__ == "__main__":
    unittest.main()
