"""A direction-derived problem must carry evidence or it can never be used.

When no harvested signal matches an agenda's scope, discover_research_problems
falls back to two problems written from the agenda's direction. Those were
persisted with node_ids=[] and paper_ids=[], and
EvidenceGraphFrontierSource._papers refuses a problem with no linked papers, so
each one failed the frontier gate forever while still consuming a per-problem
ration. On 2026-08-17 fourteen problems -- two on each of seven agendas -- sat
in exactly that state, and six of them were the only thing standing between
three agendas and a frontier packet.
"""

import unittest
from unittest import mock

from agents import problem_first
from contracts.agenda import ResearchAgenda


def _agenda(**overrides) -> ResearchAgenda:
    fields = {
        "name": "speculative-decoding",
        "agenda_id": 3,
        "focus": ["speculative decoding", "draft model"],
        "token_budget": 1000,
        "max_concurrency": 1,
        "backend_allowlist": ["llm"],
    }
    fields.update(overrides)
    return ResearchAgenda(**fields)


class DirectionEvidenceTests(unittest.TestCase):
    def test_scope_terms_select_nodes_and_their_papers(self):
        nodes = [
            {"id": "ml.dl.nlp.speculative_decoding", "name": "Speculative Decoding"},
            {"id": "ml.cv.segmentation", "name": "Segmentation"},
        ]
        papers = {
            "ml.dl.nlp.speculative_decoding": [{"id": "2601.001"}, {"id": "2601.002"}],
        }
        with mock.patch.dict("sys.modules"):
            with mock.patch("db.taxonomy.get_taxonomy_flat", return_value=nodes), \
                 mock.patch("db.taxonomy.get_node_papers",
                            side_effect=lambda node_id, limit=50: papers.get(node_id, [])):
                node_ids, paper_ids = problem_first._direction_evidence(_agenda())
        self.assertEqual(node_ids, ["ml.dl.nlp.speculative_decoding"])
        self.assertEqual(paper_ids, ["2601.001", "2601.002"])

    def test_an_unrelated_corpus_yields_nothing_rather_than_a_wrong_link(self):
        nodes = [{"id": "ml.cv.segmentation", "name": "Segmentation"}]
        with mock.patch("db.taxonomy.get_taxonomy_flat", return_value=nodes), \
             mock.patch("db.taxonomy.get_node_papers", return_value=[]):
            node_ids, paper_ids = problem_first._direction_evidence(_agenda())
        self.assertEqual(node_ids, [])
        self.assertEqual(paper_ids, [])

    def test_short_scope_terms_do_not_match_everything(self):
        """A two-letter term would otherwise select the whole taxonomy."""
        nodes = [{"id": "ml.cv.segmentation", "name": "Segmentation"}]
        with mock.patch("db.taxonomy.get_taxonomy_flat", return_value=nodes), \
             mock.patch("db.taxonomy.get_node_papers", return_value=[{"id": "x"}]):
            node_ids, _ = problem_first._direction_evidence(_agenda(focus=["ml"]))
        self.assertEqual(node_ids, [])

    def test_paper_ids_are_deduped_and_capped(self):
        nodes = [
            {"id": f"ml.a.speculative decoding.{i}", "name": "Speculative Decoding"}
            for i in range(6)
        ]
        with mock.patch("db.taxonomy.get_taxonomy_flat", return_value=nodes), \
             mock.patch("db.taxonomy.get_node_papers",
                        return_value=[{"id": "dupe"}] * 100):
            node_ids, paper_ids = problem_first._direction_evidence(_agenda())
        self.assertLessEqual(len(node_ids), problem_first.DIRECTION_EVIDENCE_NODE_LIMIT)
        self.assertEqual(paper_ids, ["dupe"])

    def test_a_broken_taxonomy_does_not_break_discovery(self):
        with mock.patch("db.taxonomy.get_taxonomy_flat", side_effect=RuntimeError("boom")):
            self.assertEqual(problem_first._direction_evidence(_agenda()), ([], []))

    def test_a_problem_without_papers_scores_zero(self):
        """Why the empty link was fatal rather than merely weak."""
        self.assertEqual(problem_first.problem_quality_score({"paper_ids": []}), 0.0)


if __name__ == "__main__":
    unittest.main()
