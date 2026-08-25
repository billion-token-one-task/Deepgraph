"""A research direction must reach the corpus, or not become a problem at all.

`_direction_evidence` matched an agenda's scope terms against taxonomy node ids
and names only. A taxonomy label is short and technical; an agenda's terms are
often research phrases -- "acceptance rule", "generalisation gap" -- that are
never substrings of one. Three agendas created on 2026-08-25 matched nothing,
their direction problems were persisted with `paper_ids=[]` anyway, and the
frontier gate refused them on every pass until their attempts ran out. Zero
tokens were spent and nothing said why.
"""

import unittest
from unittest import mock

from agents import problem_first as pf


class PapersMatchingTermsTest(unittest.TestCase):
    def _search(self, rows, terms):
        fake = mock.Mock()
        fake.fetchall.return_value = rows
        with mock.patch.object(pf, "db", fake):
            return pf._papers_matching_terms(terms)

    def test_a_research_phrase_reaches_the_corpus(self):
        rows = [{"id": "2605.08478"}, {"id": "2605.05716"}]
        self.assertEqual(self._search(rows, ["agentic reasoning"]),
                         ["2605.08478", "2605.05716"])

    def test_terms_too_short_to_be_specific_are_skipped(self):
        fake = mock.Mock()
        with mock.patch.object(pf, "db", fake):
            self.assertEqual(pf._papers_matching_terms(["ml", "rl"]), [])
        fake.fetchall.assert_not_called()

    def test_a_search_failure_does_not_break_discovery(self):
        fake = mock.Mock()
        fake.fetchall.side_effect = RuntimeError("index unavailable")
        with mock.patch.object(pf, "db", fake):
            self.assertEqual(pf._papers_matching_terms(["scaffold search"]), [])

    def test_duplicates_across_terms_collapse(self):
        rows = [{"id": "2605.08478"}, {"id": "2605.08478"}]
        self.assertEqual(self._search(rows, ["scaffold"]), ["2605.08478"])


class UnusableDirectionProblemTest(unittest.TestCase):
    def test_a_direction_problem_without_evidence_is_not_persisted(self):
        agenda_row = {"id": 17, "name": "a", "version": "v1"}
        with mock.patch.object(pf, "get_problem_signals", return_value=[]), \
             mock.patch.object(pf, "_require_agenda_id", side_effect=lambda x: x), \
             mock.patch.object(pf, "_direction_evidence", return_value=([], [])), \
             mock.patch.object(pf, "upsert_research_problem") as upsert, \
             mock.patch.object(pf, "db") as fake_db, \
             mock.patch.object(pf, "row_to_agenda") as to_agenda, \
             mock.patch.object(pf, "agenda_scope_terms", return_value=["acceptance rule"]):
            fake_db.table_exists.return_value = True
            fake_db.fetchone.return_value = agenda_row
            to_agenda.return_value = mock.Mock(description="d", name="n", version="v1")
            out = pf.discover_research_problems(limit=5, agenda_id=17, persist=True)
        self.assertEqual(out, [])
        upsert.assert_not_called()

    def test_a_direction_problem_with_evidence_is_kept(self):
        agenda_row = {"id": 17, "name": "a", "version": "v1"}
        with mock.patch.object(pf, "get_problem_signals", return_value=[]), \
             mock.patch.object(pf, "_require_agenda_id", side_effect=lambda x: x), \
             mock.patch.object(pf, "_direction_evidence",
                               return_value(["ml.agents"], ["2605.08478"])
                               if False else mock.DEFAULT) as evidence, \
             mock.patch.object(pf, "upsert_research_problem", return_value=1), \
             mock.patch.object(pf, "db") as fake_db, \
             mock.patch.object(pf, "row_to_agenda") as to_agenda, \
             mock.patch.object(pf, "agenda_scope_terms", return_value=["agentic reasoning"]):
            evidence.return_value = (["ml.agents"], ["2605.08478"])
            fake_db.table_exists.return_value = True
            fake_db.fetchone.return_value = agenda_row
            to_agenda.return_value = mock.Mock(description="d", name="n", version="v1")
            out = pf.discover_research_problems(limit=5, agenda_id=17, persist=True)
        self.assertEqual(len(out), 2)
        for problem in out:
            self.assertEqual(problem["paper_ids"], ["2605.08478"])


if __name__ == "__main__":
    unittest.main()
