import unittest

from agents.agenda_relevance import candidate_scope_text, insight_in_scope
from contracts.agenda import ResearchAgenda


class AgendaRelevanceTests(unittest.TestCase):
    def test_experimental_plan_contributes_to_scope_gate(self):
        candidate = {
            "agenda_id": 11,
            "title": "A generic candidate",
            "problem_statement": "A generic problem",
            "experimental_plan": {"benchmark": "mathematical reasoning"},
        }
        agenda = ResearchAgenda(
            agenda_id=11,
            name="test-time compute",
            description="Inference scaling",
            focus=["mathematical reasoning"],
            prefer={"keywords": []},
        )

        self.assertIn("mathematical reasoning", candidate_scope_text(candidate))
        self.assertTrue(insight_in_scope(candidate, agenda))


if __name__ == "__main__":
    unittest.main()
