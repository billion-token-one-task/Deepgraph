import unittest

from agents.insight_validation import (
    INSIGHT_INPUT_MISSING_ERROR_CODE,
    get_evosci_input_issue,
)


class EvoSciInputIssueTests(unittest.TestCase):
    def test_tier1_missing_core_fields_returns_structured_error(self):
        insight = {
            "id": 12,
            "tier": 1,
            "title": "Mechanism-first insight",
            "field_a": "{}",
            "field_b": "{}",
            "formal_structure": "",
            "transformation": "",
        }

        issue = get_evosci_input_issue(insight, mode="verification")

        self.assertIsNotNone(issue)
        self.assertEqual(issue["error_code"], INSIGHT_INPUT_MISSING_ERROR_CODE)
        self.assertCountEqual(
            issue["missing_fields"],
            ["Field A", "Field B", "formal structure", "transformation"],
        )

    def test_tier2_minimal_complete_payload_is_accepted(self):
        insight = {
            "id": 13,
            "tier": 2,
            "title": "Counterfactual Gain Gated Reasoning",
            "problem_statement": "Learn when additional reasoning budget is useful.",
            "proposed_method": (
                '{"name":"CGGR","definition":"Estimate the counterfactual gain of extra reasoning."}'
            ),
        }

        issue = get_evosci_input_issue(insight, mode="verification")

        self.assertIsNone(issue)


if __name__ == "__main__":
    unittest.main()
