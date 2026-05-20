import unittest
from unittest import mock

from agents.insight_validation import (
    INSIGHT_INPUT_MISSING_ERROR_CODE,
    get_evosci_input_issue,
)
from agents import novelty_verifier


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


class EvoSciLaunchReuseTests(unittest.TestCase):
    def test_launch_full_research_reuses_active_session(self):
        insight = {
            "id": 21,
            "tier": 2,
            "title": "Reusable research session",
            "problem_statement": "Test whether deep research reuses an active session.",
            "proposed_method": '{"name":"ReuseNet","definition":"Reuse active EvoSci sessions."}',
            "evoscientist_workdir": "/tmp/deep-research-21",
        }

        with (
            mock.patch.object(novelty_verifier.db, "fetchone", return_value=insight),
            mock.patch.object(
                novelty_verifier,
                "active_research_session",
                return_value={"workdir": "/tmp/deep-research-21", "pid": 4242, "active": True},
            ),
            mock.patch.object(
                novelty_verifier,
                "get_research_status",
                return_value={"workdir": "/tmp/deep-research-21", "active": True},
            ),
        ):
            result = novelty_verifier.launch_full_research(21)

        self.assertEqual(result["status"], "running")
        self.assertTrue(result["reused"])
        self.assertEqual(result["pid"], 4242)
        self.assertEqual(result["workdir"], "/tmp/deep-research-21")


if __name__ == "__main__":
    unittest.main()
