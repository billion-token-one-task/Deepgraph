"""parse_llm_json_text must not latch onto a prose-embedded fragment when a
larger complete object follows (four method-invention failures, 2026-08-17)."""

import unittest

from agents.llm_client import parse_llm_json_text


class BalancedObjectSelectionTests(unittest.TestCase):
    def test_prefers_the_largest_balanced_object(self):
        text = (
            'Complexity is roughly {"memory": 1, "time": 2} per step.\n'
            'Final answer:\n'
            '{"method": {"name": "X"}, "why_novel": "a novelty argument '
            'longer than thirty characters"}'
        )
        parsed, how = parse_llm_json_text(text)
        self.assertIn("method", parsed)
        self.assertTrue(how.startswith("balanced_object@"))

    def test_single_object_still_parses(self):
        parsed, how = parse_llm_json_text('prose {"a": 1} more prose')
        self.assertEqual(parsed, {"a": 1})

    def test_direct_json_unchanged(self):
        parsed, how = parse_llm_json_text('{"a": 1}')
        self.assertEqual((parsed, how), ({"a": 1}, "direct"))

    def test_latex_mixed_escapes_are_repaired(self):
        # idea 138's method definition mixed legal \\in with bare \Phi; one
        # illegal escape voided the whole object and only a two-key fragment
        # survived.
        text = (
            '{"method": {"name": "X", '
            '"definition": "$S = (V, E, \\Phi) \\\\in \\\\mathcal{S}$"}}'
        )
        parsed, how = parse_llm_json_text(text)
        self.assertEqual(parsed["method"]["name"], "X")

    def test_legal_escapes_survive_repair(self):
        parsed, _ = parse_llm_json_text('{"a": "line\\nbreak \\\\ slash"}')
        self.assertEqual(parsed, {"a": "line\nbreak \\ slash"})


if __name__ == "__main__":
    unittest.main()
