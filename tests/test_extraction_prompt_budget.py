"""Extraction prompt budget: taxonomy routing, per-role listing, visible caps.

The taxonomy has ~3.9k leaf nodes whose dotted IDs average ~120 characters. The
previous prompt sent that entire list to all five extraction sub-agents, so the
classification vocabulary cost several times more than the paper it classified,
and 92% of papers were truncated to fit alongside it.
"""

import unittest
from unittest import mock

from agents import extraction_agent
from agents import multi_agent_extraction


class TaxonomyHintBudgetTests(unittest.TestCase):
    def test_short_list_is_kept_whole_and_reports_no_truncation(self):
        leaves = ["ml.dl.cv.detection", "ml.dl.nlp.qa", "ml.theory.pac"]
        hint, truncated = extraction_agent.format_taxonomy_hint(leaves)

        self.assertFalse(truncated)
        for leaf in leaves:
            self.assertIn(leaf, hint)

    def test_budget_overrun_truncates_and_says_so(self):
        leaves = [f"ml.dl.branch{i:04d}.{'x' * 90}" for i in range(400)]
        with mock.patch.object(
            extraction_agent, "EXTRACTION_TAXONOMY_LEAF_BUDGET_CHARS", 2_000
        ):
            hint, truncated = extraction_agent.format_taxonomy_hint(leaves)

        self.assertTrue(truncated, "a dropped taxonomy must never look complete")
        self.assertLess(len(hint), 4_000)

    def test_budget_keeps_at_least_one_entry(self):
        leaves = ["ml." + "y" * 500]
        with mock.patch.object(
            extraction_agent, "EXTRACTION_TAXONOMY_LEAF_BUDGET_CHARS", 10
        ):
            hint, truncated = extraction_agent.format_taxonomy_hint(leaves)

        self.assertIn(leaves[0], hint)
        self.assertFalse(truncated)

    def test_duplicates_do_not_count_as_truncation(self):
        leaves = ["ml.dl.cv.detection"] * 5
        hint, truncated = extraction_agent.format_taxonomy_hint(leaves)

        self.assertFalse(truncated)
        self.assertEqual(hint.count("ml.dl.cv.detection"), 1)


class TaxonomyRoutingTests(unittest.TestCase):
    COARSE = [
        {"id": "ml.dl.cv", "name": "Computer Vision", "depth": 2},
        {"id": "ml.dl.nlp", "name": "Natural Language Processing", "depth": 2},
    ]

    def test_routing_sends_only_the_selected_branch(self):
        with mock.patch.object(
            extraction_agent, "get_nodes_to_depth", return_value=self.COARSE
        ), mock.patch.object(
            extraction_agent, "get_leaf_ids_under", return_value=["ml.dl.cv.detection"]
        ) as under, mock.patch.object(
            extraction_agent, "get_all_leaf_ids", return_value=["never.used"]
        ) as full, mock.patch.object(
            extraction_agent,
            "proposer_json",
            return_value=({"areas": ["ml.dl.cv"]}, 120, {}),
        ):
            hint, meta = extraction_agent.resolve_taxonomy_hint(
                "2605.1", "A Vision Paper", "we detect objects", None
            )

        under.assert_called_once_with(["ml.dl.cv"])
        full.assert_not_called()
        self.assertIn("ml.dl.cv.detection", hint)
        self.assertEqual(meta["taxonomy_routing"], "routed")
        self.assertEqual(meta["taxonomy_leaf_count"], 1)
        self.assertEqual(meta["taxonomy_routing_tokens"], 120)

    def test_hallucinated_area_falls_back_to_the_full_list(self):
        with mock.patch.object(
            extraction_agent, "get_nodes_to_depth", return_value=self.COARSE
        ), mock.patch.object(
            extraction_agent, "get_leaf_ids_under", return_value=[]
        ), mock.patch.object(
            extraction_agent, "get_all_leaf_ids", return_value=["ml.dl.cv.detection"]
        ), mock.patch.object(
            extraction_agent,
            "proposer_json",
            return_value=({"areas": ["not.a.real.area"]}, 90, {}),
        ):
            hint, meta = extraction_agent.resolve_taxonomy_hint(
                "2605.2", "Off Taxonomy", "text", None
            )

        self.assertEqual(meta["taxonomy_routing"], "fallback_full")
        self.assertIn("ml.dl.cv.detection", hint)

    def test_routing_failure_costs_tokens_not_the_paper(self):
        with mock.patch.object(
            extraction_agent, "get_nodes_to_depth", return_value=self.COARSE
        ), mock.patch.object(
            extraction_agent, "get_all_leaf_ids", return_value=["ml.dl.cv.detection"]
        ), mock.patch.object(
            extraction_agent, "proposer_json", side_effect=RuntimeError("gateway 500")
        ):
            hint, meta = extraction_agent.resolve_taxonomy_hint(
                "2605.3", "Broken Route", "text", None
            )

        self.assertEqual(meta["taxonomy_routing"], "error")
        self.assertIn("gateway 500", meta["taxonomy_routing_error"])
        self.assertIn("ml.dl.cv.detection", hint)

    def test_routing_can_be_disabled(self):
        with mock.patch.object(
            extraction_agent, "EXTRACTION_TAXONOMY_ROUTING_ENABLED", False
        ), mock.patch.object(
            extraction_agent, "get_all_leaf_ids", return_value=["ml.dl.cv.detection"]
        ), mock.patch.object(
            extraction_agent, "proposer_json"
        ) as called:
            _hint, meta = extraction_agent.resolve_taxonomy_hint(
                "2605.4", "No Routing", "text", None
            )

        called.assert_not_called()
        self.assertEqual(meta["taxonomy_routing"], "disabled")


class PerRolePromptTests(unittest.TestCase):
    def test_only_the_taxonomy_reader_is_charged_for_the_listing(self):
        hint = "Available taxonomy leaf nodes:\n  ml.dl.cv.detection"
        seen = {}

        def fake_proposer(system_prompt, user_prompt, *, llm_scope, operation):
            seen[operation.rsplit(":", 1)[-1]] = user_prompt
            return {}, 10, {}

        with mock.patch.object(
            multi_agent_extraction, "proposer_json", side_effect=fake_proposer
        ), mock.patch.object(
            multi_agent_extraction, "merge_role_extractions", return_value={}
        ):
            multi_agent_extraction.extract_paper_multi_agent(
                "2605.5", "Paper", hint, "body text", llm_scope=None
            )

        self.assertEqual(len(seen), 5)
        self.assertIn(hint, seen["taxonomy_overview"])
        for role in (
            "empirical_results",
            "claims_methods",
            "graph_context",
            "research_facets",
        ):
            self.assertNotIn(hint, seen[role])
            self.assertIn("body text", seen[role])

    def test_taxonomy_leads_the_prompt_so_it_can_be_cached(self):
        hint = "Available taxonomy leaf nodes:\n  ml.dl.cv.detection"
        prompt = multi_agent_extraction._paper_user_prompt(
            "2605.6", "Paper", hint, "body", include_taxonomy=True
        )

        self.assertTrue(prompt.startswith(hint))
        self.assertLess(prompt.index(hint), prompt.index("2605.6"))


class PromptCharBudgetTests(unittest.TestCase):
    def test_default_paper_budget_covers_a_typical_corpus_paper(self):
        # Corpus measured 2026-08-17: mean 56,727 chars, max 80,000.
        self.assertGreaterEqual(extraction_agent.MAX_PROMPT_CHARS, 56_727)

    def test_compaction_still_bounds_oversized_papers(self):
        text = "Abstract\n" + ("word " * 40_000)
        compact = extraction_agent._compact_paper_text(text, max_chars=1_000)

        self.assertLessEqual(len(compact), 1_000)


if __name__ == "__main__":
    unittest.main()
