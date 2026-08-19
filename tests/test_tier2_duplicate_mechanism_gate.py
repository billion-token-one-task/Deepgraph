"""A pool-provenance label is not a mechanism, and must not condemn an idea.

deep_insights.mechanism_type is supposed to hold one of the six values the
Tier-2 prompt asks for. discovery_supervisor instead stamps the candidate
pool's NAME there, so every idea raised from the research-problem pool carries
the literal string "problem".

On 2026-08-19 twelve of agenda 10's ideas shared that value. The duplicate
gate's mechanism clause -- ``same_mechanism and title_score >= 0.18``, the one
clause that needs no provenance evidence -- was therefore true for
essentially every pair, and the gate collapsed into "any title sharing 18% of
its tokens with any prior idea is a duplicate". Discovery rejected the newly
invented SPAC-PI against idea 138 at title_sim=0.2 and node_overlap=0.0, the
M2 acceptance window sat at zero runs, and a funded proposal grant went unused.

The gate itself is sound; it was being fed a label it could not interpret.
"""

import unittest

from agents.paper_idea_agent import (
    MECHANISM_TAXONOMY,
    _is_mechanism,
)


class MechanismGateTests(unittest.TestCase):
    def test_taxonomy_values_are_mechanisms(self):
        for value in MECHANISM_TAXONOMY:
            self.assertTrue(_is_mechanism(value), value)

    def test_pool_provenance_labels_are_not_mechanisms(self):
        # the values discovery_supervisor actually writes
        for value in ("problem", "deep_insight", "paper_idea", "contradiction"):
            self.assertFalse(_is_mechanism(value), value)

    def test_missing_mechanism_is_not_a_match(self):
        for value in (None, "", "   "):
            self.assertFalse(_is_mechanism(value))

    def test_the_taxonomy_matches_the_prompt_the_agent_sends(self):
        import inspect

        from agents import paper_idea_agent

        source = inspect.getsource(paper_idea_agent)
        # the prompt spells the taxonomy as a pipe-separated enum
        for value in MECHANISM_TAXONOMY:
            self.assertIn(value, source)


if __name__ == "__main__":
    unittest.main()
