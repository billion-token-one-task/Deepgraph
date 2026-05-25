"""Tests for per-venue style guides and conference_guidelines wiring."""

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class StyleGuideTests(unittest.TestCase):
    def test_all_registered_venues_have_style_files(self):
        from agents.manuscript_templates import list_adapters
        from agents.manuscript_templates.style_guides import VENUE_STYLES_DIR

        for name in ("_EMPIRICAL_STATS.md", "_SECTION_WRITING_FRAMEWORK.md"):
            self.assertTrue((VENUE_STYLES_DIR / name).is_file(), name)
        for tid in list_adapters():
            path = VENUE_STYLES_DIR / f"{tid}.md"
            self.assertTrue(path.is_file(), f"missing style guide for {tid}: {path}")

    def test_merged_guide_includes_framework(self):
        from agents.manuscript_templates.style_guides import load_venue_style_guide

        g = load_venue_style_guide("iclr2026")
        self.assertIn("Contributions", g)
        self.assertIn("Related Work", g)
        self.assertIn("Problem Formulation", g)
        self.assertIn("median", g.lower())

    def test_build_conference_guidelines_includes_contributions_rule(self):
        from agents.paper_orchestra_prompts import build_conference_guidelines

        g = build_conference_guidelines("emnlp2024")
        self.assertIn("Contributions", g)
        self.assertIn("emnlp2024", g)
        self.assertIn("Related Work", g)
        self.assertIn("140", g)
        self.assertIn("183", g)

    def test_emnlp_adapter_registered(self):
        from agents.manuscript_templates import get_adapter

        ad = get_adapter("emnlp2024")
        self.assertEqual(ad.column_layout, "two_column")
        self.assertEqual(ad.max_pages, 8)

    def test_venue_router_knows_emnlp(self):
        from agents.venue_router import load_venue_config

        ids = {v.template_id for v in load_venue_config()}
        self.assertIn("emnlp2024", ids)


if __name__ == "__main__":
    unittest.main()
