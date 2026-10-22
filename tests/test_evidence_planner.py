import unittest
from pathlib import Path

from agents.evidence_planner import build_evidence_plan


class EvidencePlannerTests(unittest.TestCase):
    def test_performance_claim_requires_main_table_but_can_skip_visualization(self):
        insight = {
            "tier": 2,
            "title": "Better classifier",
            "problem_statement": "Improve benchmark accuracy.",
            "proposed_method": {
                "name": "BetterNet",
                "type": "architecture",
                "one_line": "Improves classification.",
                "key_properties": ["new block"],
            },
            "experimental_plan": {
                "baselines": [{"name": "BaseNet"}],
                "datasets": [{"name": "ImageNet"}],
                "metrics": {"primary": "accuracy"},
                "ablations": [],
            },
        }
        plan = build_evidence_plan(insight)
        self.assertEqual(plan["claim_type"], "performance")
        self.assertTrue(plan["main_table"]["enabled"])
        self.assertEqual(plan["main_table"]["priority"], "required")
        self.assertFalse(plan["visualization"]["enabled"])

    def test_efficiency_claim_requires_visualization(self):
        insight = {
            "tier": 2,
            "title": "Faster model with lower latency",
            "problem_statement": "Reduce inference latency and memory.",
            "proposed_method": {
                "name": "FastNet",
                "type": "architecture",
            },
            "experimental_plan": {
                "baselines": [{"name": "BaseNet"}],
                "datasets": [{"name": "ImageNet"}],
                "metrics": {"primary": "accuracy", "secondary": ["latency", "memory"]},
            },
        }
        plan = build_evidence_plan(insight)
        self.assertEqual(plan["claim_type"], "efficiency")
        self.assertTrue(plan["visualization"]["enabled"])
        self.assertEqual(plan["visualization"]["priority"], "required")


class FigureOrchestraBoundaryTests(unittest.TestCase):
    """The V1 boundary must keep reporting blocked, not manufacture assets."""

    def test_native_rendering_stays_refused(self):
        from agents.paperorchestra import figure_orchestra as boundary

        with self.assertRaises(boundary.FigureGenerationBlocked):
            boundary.render_native_figure({}, Path("."))


if __name__ == "__main__":
    unittest.main()
