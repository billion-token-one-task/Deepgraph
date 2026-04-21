import tempfile
import unittest
from pathlib import Path

from agents.evidence_planner import build_evidence_plan
from agents.paperorchestra.figure_orchestra import run_figure_orchestra


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

    def test_disabled_visualization_skips_default_figure_generation(self):
        state = {
            "title": "No Figure Insight",
            "method_name": "NoFigureNet",
            "evidence_plan": {
                "visualization": {
                    "enabled": False,
                    "priority": "skip",
                    "reason": "Table-first claim.",
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = run_figure_orchestra(
                outline={},
                state=state,
                iterations=[],
                figures_dir=Path(tmpdir),
                baseline=None,
                metric_name="accuracy",
                paperbanana_cmd=None,
            )
            self.assertEqual(manifest["assets"], [])


if __name__ == "__main__":
    unittest.main()
