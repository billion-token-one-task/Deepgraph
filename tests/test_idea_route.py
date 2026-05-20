import unittest

from agents.idea_route import classify_idea_route


class IdeaRouteTests(unittest.TestCase):
    def test_full_paper_route_requires_complete_benchmark_contract(self):
        route = classify_idea_route(
            {
                "title": "Counterfactual gain-guided reasoning",
                "novelty_status": "confirmed novel gap",
                "problem_statement": "Extra reasoning is useful only for some instances.",
                "resource_class": "gpu_large",
                "proposed_method": {
                    "name": "CGGR",
                    "definition": "Estimate counterfactual utility gain and route additional reasoning only when the expected gain is positive.",
                },
                "experimental_plan": {
                    "benchmark_targets": [{"name": "GSM8K"}],
                    "model_targets": [{"name": "Qwen2.5-7B-Instruct"}],
                    "baselines": [{"name": "direct"}, {"name": "cot"}, {"name": "self_consistency"}],
                    "ablations": [{"name": "no_gate"}, {"name": "random_gate"}],
                    "minimum_seeds": 3,
                    "metrics": {"primary": "accuracy"},
                    "real_benchmark_required": True,
                },
            }
        )

        self.assertEqual(route["route"], "full_paper")
        self.assertEqual(route["claim_strength"], "strong")
        self.assertTrue(route["paper_allowed"])
        self.assertEqual(route["missing"], [])

    def test_partial_novelty_routes_to_workshop_not_submission(self):
        route = classify_idea_route(
            {
                "title": "Adaptive reasoning gate",
                "novelty_status": "partially_exists",
                "problem_statement": "Always-on reasoning wastes budget.",
                "resource_class": "gpu_large",
                "proposed_method": {
                    "name": "AdaptiveGate",
                    "definition": "Train a lightweight policy to decide whether to allocate additional inference-time reasoning.",
                },
                "experimental_plan": {
                    "benchmark_targets": [{"name": "GSM8K"}],
                    "model_targets": [{"name": "Qwen2.5-7B-Instruct"}],
                    "baselines": [{"name": "direct"}, {"name": "cot"}],
                    "ablations": [{"name": "no_gate"}, {"name": "random_gate"}],
                    "minimum_seeds": 3,
                    "metrics": {"primary": "accuracy"},
                    "real_benchmark_required": True,
                },
            }
        )

        self.assertEqual(route["route"], "workshop")
        self.assertFalse(route["paper_allowed"])
        self.assertIn("sharpen_novelty_boundary", route["missing"])

    def test_missing_real_benchmark_routes_to_research_note(self):
        route = classify_idea_route(
            {
                "title": "Mechanism insight",
                "problem_statement": "A plausible mechanism should be checked.",
                "proposed_method": {
                    "name": "ProbeMethod",
                    "definition": "Use a lightweight diagnostic intervention to inspect the hypothesized mechanism.",
                },
                "experimental_plan": {
                    "datasets": ["synthetic_probe"],
                    "baselines": ["baseline-a"],
                    "metrics": {"primary": "score"},
                },
            }
        )

        self.assertIn(route["route"], {"research_note", "probe"})
        self.assertFalse(route["paper_allowed"])
        self.assertIn("real_benchmark_dataset", route["missing"])


if __name__ == "__main__":
    unittest.main()
