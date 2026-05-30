import unittest

from agents.paper_orchestra_pipeline import _scientific_review_gate


class ScientificReviewGateTests(unittest.TestCase):
    def test_controlled_selector_needs_strong_baseline_and_live_checks(self):
        state = {
            "method_name": "Diversity-Preserving Consensus",
            "result_packet": {
                "evidence_tier": "audited_controlled_materialized_benchmark",
                "benchmark_summary": {
                    "primary_metric": "accuracy",
                    "candidate_method": "Diversity-Preserving Consensus (ours)",
                    "num_seeds": 8,
                    "datasets": [
                        {"name": "GSM8K-Controlled", "num_test": 120},
                        {"name": "StrategyQA-Controlled", "num_test": 120},
                    ],
                    "per_method": {
                        "Vanilla Direct Answering": {"accuracy": 0.5516, "avg_new_tokens": 38.21},
                        "Confidence Routing": {"accuracy": 0.8286, "avg_new_tokens": 96.39},
                        "Always Multi-Agent Majority": {"accuracy": 0.8021, "avg_new_tokens": 462.04},
                        "Diversity-Preserving Consensus (ours)": {"accuracy": 0.8615, "avg_new_tokens": 320.47},
                        "Oracle Routing": {"accuracy": 0.9896, "avg_new_tokens": 62.44},
                    },
                    "bootstrap_ci": {
                        "p_value": 0.0078,
                        "paired_permutation_p_vs_direct": 0.0078,
                        "paired_permutation_p_vs_always_multi_agent": 0.0078,
                    },
                },
            },
        }
        tex = "DPC preserves dissent under disagreement and improves multi-agent consensus selection."

        review = _scientific_review_gate(tex, state)

        self.assertEqual(review["schema_version"], "scientific_review_gate_v2")
        self.assertEqual(review["total_examples"], 240)
        self.assertEqual(review["num_seeds"], 8)
        strongest = review["strongest_practical_baseline"]
        self.assertEqual(strongest["name"], "Confidence Routing")
        self.assertAlmostEqual(strongest["metric_gap"], 0.0329)
        self.assertAlmostEqual(strongest["token_delta"], 224.08)
        self.assertFalse(strongest["has_pairwise_test"])
        self.assertTrue(review["missing_analyses"]["pairwise_vs_strongest_baseline"])
        self.assertTrue(review["missing_analyses"]["disagreement_subset"])
        self.assertTrue(review["missing_analyses"]["quality_cost_frontier"])
        self.assertTrue(review["missing_analyses"]["live_sanity_check"])
        self.assertEqual(review["target_assessments"]["iclr_main"]["verdict"], "reject")
        self.assertIn(review["target_assessments"]["workshop"]["verdict"], {"promising_with_revisions", "borderline"})

    def test_pairwise_and_analysis_artifacts_reduce_review_risk(self):
        state = {
            "method_name": "Diversity-Preserving Consensus",
            "result_packet": {
                "evidence_tier": "audited_live_benchmark",
                "benchmark_summary": {
                    "primary_metric": "accuracy",
                    "candidate_method": "Diversity-Preserving Consensus (ours)",
                    "num_seeds": 8,
                    "datasets": [{"name": "GSM8K", "num_test": 300}],
                    "per_method": {
                        "Self-Consistency": {"accuracy": 0.81, "avg_new_tokens": 210.0},
                        "Confidence Routing": {"accuracy": 0.82, "avg_new_tokens": 100.0},
                        "Best-of-N Selector": {"accuracy": 0.83, "avg_new_tokens": 160.0},
                        "Debate Vote": {"accuracy": 0.84, "avg_new_tokens": 260.0},
                        "Adaptive Early Routing": {"accuracy": 0.80, "avg_new_tokens": 120.0},
                        "Diversity-Preserving Consensus (ours)": {"accuracy": 0.87, "avg_new_tokens": 180.0},
                    },
                    "bootstrap_ci": {"p_value": 0.01},
                    "pairwise_tests": {
                        "DPC_vs_Confidence_Routing": {"p_value": 0.02, "accuracy_gain": 0.05},
                        "DPC_vs_Debate_Vote": {"p_value": 0.03, "accuracy_gain": 0.03},
                    },
                    "subset_analysis": {"severe_disagreement": {"accuracy_gain": 0.08}},
                    "quality_cost_frontier": {"pareto_methods": ["Confidence Routing", "DPC"]},
                    "live_sanity_check": {"datasets": ["GSM8K"], "num_examples": 100},
                },
            },
        }
        tex = "DPC preserves dissent under disagreement and improves multi-agent consensus selection."

        review = _scientific_review_gate(tex, state)

        self.assertFalse(review["missing_analyses"]["pairwise_vs_strongest_baseline"])
        self.assertFalse(review["missing_analyses"]["disagreement_subset"])
        self.assertFalse(review["missing_analyses"]["quality_cost_frontier"])
        self.assertFalse(review["missing_analyses"]["live_sanity_check"])
        self.assertNotEqual(review["target_assessments"]["iclr_main"]["verdict"], "reject")


if __name__ == "__main__":
    unittest.main()
