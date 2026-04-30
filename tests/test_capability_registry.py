import unittest
from unittest.mock import patch


class CapabilityRegistryTests(unittest.TestCase):
    def test_registry_matches_fairness_spec_to_fairness_capability(self):
        from agents.capability_registry import select_capability

        spec = {
            "domain": "fairness",
            "task_type": "classification",
            "candidate_capabilities": ["fairness_classification"],
        }

        capability = select_capability(spec)

        self.assertEqual(capability["id"], "fairness_classification")
        self.assertEqual(capability["runner"], "benchmarks.fairness_classification.harness")

    def test_fairness_default_config_includes_public_fairness_datasets_and_postprocessing_baseline(self):
        from agents.capability_registry import get_capability

        capability = get_capability("fairness_classification")
        config = capability["default_config"]

        self.assertIn("openml_adult_sex", config["datasets"])
        self.assertIn("openml_german_credit_sex", config["datasets"])
        self.assertIn("fairlearn_credit_card_sex", config["datasets"])
        self.assertIn("fairlearn_bank_marketing_age", config["datasets"])
        self.assertIn("threshold_optimizer", config["methods"])
        self.assertIn("validation_selected_fairlearn_baseline", config["methods"])
        self.assertIn("validation_selected_preference_cone", config["methods"])
        self.assertNotIn("preference_cone_threshold", config["methods"])
        self.assertNotIn("synthetic_grouped", config["datasets"])
        self.assertNotIn("sklearn_breast_cancer_grouped", config["datasets"])
        self.assertNotIn("sklearn_wine_grouped", config["datasets"])
        self.assertIn("exposes trade-offs", config["scoped_claim"])
        self.assertIn("Validation-selected", config["scoped_claim"])
        self.assertNotIn("over logistic regression and fairness-aware baselines", config["scoped_claim"])

    def test_registry_falls_back_to_generic_when_specific_missing(self):
        from agents.capability_registry import select_capability

        spec = {
            "domain": "unknown",
            "task_type": "benchmark",
            "candidate_capabilities": ["missing", "generic_python_benchmark"],
        }

        capability = select_capability(spec)

        self.assertEqual(capability["id"], "generic_python_benchmark")

    def test_capability_reports_missing_dependencies(self):
        from agents.capability_registry import get_capability, missing_dependencies

        with patch("agents.capability_registry.importlib.util.find_spec", return_value=None):
            capability = get_capability("fairness_classification")
            missing = missing_dependencies(capability)

        self.assertIn("numpy", missing)
        self.assertIn("sklearn", missing)
        self.assertIn("fairlearn", missing)

    def test_safe_rl_capability_is_implemented_for_finite_cmdp_benchmarks(self):
        from agents.capability_registry import get_capability

        capability = get_capability("safe_rl_cmdp")
        config = capability["default_config"]

        self.assertEqual(capability["id"], "safe_rl_cmdp")
        self.assertTrue(capability["implemented"])
        self.assertEqual(capability["runner"], "benchmarks.safe_rl_cmdp.harness")
        self.assertIn("scipy", capability["required_packages"])
        self.assertIn("risky_shortcut", config["datasets"])
        self.assertIn("preference_cone_policy", config["methods"])
        self.assertIn("occupancy_lp_optimal", config["methods"])
        self.assertIn("lagrangian_penalty_4.00", config["methods"])
        self.assertEqual(config["candidate_method"], "lagrangian_penalty_4.00")
        self.assertEqual(config["reference_method"], "occupancy_lp_optimal")
        self.assertEqual(config["primary_metric"], "safe_return")


if __name__ == "__main__":
    unittest.main()
