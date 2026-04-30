import json
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.experiment_plan_compiler import compile_execution_plan
from tests.temp_utils import temporary_workdir


class ExperimentPlanCompilerTests(unittest.TestCase):
    def test_tier2_experimental_plan_json_produces_execution_plan(self):
        insight = {
            "id": 11,
            "tier": 2,
            "title": "Better calibration for small models",
            "problem_statement": "Small models are poorly calibrated.",
            "experimental_plan": json.dumps({
                "metrics": {"primary": "ece", "direction": "lower"},
                "datasets": [{"name": "SyntheticQA"}],
                "baselines": [{"name": "temperature_scaling"}],
                "ablations": ["without_prior"],
            }),
        }
        scaffold = {"success_criteria": {"metric_name": "ece", "metric_direction": "lower", "solid": 0.1}}

        with temporary_workdir() as tmp:
            plan = compile_execution_plan(
                insight,
                tmp,
                {"url": "scratch", "name": "minimal"},
                scaffold,
            )

            self.assertEqual(plan["schema_version"], 1)
            self.assertEqual(plan["hypothesis"], "Small models are poorly calibrated.")
            self.assertEqual(plan["primary_metric"], "ece")
            self.assertEqual(plan["metric_direction"], "lower")
            self.assertEqual(plan["datasets"], ["SyntheticQA"])
            self.assertEqual(plan["baselines"], ["temperature_scaling"])
            self.assertIn({"name": "ablation", "required": False}, plan["stages"])
            self.assertTrue((tmp / "execution_plan.json").exists())

    def test_missing_experimental_plan_falls_back_to_scaffold_success_criteria(self):
        insight = {"id": 12, "tier": 2, "title": "Sparse adapters", "hypothesis": "Adapters reduce compute."}
        scaffold = {"success_criteria": {"metric_name": "accuracy", "metric_direction": "higher", "solid": 0.75}}

        with temporary_workdir() as tmp:
            plan = compile_execution_plan(insight, tmp, {"name": "scratch"}, scaffold)

            self.assertEqual(plan["primary_metric"], "accuracy")
            self.assertEqual(plan["metric_direction"], "higher")
            self.assertEqual(plan["success_criteria"]["solid"], 0.75)

    def test_invalid_json_uses_deterministic_minimal_plan(self):
        insight = {
            "id": 13,
            "tier": 2,
            "title": "Invalid plan",
            "hypothesis": "Fallback should be deterministic.",
            "experimental_plan": "{not json",
        }
        scaffold = {"success_criteria": {"metric_name": "metric", "metric_direction": "higher"}}

        with temporary_workdir() as tmp:
            first = compile_execution_plan(insight, tmp, {"name": "scratch"}, scaffold)
            second = compile_execution_plan(insight, tmp, {"name": "scratch"}, scaffold)

            self.assertEqual(first, second)
            self.assertEqual(first["datasets"], [])
            self.assertEqual(first["baselines"], [])
            self.assertEqual(first["primary_metric"], "metric")

    def test_compiler_does_not_call_llm(self):
        insight = {"id": 14, "title": "No API", "hypothesis": "No network calls."}

        with temporary_workdir() as tmp:
            with patch("agents.llm_client.call_llm_json", side_effect=AssertionError("should not call LLM")):
                plan = compile_execution_plan(insight, tmp, {"name": "scratch"}, {"success_criteria": {}})

            self.assertEqual(plan["hypothesis"], "No network calls.")


if __name__ == "__main__":
    unittest.main()
