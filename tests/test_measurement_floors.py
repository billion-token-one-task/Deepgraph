"""Measurement floors: a plan below the statistical floor is clamped at the
point where it becomes executable requirements (V1_SCAFFOLD_REGISTER.md)."""

import unittest

from meta_harness.runner_capability import (
    GENERATIVE_QA_MIN_SAMPLE_CAP,
    apply_measurement_floors,
    fold_field_mapping_roles,
    requirements_from_plan,
)
from meta_harness.runner_contract import recompute_metric


def _gsm8k_plan(**overrides):
    plan = {
        "benchmark_targets": [
            {
                "hf_dataset": "openai/gsm8k",
                "task_type": "math_qa",
                "primary_metric": "exact_match",
                "max_eval_examples": 4,
            }
        ],
        "model_targets": [
            {
                "hf_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "task": "causal_lm",
            }
        ],
        "minimum_seeds": 1,
    }
    plan.update(overrides)
    return plan


class MeasurementFloorTests(unittest.TestCase):
    def test_gsm8k_plan_is_clamped_to_the_floor(self):
        requirements = requirements_from_plan(_gsm8k_plan())
        self.assertEqual(requirements.sample_cap, GENERATIVE_QA_MIN_SAMPLE_CAP)
        self.assertEqual(requirements.metric.name, "numeric_accuracy")

    def test_plan_above_the_floor_is_untouched(self):
        plan = _gsm8k_plan()
        plan["benchmark_targets"][0]["max_eval_examples"] = 500
        requirements = requirements_from_plan(plan)
        self.assertEqual(requirements.sample_cap, 500)

    def test_non_numeric_dataset_keeps_its_metric(self):
        plan = _gsm8k_plan()
        plan["benchmark_targets"][0]["hf_dataset"] = "example/free-text-qa"
        requirements = requirements_from_plan(plan)
        self.assertEqual(requirements.metric.name, "exact_match")
        self.assertEqual(requirements.sample_cap, GENERATIVE_QA_MIN_SAMPLE_CAP)

    def test_classification_protocol_is_untouched(self):
        plan = {
            "benchmark_targets": [
                {
                    "hf_dataset": "example/reviews",
                    "task_type": "text_classification",
                    "text_field": "body",
                    "label_field": "category",
                    "primary_metric": "accuracy",
                    "max_eval_examples": 4,
                }
            ],
            "model_targets": [
                {"hf_model": "example/classifier", "task": "sequence_classification"}
            ],
        }
        requirements = requirements_from_plan(plan)
        self.assertEqual(requirements.sample_cap, 4)
        self.assertEqual(requirements.metric.name, "accuracy")

    def test_explicit_execution_requirements_are_also_clamped(self):
        explicit = requirements_from_plan(_gsm8k_plan()).to_dict()
        explicit["sample_cap"] = 8
        explicit["metric"]["name"] = "exact_match"
        requirements = requirements_from_plan({"execution_requirements": explicit})
        self.assertEqual(requirements.sample_cap, GENERATIVE_QA_MIN_SAMPLE_CAP)
        self.assertEqual(requirements.metric.name, "numeric_accuracy")

    def test_floor_application_is_idempotent(self):
        once = requirements_from_plan(_gsm8k_plan())
        self.assertEqual(apply_measurement_floors(once), once)


class FieldRoleAliasTests(unittest.TestCase):
    def _explicit(self, field_mapping):
        base = requirements_from_plan(_gsm8k_plan()).to_dict()
        base["dataset"]["field_mapping"] = field_mapping
        return {"execution_requirements": base}

    def test_plan_role_synonyms_are_folded_onto_the_contract(self):
        requirements = requirements_from_plan(
            self._explicit({"input": "question", "ground_truth": "answer"})
        )
        self.assertEqual(
            requirements.dataset.field_mapping,
            {"prompt": "question", "target": "answer"},
        )

    def test_colliding_roles_are_left_for_preflight_to_refuse(self):
        requirements = requirements_from_plan(
            self._explicit({"input": "q1", "question": "q2"})
        )
        self.assertEqual(
            requirements.dataset.field_mapping, {"input": "q1", "question": "q2"}
        )

    def test_contract_roles_pass_through_untouched(self):
        requirements = requirements_from_plan(
            self._explicit({"prompt": "question", "target": "answer"})
        )
        self.assertEqual(
            requirements.dataset.field_mapping,
            {"prompt": "question", "target": "answer"},
        )


class NumericAccuracyTests(unittest.TestCase):
    def test_last_number_comparison(self):
        rows = [
            {"prediction": "so the answer is 18 dollars", "target": "#### 18"},
            {"prediction": "step 2 gives 7", "target": "#### 21"},
        ]
        self.assertEqual(recompute_metric(rows, "numeric_accuracy"), 0.5)

    def test_numberless_pair_is_not_a_match(self):
        rows = [{"prediction": "no digits here", "target": "none either"}]
        self.assertEqual(recompute_metric(rows, "numeric_accuracy"), 0.0)


if __name__ == "__main__":
    unittest.main()


def test_dependency_available_accepts_requirement_specifiers():
    """A version specifier's dot must not be read as a package separator."""
    from meta_harness.runner_capability import HuggingFaceMetadataProbe

    probe = HuggingFaceMetadataProbe().dependency_available
    assert probe("json>=99.9") in (True, False)  # must not raise
    assert probe("definitely-not-a-real-module>=1.0") is False
    assert probe("json") is True
