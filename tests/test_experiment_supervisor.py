import unittest

from agents.experiment_supervisor import build_supervisor_plan
from contracts import ExperimentSpec


class ExperimentSupervisorPlanTests(unittest.TestCase):
    def _spec(self) -> ExperimentSpec:
        return ExperimentSpec(
            deep_insight_id=1,
            proxy_config={"main_train_file": "train.py", "baseline_command": "python train.py"},
            experimental_plan={"datasets": ["bench"]},
            success_criteria={},
        )

    def _success_criteria(self) -> dict:
        return {"metric_name": "utility", "metric_direction": "higher"}

    def test_kept_change_below_baseline_is_recovery_not_false_refine(self):
        plan = build_supervisor_plan(
            spec=self._spec(),
            environment_report={"resolved_train_file": "train.py", "formal_ready": True},
            baseline=0.4206,
            best_so_far=0.3828,
            history=[{"iteration": 1, "status": "keep", "metric": 0.3828}],
            iteration=2,
            success_criteria=self._success_criteria(),
        )

        self.assertEqual(plan["mode"], "recover")
        self.assertIn("below baseline", plan["diagnosis"])
        self.assertIn("negative effect", plan["diagnosis"])
        self.assertLess(plan["baseline_comparison"]["effect"], 0)
        self.assertTrue(any("fails to beat" in action for action in plan["next_actions"]))

    def test_kept_change_above_baseline_remains_refine(self):
        plan = build_supervisor_plan(
            spec=self._spec(),
            environment_report={"resolved_train_file": "train.py", "formal_ready": True},
            baseline=0.40,
            best_so_far=0.43,
            history=[{"iteration": 1, "status": "keep", "metric": 0.43}],
            iteration=2,
            success_criteria=self._success_criteria(),
        )

        self.assertEqual(plan["mode"], "refine")
        self.assertIn("beats the baseline", plan["diagnosis"])
        self.assertGreater(plan["baseline_comparison"]["effect"], 0)

    def test_repeated_token_cap_keeps_redirect_away_from_microtuning(self):
        plan = build_supervisor_plan(
            spec=self._spec(),
            environment_report={"resolved_train_file": "train.py", "formal_ready": True},
            baseline=0.40,
            best_so_far=0.5434,
            history=[
                {
                    "iteration": 20,
                    "status": "keep",
                    "metric": 0.5434,
                    "description": "Updated yes/no zero-budget pass to use a 2-token cap.",
                },
                {
                    "iteration": 21,
                    "status": "keep",
                    "metric": 0.54348,
                    "description": "Tightened yes/no zero-budget pass to a 1-token cap.",
                },
                {
                    "iteration": 22,
                    "status": "discard",
                    "metric": 0.54348,
                    "description": "Reduced a non-yes/no cap and tied the best metric.",
                },
                {
                    "iteration": 23,
                    "status": "discard",
                    "metric": 0.5142,
                    "description": "Tried a relation-chain prompt and lost utility.",
                },
            ],
            iteration=24,
            success_criteria=self._success_criteria(),
        )

        self.assertEqual(plan["mode"], "redirect")
        self.assertIn("micro-tuning", plan["diagnosis"])
        self.assertTrue(any("Stop repeating token-cap" in action for action in plan["next_actions"]))

    def test_discarded_prompt_shortening_is_remembered_after_later_discards(self):
        plan = build_supervisor_plan(
            spec=self._spec(),
            environment_report={"resolved_train_file": "train.py", "formal_ready": True},
            baseline=0.40,
            best_so_far=0.5434,
            history=[
                {"iteration": 17, "status": "discard", "metric": 0.4627, "description": "Changed non-boolean zero-budget passes to ask for the shortest exact answer phrase only."},
                {"iteration": 18, "status": "keep", "metric": 0.5432, "description": "Added relation-chain routing."},
                {"iteration": 19, "status": "keep", "metric": 0.5433, "description": "Changed routing threshold."},
                {"iteration": 20, "status": "keep", "metric": 0.5434, "description": "Updated yes/no zero-budget pass to use a 2-token cap."},
                {"iteration": 21, "status": "keep", "metric": 0.54348, "description": "Tightened yes/no zero-budget pass to a 1-token cap."},
                {"iteration": 22, "status": "discard", "metric": 0.54348, "description": "Reduced a non-yes/no cap and tied the best metric."},
                {"iteration": 23, "status": "discard", "metric": 0.5142, "description": "Tried a relation-chain prompt and lost utility."},
                {"iteration": 24, "status": "discard", "metric": 0.5416, "description": "Tried organization relation routing."},
                {"iteration": 25, "status": "discard", "metric": 0.5123, "description": "Reduced high-risk relation-chain budget."},
                {"iteration": 26, "status": "discard", "metric": 0.3886, "description": "Implemented context-preserving prompting for benchmark context/passages/facts across all methods."},
            ],
            iteration=27,
            success_criteria=self._success_criteria(),
        )

        self.assertTrue(any("zero-budget open-answer prompt" in action for action in plan["next_actions"]))
        self.assertTrue(any("_build_cggr_zero_budget_prompt" in action for action in plan["next_actions"]))
        self.assertTrue(any("benchmark-context propagation" in action for action in plan["next_actions"]))
        self.assertTrue(any("phrase-only" in guardrail for guardrail in plan["guardrails"]))
        self.assertTrue(any("answer-shape regex" in guardrail for guardrail in plan["guardrails"]))
        self.assertTrue(any("context-propagation" in guardrail for guardrail in plan["guardrails"]))


if __name__ == "__main__":
    unittest.main()
