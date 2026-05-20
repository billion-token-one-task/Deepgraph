import unittest

from agents.experiment_review import review_experiment_candidate


class ExperimentReviewTests(unittest.TestCase):
    def test_review_blocks_missing_baselines(self):
        judgement = review_experiment_candidate(
            {
                "id": 1,
                "tier": 2,
                "title": "Weak plan",
                "resource_class": "cpu",
                "proposed_method": {"name": "Method", "definition": "f(x)"},
                "experimental_plan": {
                    "datasets": [{"name": "dataset-a"}],
                    "metrics": {"primary": "accuracy"},
                },
            },
            codebase={"url": "https://github.com/example/repo", "main_train_file": "train.py"},
            entrypoint_available=True,
        )

        self.assertEqual(judgement.recommended_route, "blocked")
        self.assertIn("baselines", " ".join(judgement.blockers).lower())

    def test_review_blocks_scratch_without_real_model_target(self):
        judgement = review_experiment_candidate(
            {
                "id": 2,
                "tier": 2,
                "title": "Scratchable plan",
                "resource_class": "cpu",
                "proposed_method": {"name": "Method", "definition": "f(x)"},
                "experimental_plan": {
                    "baselines": [{"name": "BaselineA"}, {"name": "BaselineB"}],
                    "datasets": [{"name": "dataset-a"}],
                    "metrics": {"primary": "accuracy"},
                },
            },
            codebase={"url": "scratch", "name": "minimal"},
            entrypoint_available=False,
        )

        self.assertEqual(judgement.recommended_route, "blocked")
        self.assertFalse(judgement.formal_experiment)
        self.assertFalse(judgement.smoke_test_only)
        self.assertIn("real model", " ".join(judgement.blockers).lower())

    def test_review_allows_generated_real_benchmark_runner(self):
        judgement = review_experiment_candidate(
            {
                "id": 3,
                "tier": 2,
                "title": "Real benchmark plan",
                "resource_class": "gpu_large",
                "proposed_method": {"name": "Method", "definition": "f(x)"},
                "experimental_plan": {
                    "baselines": [{"name": "BaselineA"}, {"name": "BaselineB"}],
                    "datasets": [{"name": "GSM8K"}],
                    "model_targets": [{"name": "Qwen/Qwen2.5-7B-Instruct"}],
                    "metrics": {"primary": "accuracy"},
                    "compute_budget": {"total_gpu_hours": 12},
                },
            },
            codebase={"url": "scratch", "name": "generated-real-benchmark"},
            entrypoint_available=False,
        )

        self.assertEqual(judgement.recommended_route, "formal")
        self.assertTrue(judgement.formal_experiment)
        self.assertFalse(judgement.smoke_test_only)


if __name__ == "__main__":
    unittest.main()
