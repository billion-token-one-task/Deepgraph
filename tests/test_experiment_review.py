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

    def test_review_marks_scratch_repo_as_smoke_only(self):
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

        self.assertEqual(judgement.recommended_route, "smoke_test")
        self.assertFalse(judgement.formal_experiment)
        self.assertTrue(judgement.smoke_test_only)


if __name__ == "__main__":
    unittest.main()
