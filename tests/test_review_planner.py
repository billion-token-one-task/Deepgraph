import json
import unittest

from agents.artifact_manager import artifact_path
from tests.temp_utils import temporary_workdir


class ReviewPlannerTests(unittest.TestCase):
    def test_review_planner_turns_required_experiments_into_followup_plan(self):
        from agents.review_planner import plan_followup_experiments

        with temporary_workdir() as workdir:
            review_path = artifact_path(workdir, "artifacts/reviews/review.json")
            review_path.parent.mkdir(parents=True)
            review_path.write_text(json.dumps({
                "recommendation": "reject",
                "required_experiments": ["Run 10 seeds.", "Add ablation."],
                "major_concerns": ["No statistics."],
            }), encoding="utf-8")

            plan = plan_followup_experiments(run_id=3, workdir=workdir)

            self.assertEqual(plan["status"], "needs_followup")
            self.assertEqual(len(plan["experiments"]), 2)
            self.assertEqual(plan["experiments"][0]["source"], "ai_review")
            self.assertEqual(plan["experiments"][0]["suggested_artifact"], "benchmark_config.json")
            self.assertTrue((workdir / "artifacts" / "results" / "followup_experiment_plan.json").exists())
            manifest = json.loads((workdir / "artifact_manifest.json").read_text(encoding="utf-8"))
            self.assertIn(
                "artifacts/results/followup_experiment_plan.json",
                {item["path"] for item in manifest["artifacts"]},
            )

    def test_review_planner_handles_missing_review(self):
        from agents.review_planner import plan_followup_experiments

        with temporary_workdir() as workdir:
            plan = plan_followup_experiments(run_id=3, workdir=workdir)

            self.assertEqual(plan["status"], "error")
            self.assertEqual(plan["reason"], "review_not_found")


if __name__ == "__main__":
    unittest.main()
