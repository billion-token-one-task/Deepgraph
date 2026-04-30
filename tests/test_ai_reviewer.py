import json
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.ai_reviewer import review_manuscript
from agents.artifact_manager import artifact_path, record_artifact
from tests.temp_utils import temporary_workdir


class FakeReviewerDb:
    def __init__(self, run):
        self.run = run

    def fetchone(self, sql, params=()):
        if "FROM experiment_runs" in sql:
            return self.run
        return None


def _run(workdir: Path) -> dict:
    return {
        "id": 31,
        "deep_insight_id": 9,
        "workdir": str(workdir),
        "hypothesis_verdict": "confirmed",
    }


def _write_manuscript(workdir: Path):
    paper = artifact_path(workdir, "artifacts/manuscript/paper.md")
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text("# Paper\n\nA grounded result.", encoding="utf-8")
    record_artifact(workdir, 31, "manuscript", paper)


def _write_candidate_manuscript(workdir: Path):
    paper = artifact_path(workdir, "artifacts/manuscript/paper_candidate.md")
    paper.parent.mkdir(parents=True)
    paper.write_text("# Candidate\n\nEvidence-gated candidate.", encoding="utf-8")
    record_artifact(workdir, 31, "manuscript", paper)


class AiReviewerTests(unittest.TestCase):
    def test_review_json_validates_and_writes_artifacts(self):
        with temporary_workdir() as workdir:
            _write_manuscript(workdir)
            fake_db = FakeReviewerDb(_run(workdir))
            llm_review = {
                "overall_score": 6,
                "recommendation": "weak_reject",
                "major_concerns": ["Needs a stronger baseline."],
                "minor_concerns": ["Clarify setup."],
                "required_experiments": ["Add one ablation."],
                "citation_risks": [],
                "reproducibility_risks": ["Seeds not reported."],
            }

            with patch("agents.ai_reviewer.db", fake_db), \
                 patch("agents.ai_reviewer.call_llm_json", return_value=(llm_review, 123)):
                result = review_manuscript(31)

            review_json = json.loads((workdir / "artifacts" / "reviews" / "review.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(review_json["overall_score"], 6)
            self.assertTrue((workdir / "artifacts" / "reviews" / "review.md").exists())
            manifest = json.loads((workdir / "artifact_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("artifacts/reviews/review.json", {item["path"] for item in manifest["artifacts"]})

    def test_missing_manuscript_returns_error_without_llm_call(self):
        with temporary_workdir() as workdir:
            fake_db = FakeReviewerDb(_run(workdir))

            with patch("agents.ai_reviewer.db", fake_db), \
                 patch("agents.ai_reviewer.call_llm_json", side_effect=AssertionError("should not call LLM")):
                result = review_manuscript(31)

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["reason"], "manuscript_not_found")

    def test_review_accepts_evidence_gated_candidate_manuscript(self):
        with temporary_workdir() as workdir:
            _write_candidate_manuscript(workdir)
            fake_db = FakeReviewerDb(_run(workdir))

            with patch("agents.ai_reviewer.db", fake_db), \
                 patch("agents.ai_reviewer.call_llm_json", return_value=({
                     "overall_score": 5,
                     "recommendation": "borderline",
                     "major_concerns": [],
                     "minor_concerns": [],
                     "required_experiments": [],
                     "citation_risks": [],
                     "reproducibility_risks": [],
                 }, 22)):
                result = review_manuscript(31)

            self.assertEqual(result["status"], "complete")

    def test_review_prompt_preserves_long_manuscript_tail_and_artifact_context(self):
        with temporary_workdir() as workdir:
            paper = artifact_path(workdir, "artifacts/manuscript/paper_candidate.md")
            paper.parent.mkdir(parents=True)
            paper.write_text(
                "# Candidate\n\n"
                + ("Middle evidence.\n" * 6000)
                + "## Limitations\n\nTail limitation that must reach the reviewer.\n",
                encoding="utf-8",
            )
            result_dir = workdir / "artifacts" / "results"
            result_dir.mkdir(parents=True)
            (result_dir / "statistical_report.json").write_text(json.dumps({
                "primary_metric": "safe_return",
                "baseline_method": "reward_only",
                "best_method": "lagrangian_penalty_4.00",
                "absolute_best_method": "occupancy_lp_optimal",
                "aggregate_metric_summaries": [{
                    "method": "occupancy_lp_optimal",
                    "metric": "safe_return",
                    "mean": 2.0,
                    "n": 10,
                }],
            }), encoding="utf-8")
            fake_db = FakeReviewerDb(_run(workdir))

            with patch("agents.ai_reviewer.db", fake_db), \
                 patch("agents.ai_reviewer.call_llm_json", return_value=({
                     "overall_score": 5,
                     "recommendation": "borderline",
                     "major_concerns": [],
                     "minor_concerns": [],
                     "required_experiments": [],
                     "citation_risks": [],
                     "reproducibility_risks": [],
                 }, 22)) as llm:
                result = review_manuscript(31)

            prompt = llm.call_args.args[1]
            self.assertEqual(result["status"], "complete")
            self.assertIn("Tail limitation that must reach the reviewer", prompt)
            self.assertIn("statistical_report.json", prompt)
            self.assertIn("occupancy_lp_optimal", prompt)


if __name__ == "__main__":
    unittest.main()
