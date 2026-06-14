import json
import tempfile
import unittest
from pathlib import Path

from agents.experiment_feedback import build_method_feedback, load_latest_method_feedback, write_method_feedback


class ExperimentFeedbackTests(unittest.TestCase):
    def test_candidate_trailing_baseline_generates_actionable_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            results = workdir / "results"
            results.mkdir(parents=True)
            (results / "benchmark_summary.json").write_text(
                json.dumps(
                    {
                        "metric_name": "accuracy",
                        "candidate_method": "GateSpec-SCIO",
                        "per_method": [
                            {"method": "GateSpec-SCIO", "accuracy": 0.2047},
                            {"method": "Always-Reason CoT", "accuracy": 0.2433},
                            {"method": "oracle_router", "accuracy": 0.2657},
                        ],
                        "paired_bootstrap_p": 0.42,
                    }
                ),
                encoding="utf-8",
            )
            (results / "routing_analysis.json").write_text(json.dumps({"route_rate": 0.01}), encoding="utf-8")

            payload = build_method_feedback(
                workdir=workdir,
                run_id=270,
                iteration=3,
                result={},
                result_judgement={"status": "discard"},
                history=[],
                criteria={"metric_name": "accuracy", "metric_direction": "higher"},
                baseline=0.2433,
                best_value=0.2047,
            )
            path = write_method_feedback(workdir, payload)

            self.assertTrue(path.exists())
            self.assertEqual(load_latest_method_feedback(workdir)["run_id"], 270)
            self.assertFalse(payload["method_diagnosis"]["beats_best_non_oracle"])
            self.assertTrue(any("Candidate trails" in row for row in payload["findings"]))
            self.assertTrue(any("gate" in row.lower() or "routing" in row.lower() for row in payload["next_actions"]))
            self.assertTrue(any("oracle" in row.lower() for row in payload["guardrails"]))

    def test_no_candidate_diff_is_automation_feedback_not_refutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            payload = build_method_feedback(
                workdir=workdir,
                run_id=41,
                iteration=1,
                result={"status": "blocked"},
                result_judgement={"status": "discard", "anomaly_type": "no_candidate_diff"},
                history=[],
                criteria={"metric_name": "accuracy", "metric_direction": "higher"},
                baseline=0.5,
                best_value=0.5,
            )

            self.assertTrue(any("automation failure" in row for row in payload["findings"]))
            self.assertTrue(any("real tracked" in row for row in payload["next_actions"]))


if __name__ == "__main__":
    unittest.main()
