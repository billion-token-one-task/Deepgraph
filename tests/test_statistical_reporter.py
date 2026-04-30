import json
import unittest
from unittest.mock import patch

from tests.temp_utils import temporary_workdir


class FakeStatsDb:
    def __init__(self, run: dict):
        self.run = run

    def fetchone(self, sql, params=()):
        if "FROM experiment_runs" in sql:
            return self.run
        return None


class StatisticalReporterTests(unittest.TestCase):
    def test_statistical_report_selects_best_method_and_comparison(self):
        from agents.statistical_reporter import build_statistical_report

        rows = [
            {"dataset": "synthetic_grouped", "seed": 0, "method": "logistic_regression", "status": "ok", "metrics": {"fairness_score": 0.50}},
            {"dataset": "synthetic_grouped", "seed": 1, "method": "logistic_regression", "status": "ok", "metrics": {"fairness_score": 0.51}},
            {"dataset": "synthetic_grouped", "seed": 0, "method": "preference_cone_threshold", "status": "ok", "metrics": {"fairness_score": 0.61}},
            {"dataset": "synthetic_grouped", "seed": 1, "method": "preference_cone_threshold", "status": "ok", "metrics": {"fairness_score": 0.60}},
        ]

        report = build_statistical_report(
            rows,
            "fairness_score",
            "higher",
            baseline_method="logistic_regression",
        )

        self.assertEqual(report["best_method"], "preference_cone_threshold")
        self.assertEqual(report["comparisons"][0]["wins"], 2)

    def test_statistical_report_includes_secondary_metric_summaries(self):
        from agents.statistical_reporter import build_statistical_report

        rows = [
            {"dataset": "d", "seed": 0, "method": "logistic_regression", "status": "ok", "metrics": {
                "fairness_score": 0.50,
                "accuracy": 0.80,
                "demographic_parity_gap": 0.30,
                "equalized_odds_gap": 0.20,
            }},
            {"dataset": "d", "seed": 0, "method": "preference_cone_threshold", "status": "ok", "metrics": {
                "fairness_score": 0.61,
                "accuracy": 0.78,
                "demographic_parity_gap": 0.10,
                "equalized_odds_gap": 0.15,
            }},
        ]

        report = build_statistical_report(rows, "fairness_score", "higher", "logistic_regression")

        metric_names = {item["metric"] for item in report["metric_summaries"]}
        self.assertGreaterEqual(metric_names, {
            "fairness_score",
            "accuracy",
            "demographic_parity_gap",
            "equalized_odds_gap",
        })

    def test_statistical_report_compares_best_method_against_all_other_main_methods(self):
        from agents.statistical_reporter import build_statistical_report

        rows = []
        for seed in (0, 1, 2):
            rows.extend([
                {"dataset": "d", "seed": seed, "method": "logistic_regression", "status": "ok", "metrics": {"fairness_score": 0.50}},
                {"dataset": "d", "seed": seed, "method": "exponentiated_gradient", "status": "ok", "metrics": {"fairness_score": 0.56}},
                {"dataset": "d", "seed": seed, "method": "preference_cone_threshold", "status": "ok", "metrics": {"fairness_score": 0.62}},
            ])

        report = build_statistical_report(rows, "fairness_score", "higher", "logistic_regression")

        compared_baselines = {
            item["baseline"]
            for item in report["pairwise_comparisons"]
            if item["candidate"] == "preference_cone_threshold"
        }
        self.assertEqual(compared_baselines, {"logistic_regression", "exponentiated_gradient"})

    def test_statistical_report_can_use_configured_candidate_method_with_stronger_reference(self):
        from agents.statistical_reporter import build_statistical_report

        rows = []
        for seed in (0, 1, 2):
            rows.extend([
                {"dataset": "cmdp", "seed": seed, "method": "reward_only", "status": "ok", "metrics": {"safe_return": -1.0}},
                {"dataset": "cmdp", "seed": seed, "method": "preference_cone_policy", "status": "ok", "metrics": {"safe_return": 0.8}},
                {"dataset": "cmdp", "seed": seed, "method": "occupancy_lp_optimal", "status": "ok", "metrics": {"safe_return": 1.0}},
            ])

        report = build_statistical_report(
            rows,
            "safe_return",
            "higher",
            baseline_method="reward_only",
            candidate_method="preference_cone_policy",
        )

        self.assertEqual(report["best_method"], "preference_cone_policy")
        self.assertEqual(report["absolute_best_method"], "occupancy_lp_optimal")
        self.assertEqual(report["comparisons"][0]["candidate"], "preference_cone_policy")
        self.assertIn("occupancy_lp_optimal", {
            item["baseline"] for item in report["pairwise_comparisons"]
        })

    def test_statistical_report_includes_aggregate_metric_summaries(self):
        from agents.statistical_reporter import build_statistical_report

        rows = [
            {"dataset": "a", "seed": 0, "method": "baseline", "status": "ok", "metrics": {
                "safe_return": 0.0,
                "reward": 1.0,
                "cost": 0.5,
                "constraint_violation": 0.1,
            }},
            {"dataset": "b", "seed": 0, "method": "baseline", "status": "ok", "metrics": {
                "safe_return": 2.0,
                "reward": 3.0,
                "cost": 0.3,
                "constraint_violation": 0.0,
            }},
            {"dataset": "a", "seed": 0, "method": "candidate", "status": "ok", "metrics": {
                "safe_return": 4.0,
                "reward": 4.5,
                "cost": 0.2,
                "constraint_violation": 0.0,
            }},
            {"dataset": "b", "seed": 0, "method": "candidate", "status": "ok", "metrics": {
                "safe_return": 6.0,
                "reward": 6.5,
                "cost": 0.1,
                "constraint_violation": 0.0,
            }},
        ]

        report = build_statistical_report(rows, "safe_return", "higher", "baseline")

        aggregate = {
            (item["method"], item["metric"]): item["mean"]
            for item in report["aggregate_metric_summaries"]
        }
        self.assertEqual(aggregate[("baseline", "safe_return")], 1.0)
        self.assertEqual(aggregate[("candidate", "reward")], 5.5)
        self.assertEqual(aggregate[("baseline", "constraint_violation")], 0.05)

    def test_statistical_report_includes_runtime_summary(self):
        from agents.statistical_reporter import build_statistical_report

        rows = [
            {"dataset": "d", "seed": 0, "method": "baseline", "status": "ok", "metrics": {
                "score": 0.5,
                "runtime_seconds": 0.1,
            }},
            {"dataset": "d", "seed": 0, "method": "candidate", "status": "ok", "metrics": {
                "score": 0.6,
                "runtime_seconds": 0.2,
            }},
        ]

        report = build_statistical_report(rows, "score", "higher", "baseline", "candidate")

        keys = {
            (item["method"], item["metric"])
            for item in report["aggregate_metric_summaries"]
        }
        self.assertIn(("candidate", "runtime_seconds"), keys)

    def test_statistical_report_separates_ablation_rows_from_main_claim(self):
        from agents.statistical_reporter import build_statistical_report

        rows = [
            {"dataset": "d", "seed": 0, "method": "logistic_regression", "status": "ok", "analysis_type": "main", "metrics": {"score": 0.5}},
            {"dataset": "d", "seed": 0, "method": "candidate", "status": "ok", "analysis_type": "main", "metrics": {"score": 0.6}},
            {"dataset": "d", "seed": 0, "method": "candidate_ablation", "status": "ok", "analysis_type": "ablation", "ablation": "sweep", "metrics": {"score": 0.9}},
        ]

        report = build_statistical_report(rows, "score", "higher", "logistic_regression")

        self.assertEqual(report["best_method"], "candidate")
        self.assertEqual(report["ablation_summary"][0]["method"], "candidate_ablation")
        self.assertEqual(report["comparisons"][0]["losses"], 0)
        self.assertLessEqual(report["comparisons"][0]["paired_sign_test_p"], 1.0)

    def test_ablation_summary_keeps_safety_penalty_labels_separate(self):
        from agents.statistical_reporter import build_statistical_report

        rows = [
            {"dataset": "d", "seed": 0, "method": "candidate", "status": "ok", "analysis_type": "main", "metrics": {"score": 0.6}},
            {"dataset": "d", "seed": 0, "method": "candidate", "status": "ok", "analysis_type": "ablation", "ablation": "safety:safety_penalty=3.00", "metrics": {"score": 0.5}},
            {"dataset": "d", "seed": 0, "method": "candidate", "status": "ok", "analysis_type": "ablation", "ablation": "safety:safety_penalty=9.00", "metrics": {"score": 0.7}},
        ]

        report = build_statistical_report(rows, "score", "higher", "baseline", "candidate")

        labels = {item.get("ablation") for item in report["ablation_summary"]}
        self.assertIn("safety:safety_penalty=3.00", labels)
        self.assertIn("safety:safety_penalty=9.00", labels)

    def test_write_statistical_report_writes_json_and_table_artifacts(self):
        from agents.artifact_manager import list_artifacts
        from agents.statistical_reporter import write_statistical_report

        with temporary_workdir() as workdir:
            results_dir = workdir / "artifacts" / "results"
            results_dir.mkdir(parents=True)
            (results_dir / "benchmark_results.json").write_text(json.dumps({
                "rows": [
                    {"dataset": "synthetic_grouped", "seed": 0, "method": "logistic_regression", "status": "ok", "metrics": {"fairness_score": 0.50}},
                    {"dataset": "synthetic_grouped", "seed": 0, "method": "preference_cone_threshold", "status": "ok", "metrics": {"fairness_score": 0.60}},
                ]
            }), encoding="utf-8")
            (workdir / "benchmark_config.json").write_text(json.dumps({
                "primary_metric": "fairness_score",
                "metric_direction": "higher",
                "methods": ["logistic_regression", "preference_cone_threshold"],
                "fairness_penalty": 0.7,
            }), encoding="utf-8")
            fake_db = FakeStatsDb({"id": 14, "workdir": str(workdir)})

            with patch("agents.statistical_reporter.db", fake_db):
                result = write_statistical_report(14)

            self.assertEqual(result["status"], "complete")
            self.assertTrue((results_dir / "statistical_report.json").exists())
            self.assertTrue((workdir / "artifacts" / "tables" / "main_results.md").exists())
            report = json.loads((results_dir / "statistical_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["fairness_penalty"], 0.7)
            artifact_paths = {item["path"] for item in list_artifacts(workdir)}
            self.assertIn("artifacts/results/statistical_report.json", artifact_paths)
            self.assertIn("artifacts/tables/main_results.md", artifact_paths)


if __name__ == "__main__":
    unittest.main()
