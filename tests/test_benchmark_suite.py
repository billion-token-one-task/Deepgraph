import json
import unittest
from unittest.mock import patch

from tests.temp_utils import temporary_workdir


class FakeBenchmarkDb:
    def __init__(self, run: dict):
        self.run = run

    def fetchone(self, sql, params=()):
        if "FROM experiment_runs" in sql:
            return self.run
        return None


class BenchmarkSuiteTests(unittest.TestCase):
    def test_run_benchmark_suite_writes_results_artifact(self):
        from agents.artifact_manager import list_artifacts
        from agents.benchmark_suite import run_benchmark_suite

        with temporary_workdir() as workdir:
            (workdir / "benchmark_config.json").write_text(json.dumps({
                "schema_version": 1,
                "capability": "fairness_classification",
                "datasets": ["synthetic_grouped"],
                "methods": ["logistic_regression"],
                "seeds": [0],
                "primary_metric": "fairness_score",
                "metric_direction": "higher",
            }), encoding="utf-8")
            fake_db = FakeBenchmarkDb({"id": 12, "workdir": str(workdir)})

            with patch("agents.benchmark_suite.db", fake_db):
                result = run_benchmark_suite(12)

            results_path = workdir / "artifacts" / "results" / "benchmark_results.json"
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["rows"], 1)
            self.assertTrue(results_path.exists())
            self.assertIn(
                "artifacts/results/benchmark_results.json",
                {item["path"] for item in list_artifacts(workdir)},
            )

    def test_run_benchmark_suite_appends_ablation_rows(self):
        from agents.benchmark_suite import run_benchmark_suite

        with temporary_workdir() as workdir:
            (workdir / "benchmark_config.json").write_text(json.dumps({
                "schema_version": 1,
                "capability": "fairness_classification",
                "datasets": ["synthetic_grouped"],
                "methods": ["logistic_regression"],
                "seeds": [0],
                "primary_metric": "fairness_score",
                "metric_direction": "higher",
                "ablations": [{
                    "name": "preference_cone_penalty_sweep",
                    "methods": ["preference_cone_threshold_penalty_0.00"],
                }],
            }), encoding="utf-8")
            fake_db = FakeBenchmarkDb({"id": 12, "workdir": str(workdir)})

            with patch("agents.benchmark_suite.db", fake_db):
                result = run_benchmark_suite(12)

            payload = json.loads(
                (workdir / "artifacts" / "results" / "benchmark_results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["rows"], 2)
            ablation_rows = [row for row in payload["rows"] if row.get("analysis_type") == "ablation"]
            self.assertEqual(len(ablation_rows), 1)
            self.assertEqual(ablation_rows[0]["ablation"], "preference_cone_penalty_sweep")

    def test_run_benchmark_suite_dispatches_safe_rl_cmdp(self):
        from agents.benchmark_suite import run_benchmark_suite

        with temporary_workdir() as workdir:
            (workdir / "benchmark_config.json").write_text(json.dumps({
                "schema_version": 1,
                "capability": "safe_rl_cmdp",
                "datasets": ["risky_shortcut"],
                "methods": ["reward_only", "preference_cone_policy"],
                "seeds": [0],
                "primary_metric": "safe_return",
                "metric_direction": "higher",
            }), encoding="utf-8")
            fake_db = FakeBenchmarkDb({"id": 22, "workdir": str(workdir)})

            with patch("agents.benchmark_suite.db", fake_db):
                result = run_benchmark_suite(22)

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["capability"], "safe_rl_cmdp")
            self.assertEqual(result["rows"], 2)

    def test_run_benchmark_suite_writes_safe_rl_reproducibility_artifacts(self):
        from agents.artifact_manager import list_artifacts
        from agents.benchmark_suite import run_benchmark_suite

        with temporary_workdir() as workdir:
            (workdir / "benchmark_config.json").write_text(json.dumps({
                "schema_version": 1,
                "capability": "safe_rl_cmdp",
                "datasets": ["randomized_bandit"],
                "methods": ["reward_only", "deterministic_feasible_best", "occupancy_lp_optimal"],
                "seeds": [0, 1],
                "primary_metric": "safe_return",
                "metric_direction": "higher",
                "safety_penalty": 6.0,
            }), encoding="utf-8")
            fake_db = FakeBenchmarkDb({"id": 23, "workdir": str(workdir)})

            with patch("agents.benchmark_suite.db", fake_db):
                result = run_benchmark_suite(23)

            self.assertEqual(result["status"], "complete")
            env_appendix = workdir / "artifacts" / "results" / "cmdp_environment_appendix.json"
            lp_validation = workdir / "artifacts" / "results" / "lp_validation.json"
            repro_manifest = workdir / "artifacts" / "results" / "reproduction_manifest.json"
            self.assertTrue(env_appendix.exists())
            self.assertTrue(lp_validation.exists())
            self.assertTrue(repro_manifest.exists())

            appendix = json.loads(env_appendix.read_text(encoding="utf-8"))
            validation = json.loads(lp_validation.read_text(encoding="utf-8"))
            manifest = json.loads(repro_manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(appendix["environments"]), 2)
            first_env = appendix["environments"][0]
            self.assertIn("transitions", first_env)
            self.assertIn("rewards", first_env)
            self.assertIn("costs", first_env)
            self.assertIn("gamma", first_env)
            self.assertIn("cost_limit", first_env)
            self.assertEqual(validation["status"], "ok")
            self.assertIn("solver_backend_cross_checks", validation)
            self.assertIn("python", manifest)
            self.assertIn("commands", manifest)
            artifact_paths = {item["path"] for item in list_artifacts(workdir)}
            self.assertIn("artifacts/results/cmdp_environment_appendix.json", artifact_paths)
            self.assertIn("artifacts/results/lp_validation.json", artifact_paths)
            self.assertIn("artifacts/results/reproduction_manifest.json", artifact_paths)

    def test_missing_run_returns_error(self):
        from agents.benchmark_suite import run_benchmark_suite

        fake_db = FakeBenchmarkDb(None)

        with patch("agents.benchmark_suite.db", fake_db):
            result = run_benchmark_suite(99)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["reason"], "run_not_found")

    def test_missing_config_returns_error(self):
        from agents.benchmark_suite import run_benchmark_suite

        with temporary_workdir() as workdir:
            fake_db = FakeBenchmarkDb({"id": 13, "workdir": str(workdir)})

            with patch("agents.benchmark_suite.db", fake_db):
                result = run_benchmark_suite(13)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["reason"], "benchmark_config_not_found")


if __name__ == "__main__":
    unittest.main()
