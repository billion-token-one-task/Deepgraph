import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.experiment_forge import forge_experiment, generate_scaffold, scout_codebase, setup_workspace
from tests.temp_utils import temporary_workdir


class ExperimentForgePlanIntegrationTests(unittest.TestCase):
    def test_setup_workspace_creates_fresh_directory_per_forge(self):
        with temporary_workdir() as root:
            stale = root / "exp_9_Reused"
            stale.mkdir()
            (stale / "artifact_manifest.json").write_text("stale", encoding="utf-8")

            with patch("agents.experiment_forge.EXPERIMENT_WORKDIR", root), \
                 patch("agents.experiment_forge.db.fetchone", return_value={"title": "Reused"}):
                first = setup_workspace(9, {"url": "scratch", "name": "minimal"})
                second = setup_workspace(9, {"url": "scratch", "name": "minimal"})

            self.assertNotEqual(first, second)
            self.assertTrue(first.name.startswith("exp_9_Reused_"))
            self.assertTrue(second.name.startswith("exp_9_Reused_"))
            self.assertFalse((first / "artifact_manifest.json").exists())
            self.assertFalse((second / "artifact_manifest.json").exists())

    def test_tier1_insight_without_proposed_method_can_be_scouted(self):
        insight = {
            "id": 1,
            "tier": 1,
            "title": "Social-choice CMDPs",
            "formal_structure": "Safe RL and fairness share constrained occupancy measures.",
            "transformation": "Map group constraints to safety-margin utilities.",
            "predictions": json.dumps(["Dual variables should match."]),
            "source_node_ids": json.dumps(["ml.rl.safe", "ml.theory.fairness"]),
        }

        with patch("agents.experiment_forge.db.fetchall", return_value=[]), \
             patch("agents.experiment_forge.call_llm_json", return_value=({
                 "codebase": {"url": "scratch", "name": "minimal"}
             }, 0)):
            result = scout_codebase(insight)

        self.assertEqual(result["url"], "scratch")

    def test_tier1_insight_without_proposed_method_can_generate_fallback_scaffold(self):
        insight = {
            "id": 1,
            "tier": 1,
            "title": "Social-choice CMDPs",
            "formal_structure": "Safe RL and fairness share constrained occupancy measures.",
            "problem_statement": None,
            "existing_weakness": None,
            "source_node_ids": json.dumps(["ml.rl.safe"]),
        }

        with temporary_workdir() as workdir:
            with patch("agents.experiment_forge.call_llm_json", side_effect=RuntimeError("offline")):
                result = generate_scaffold(insight, {"url": "scratch", "name": "minimal"}, workdir)

            self.assertIn("Social-choice CMDPs", result["program_md"])
            self.assertTrue((workdir / "program.md").exists())
            self.assertTrue((workdir / "evaluate.py").exists())
            self.assertTrue((workdir / "code" / "train.py").exists())

    def test_fallback_evaluator_does_not_emit_fake_zero_on_missing_metric(self):
        insight = {
            "id": 2,
            "tier": 1,
            "title": "Metric Discipline",
            "formal_structure": "A scaffold must not invent metrics.",
        }

        with temporary_workdir() as workdir:
            with patch("agents.experiment_forge.call_llm_json", side_effect=RuntimeError("offline")):
                generate_scaffold(insight, {"url": "scratch", "name": "minimal"}, workdir)

            evaluate_py = (workdir / "evaluate.py").read_text(encoding="utf-8")
            self.assertNotIn('print("metric_value: 0.0")', evaluate_py)
            self.assertIn("metric not found", evaluate_py)

    def test_scratch_scaffold_gets_bootstrap_train_when_llm_omits_it(self):
        insight = {
            "id": 3,
            "tier": 2,
            "title": "Runnable Scratch Scaffold",
            "experimental_plan": json.dumps({"metrics": {"primary": "accuracy", "direction": "higher"}}),
        }

        with temporary_workdir() as workdir:
            with patch("agents.experiment_forge.call_llm_json", return_value=({
                "program_md": "Run: `cd code && python train.py > ../run.log 2>&1`",
                "evaluate_py": "print('metric not found')",
                "success_criteria": {"metric_name": "accuracy", "metric_direction": "higher"},
            }, 0)):
                result = generate_scaffold(insight, {"url": "scratch", "name": "minimal"}, workdir)

            self.assertTrue(result["train_py_written"])
            self.assertTrue((workdir / "code" / "train.py").exists())

    def test_non_scratch_scaffold_gets_bootstrap_train_when_llm_omits_entrypoint(self):
        insight = {
            "id": 4,
            "tier": 1,
            "title": "Fairness Benchmark Harness",
            "experimental_plan": json.dumps({"metrics": {"primary": "accuracy", "direction": "higher"}}),
        }

        with temporary_workdir() as workdir:
            package_dir = workdir / "code" / "library_pkg"
            package_dir.mkdir(parents=True)
            (package_dir / "module.py").write_text("class LibraryOnly:\n    pass\n", encoding="utf-8")

            with patch("agents.experiment_forge.call_llm_json", return_value=({
                "program_md": "Run: `cd code && python library_pkg/module.py > ../run.log 2>&1`",
                "evaluate_py": "print('metric not found')",
                "success_criteria": {"metric_name": "accuracy", "metric_direction": "higher"},
            }, 0)):
                result = generate_scaffold(
                    insight,
                    {
                        "url": "https://github.com/example/library-only",
                        "name": "library-only",
                        "main_train_file": "library_pkg/module.py",
                    },
                    workdir,
                )

            self.assertTrue(result["train_py_written"])
            self.assertTrue((workdir / "code" / "train.py").exists())

    def test_fairlearn_fallback_scaffold_uses_real_fairness_harness(self):
        insight = {
            "id": 5,
            "tier": 1,
            "title": "Fairlearn Preference Cone Proxy",
            "experimental_plan": json.dumps({
                "datasets": ["synthetic grouped classification"],
                "success_metric": "Improve fairness-adjusted score while reporting accuracy and demographic parity gap.",
            }),
        }

        with temporary_workdir() as workdir:
            with patch("agents.experiment_forge.call_llm_json", side_effect=RuntimeError("offline")):
                result = generate_scaffold(
                    insight,
                    {
                        "url": "https://github.com/fairlearn/fairlearn",
                        "name": "fairlearn",
                        "main_train_file": "fairlearn/reductions/_exponentiated_gradient/exponentiated_gradient.py",
                    },
                    workdir,
                )

            train_py = (workdir / "code" / "train.py").read_text(encoding="utf-8")
            self.assertEqual(result["success_criteria"]["metric_name"], "fairness_score")
            self.assertIn("ExponentiatedGradient", train_py)
            self.assertIn("DemographicParity", train_py)
            self.assertIn("fairness_score", train_py)

    def test_forge_writes_execution_plan_and_records_artifact(self):
        insight = {
            "id": 21,
            "tier": 2,
            "title": "Calibrated adapters",
            "hypothesis": "Adapters improve calibration.",
            "experimental_plan": json.dumps({"metrics": {"primary": "ece", "direction": "lower"}}),
        }

        with temporary_workdir() as workdir:
            cursor = SimpleNamespace(lastrowid=55)

            with patch("agents.experiment_forge.db.fetchone", return_value=insight), \
                 patch("agents.experiment_forge.db.execute", return_value=cursor), \
                 patch("agents.experiment_forge.db.commit"), \
                 patch("agents.experiment_forge.scout_codebase", return_value={"url": "scratch", "name": "minimal"}), \
                 patch("agents.experiment_forge.setup_workspace", return_value=workdir), \
                 patch("agents.experiment_forge.generate_scaffold", return_value={
                     "program_md": "Run the experiment.",
                     "success_criteria": {"metric_name": "ece", "metric_direction": "lower"},
                     "tokens": 0,
                 }):
                result = forge_experiment(21)

            self.assertEqual(result["run_id"], 55)
            self.assertTrue((workdir / "execution_plan.json").exists())
            manifest = json.loads((workdir / "artifact_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_id"], 55)
            self.assertIn("execution_plan.json", {item["path"] for item in manifest["artifacts"]})

    def test_forge_writes_research_spec_and_benchmark_config_for_fairness_insight(self):
        insight = {
            "id": 31,
            "tier": 1,
            "title": "Group fairness via constrained preference optimization",
            "experimental_plan": json.dumps({
                "metrics": {"primary": "fairness_score", "direction": "higher"}
            }),
        }

        with temporary_workdir() as workdir:
            cursor = SimpleNamespace(lastrowid=77)

            with patch("agents.experiment_forge.db.fetchone", return_value=insight), \
                 patch("agents.experiment_forge.db.execute", return_value=cursor), \
                 patch("agents.experiment_forge.db.commit"), \
                 patch("agents.experiment_forge.scout_codebase", return_value={"url": "scratch", "name": "minimal"}), \
                 patch("agents.experiment_forge.setup_workspace", return_value=workdir), \
                 patch("agents.experiment_forge.generate_scaffold", return_value={
                     "program_md": "Run the benchmark suite.",
                     "success_criteria": {"metric_name": "fairness_score", "metric_direction": "higher"},
                     "tokens": 0,
                 }):
                result = forge_experiment(31)

            self.assertEqual(result["execution_mode"], "benchmark_suite")
            self.assertTrue((workdir / "research_spec.json").exists())
            self.assertTrue((workdir / "benchmark_config.json").exists())
            benchmark_config = json.loads((workdir / "benchmark_config.json").read_text(encoding="utf-8"))
            self.assertEqual(benchmark_config["capability"], "fairness_classification")
            manifest_paths = {
                item["path"]
                for item in json.loads((workdir / "artifact_manifest.json").read_text(encoding="utf-8"))["artifacts"]
            }
            self.assertIn("research_spec.json", manifest_paths)
            self.assertIn("benchmark_config.json", manifest_paths)

    def test_forge_writes_safe_rl_benchmark_config_for_cmdp_insight(self):
        insight = {
            "id": 32,
            "tier": 1,
            "title": "Social-choice CMDP policy optimization for safe RL and fairness",
            "formal_structure": "Constrained occupancy measures compare reward and safety costs.",
            "experimental_plan": "",
        }

        with temporary_workdir() as workdir:
            cursor = SimpleNamespace(lastrowid=78)

            with patch("agents.experiment_forge.db.fetchone", return_value=insight), \
                 patch("agents.experiment_forge.db.execute", return_value=cursor), \
                 patch("agents.experiment_forge.db.commit"), \
                 patch("agents.experiment_forge.scout_codebase", return_value={"url": "scratch", "name": "minimal"}), \
                 patch("agents.experiment_forge.setup_workspace", return_value=workdir), \
                 patch("agents.experiment_forge.generate_scaffold", return_value={
                     "program_md": "Run the benchmark suite.",
                     "success_criteria": {"metric_name": "safe_return", "metric_direction": "higher"},
                     "tokens": 0,
                 }):
                result = forge_experiment(32)

            self.assertEqual(result["execution_mode"], "benchmark_suite")
            benchmark_config = json.loads((workdir / "benchmark_config.json").read_text(encoding="utf-8"))
            self.assertEqual(benchmark_config["capability"], "safe_rl_cmdp")
            self.assertEqual(benchmark_config["primary_metric"], "safe_return")
            self.assertIn("occupancy_lp_optimal", benchmark_config["methods"])
            self.assertIn("lagrangian_penalty_4.00", benchmark_config["methods"])
            self.assertEqual(benchmark_config["candidate_method"], "lagrangian_penalty_4.00")


if __name__ == "__main__":
    unittest.main()
