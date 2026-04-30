import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.validation_loop import run_validation_loop
from tests.temp_utils import temporary_workdir


class FakeValidationDb:
    def __init__(self, run: dict, insight: dict | None = None):
        self.run = run
        self.insight = insight or {"id": run["deep_insight_id"], "title": "Toy insight"}
        self.iterations = []

    def fetchone(self, sql, params=()):
        if "FROM experiment_runs" in sql:
            return self.run
        if "FROM deep_insights" in sql:
            return self.insight
        return None

    def execute(self, sql, params=()):
        if sql.strip().startswith("INSERT INTO experiment_iterations"):
            self.iterations.append({
                "run_id": params[0],
                "iteration_number": params[1],
                "phase": params[2],
                "metric_value": params[3],
                "metric_name": params[4],
                "status": params[7],
                "description": params[8],
            })
            return None
        if "SET status='reproducing'" in sql:
            self.run["status"] = "reproducing"
            self.run["phase"] = "reproduction"
        elif "SET status='failed'" in sql:
            self.run["status"] = "failed"
            self.run["error_message"] = params[0]
        elif "hypothesis_verdict" in sql:
            self.run["status"] = "completed"
            self.run["hypothesis_verdict"] = params[0]
            if len(params) > 2:
                self.run["effect_size"] = params[-3]
                self.run["effect_pct"] = params[-2]
        elif "iterations_total" in sql and "iterations_kept" in sql:
            self.run["iterations_total"] = params[0]
            self.run["iterations_kept"] = params[1]
            self.run["best_metric_value"] = params[2]
            self.run["effect_size"] = params[3]
            self.run["effect_pct"] = params[4]
        elif "baseline_metric_value" in sql and "best_metric_value" in sql:
            self.run["baseline_metric_value"] = params[0]
            self.run["best_metric_value"] = params[1]
            self.run["phase"] = "hypothesis_testing"
            self.run["status"] = "testing"
        return None

    def commit(self):
        return None


class ValidationLoopArtifactTests(unittest.TestCase):
    def _write_success_files(self, workdir: Path, *, max_iterations: int = 0):
        (workdir / "success_criteria.json").write_text(json.dumps({
            "metric_name": "accuracy",
            "metric_direction": "higher",
            "solid": 0.75,
        }), encoding="utf-8")
        (workdir / "proxy_config.json").write_text(json.dumps({
            "time_budget_seconds": 10,
            "max_iterations": max_iterations,
            "reproduction_iterations": 1,
            "refute_min_iterations": 1,
        }), encoding="utf-8")

    def test_benchmark_mode_keeps_confirmed_verdict_when_only_review_revision_blocks_manuscript(self):
        from agents.validation_loop import _run_benchmark_mode

        with temporary_workdir() as workdir:
            fake_db = FakeValidationDb({
                "id": 10,
                "workdir": str(workdir),
                "deep_insight_id": 51,
            })

            with patch("agents.validation_loop.db", fake_db), \
                 patch("agents.validation_loop.run_benchmark_suite", return_value={"status": "complete"}), \
                 patch("agents.validation_loop.write_statistical_report", return_value={"status": "complete"}), \
                 patch("agents.validation_loop.write_evidence_gate", return_value={
                     "manuscript_status": "needs_more_experiments",
                     "blocking_reasons": ["review_requires_revision"],
                     "satisfied_requirements": [
                         "has_benchmark_results",
                         "has_baseline_comparison",
                         "has_statistical_report",
                         "has_multi_seed",
                         "has_multi_dataset",
                         "has_ablation",
                     ],
                 }), \
                 patch("agents.validation_loop._benchmark_metric_summary", return_value={
                     "baseline": 0.0,
                     "best": 1.0,
                     "effect": 1.0,
                     "effect_pct": 0.0,
                 }):
                result = _run_benchmark_mode(10, workdir)

            self.assertEqual(result["verdict"], "confirmed")
            self.assertEqual(fake_db.run["hypothesis_verdict"], "confirmed")

    def test_run_validation_loop_writes_metrics_and_logs_artifacts(self):
        with temporary_workdir() as workdir:
            code_dir = workdir / "code"
            code_dir.mkdir()
            (code_dir / "train.py").write_text("print('accuracy: 0.8')\n", encoding="utf-8")
            self._write_success_files(workdir)

            fake_db = FakeValidationDb({
                "id": 3,
                "workdir": str(workdir),
                "deep_insight_id": 44,
            })

            with patch("agents.validation_loop.db", fake_db):
                result = run_validation_loop(3)

            metrics = json.loads((workdir / "artifacts" / "results" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(result["verdict"], "inconclusive")
            self.assertEqual(metrics["baseline"], 0.8)
            self.assertEqual(metrics["best_value"], 0.8)
            self.assertEqual(fake_db.run["baseline_metric_value"], metrics["baseline"])
            self.assertEqual(fake_db.run["best_metric_value"], metrics["best_value"])
            self.assertTrue((workdir / "artifacts" / "logs" / "run.log").exists())

    def test_validation_confirms_only_after_kept_improvement(self):
        with temporary_workdir() as workdir:
            code_dir = workdir / "code"
            code_dir.mkdir()
            train_path = code_dir / "train.py"
            train_path.write_text("print('accuracy: 0.8')\n", encoding="utf-8")
            self._write_success_files(workdir, max_iterations=1)

            fake_db = FakeValidationDb({
                "id": 8,
                "workdir": str(workdir),
                "deep_insight_id": 49,
            })

            def improve(_workdir, _code_dir, *_args):
                train_path.write_text("print('accuracy: 0.9')\n", encoding="utf-8")
                return "Improved synthetic baseline"

            with patch("agents.validation_loop.db", fake_db), \
                 patch("agents.validation_loop._launch_coding_agent", side_effect=improve):
                result = run_validation_loop(8)

            self.assertEqual(result["verdict"], "confirmed")
            self.assertEqual(result["best_value"], 0.9)
            self.assertEqual(fake_db.run["iterations_kept"], 1)

    def test_failed_baseline_writes_failure_artifacts(self):
        with temporary_workdir() as workdir:
            code_dir = workdir / "code"
            code_dir.mkdir()
            (code_dir / "train.py").write_text("raise SystemExit(2)\n", encoding="utf-8")
            self._write_success_files(workdir)

            fake_db = FakeValidationDb({
                "id": 4,
                "workdir": str(workdir),
                "deep_insight_id": 45,
            })

            with patch("agents.validation_loop.db", fake_db):
                result = run_validation_loop(4)

            metrics = json.loads((workdir / "artifacts" / "results" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(result["verdict"], "failed")
            self.assertEqual(metrics["verdict"], "failed")
            self.assertEqual(metrics["reason"], "reproduction_failure")
            self.assertIn("reproduction failed", fake_db.run["error_message"])
            self.assertFalse((workdir / "artifacts" / "manuscript" / "paper.md").exists())

    def test_eval_fallback_zero_without_metric_does_not_create_baseline(self):
        with temporary_workdir() as workdir:
            code_dir = workdir / "code"
            code_dir.mkdir()
            (code_dir / "train.py").write_text("print('no metric here')\n", encoding="utf-8")
            (workdir / "evaluate.py").write_text(
                "print('metric_value: 0.0')\n",
                encoding="utf-8",
            )
            self._write_success_files(workdir)

            fake_db = FakeValidationDb({
                "id": 6,
                "workdir": str(workdir),
                "deep_insight_id": 47,
            })

            with patch("agents.validation_loop.db", fake_db):
                result = run_validation_loop(6)

            metrics = json.loads((workdir / "artifacts" / "results" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(result["verdict"], "failed")
            self.assertEqual(metrics["baseline"], None)
            self.assertEqual(fake_db.run["status"], "failed")

    def test_log_metric_survives_when_evaluator_finds_no_metric(self):
        with temporary_workdir() as workdir:
            code_dir = workdir / "code"
            code_dir.mkdir()
            (code_dir / "train.py").write_text("print('accuracy: 0.8')\n", encoding="utf-8")
            (workdir / "evaluate.py").write_text(
                "print('metric not found')\nraise SystemExit(1)\n",
                encoding="utf-8",
            )
            self._write_success_files(workdir)

            fake_db = FakeValidationDb({
                "id": 7,
                "workdir": str(workdir),
                "deep_insight_id": 48,
            })

            with patch("agents.validation_loop.db", fake_db):
                result = run_validation_loop(7)

            self.assertEqual(result["verdict"], "inconclusive")
            self.assertEqual(fake_db.run["baseline_metric_value"], 0.8)

    def test_uses_program_run_command_when_train_py_is_absent(self):
        with temporary_workdir() as workdir:
            code_dir = workdir / "code"
            target_dir = code_dir / "package"
            target_dir.mkdir(parents=True)
            (target_dir / "run_target.py").write_text("print('accuracy: 0.8')\n", encoding="utf-8")
            (workdir / "program.md").write_text(
                "Run: `cd code && python package/run_target.py > ../run.log 2>&1`\n",
                encoding="utf-8",
            )
            self._write_success_files(workdir)

            fake_db = FakeValidationDb({
                "id": 5,
                "workdir": str(workdir),
                "deep_insight_id": 46,
            })

            with patch("agents.validation_loop.db", fake_db):
                result = run_validation_loop(5)

            metrics = json.loads((workdir / "artifacts" / "results" / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(result["verdict"], "inconclusive")
            self.assertEqual(metrics["baseline"], 0.8)
            self.assertEqual(fake_db.iterations[0]["status"], "ok")

    def test_tier1_insight_context_reaches_coding_agent(self):
        with temporary_workdir() as workdir:
            code_dir = workdir / "code"
            code_dir.mkdir()
            train_path = code_dir / "train.py"
            train_path.write_text("print('accuracy: 0.8')\n", encoding="utf-8")
            self._write_success_files(workdir, max_iterations=1)

            fake_db = FakeValidationDb(
                {
                    "id": 9,
                    "workdir": str(workdir),
                    "deep_insight_id": 50,
                },
                {
                    "id": 50,
                    "title": "Preference cone fairness",
                    "proposed_method": None,
                    "formal_structure": "Closed convex preference cone over stakeholder utilities.",
                    "transformation": "Map group constraints into cone inequalities.",
                    "experimental_plan": json.dumps({
                        "procedure": "Compare unconstrained and constrained policies.",
                        "success_metric": "fairness_score improves while reporting accuracy.",
                    }),
                },
            )
            captured = {}

            def improve(_workdir, _code_dir, _iteration, method_desc, *_args):
                captured["method_desc"] = method_desc
                train_path.write_text("print('accuracy: 0.9')\n", encoding="utf-8")
                return "Used Tier 1 context"

            with patch("agents.validation_loop.db", fake_db), \
                 patch("agents.validation_loop._launch_coding_agent", side_effect=improve):
                run_validation_loop(9)

            self.assertIn("Closed convex preference cone", captured["method_desc"])
            self.assertIn("fairness_score", captured["method_desc"])

    def test_existing_git_repo_keeps_scaffold_train_after_discard_reset(self):
        with temporary_workdir() as workdir:
            code_dir = workdir / "code"
            code_dir.mkdir()
            (code_dir / "README.md").write_text("library checkout\n", encoding="utf-8")
            subprocess.run(["git", "init"], cwd=str(code_dir), capture_output=True, timeout=10)
            subprocess.run(["git", "add", "README.md"], cwd=str(code_dir), capture_output=True, timeout=10)
            subprocess.run(["git", "commit", "-m", "upstream"], cwd=str(code_dir), capture_output=True, timeout=10)
            train_path = code_dir / "train.py"
            train_path.write_text("print('accuracy: 0.8')\n", encoding="utf-8")
            self._write_success_files(workdir, max_iterations=1)

            fake_db = FakeValidationDb({
                "id": 10,
                "workdir": str(workdir),
                "deep_insight_id": 51,
            })

            def worsen(_workdir, _code_dir, *_args):
                train_path.write_text("print('accuracy: 0.7')\n", encoding="utf-8")
                return "Worse change"

            with patch("agents.validation_loop.db", fake_db), \
                 patch("agents.validation_loop._launch_coding_agent", side_effect=worsen):
                run_validation_loop(10)

            self.assertTrue(train_path.exists())
            self.assertIn("0.8", train_path.read_text(encoding="utf-8"))

    def test_validation_loop_uses_benchmark_suite_when_config_exists(self):
        with temporary_workdir() as workdir:
            (workdir / "benchmark_config.json").write_text(json.dumps({
                "capability": "fairness_classification",
                "primary_metric": "fairness_score",
                "metric_direction": "higher",
            }), encoding="utf-8")
            fake_db = FakeValidationDb({
                "id": 11,
                "workdir": str(workdir),
                "deep_insight_id": 52,
            })

            with patch("agents.validation_loop.db", fake_db), \
                 patch("agents.validation_loop.run_benchmark_suite", return_value={"status": "complete"}), \
                 patch("agents.validation_loop.write_statistical_report", return_value={"status": "complete"}), \
                 patch("agents.validation_loop.write_evidence_gate", return_value={
                     "manuscript_status": "needs_more_experiments",
                     "blocking_reasons": ["benchmark_suite_has_fewer_than_10_seeds"],
                 }):
                result = run_validation_loop(11)

            self.assertEqual(result["execution_mode"], "benchmark_suite")
            self.assertEqual(result["verdict"], "inconclusive")
            self.assertEqual(fake_db.run["status"], "completed")
            self.assertEqual(fake_db.run["hypothesis_verdict"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
