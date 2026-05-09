import tempfile
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from agents import validation_loop


class ValidationLoopGitFallbackTests(unittest.TestCase):
    def test_git_helpers_are_safe_when_git_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir)
            (code_dir / "train.py").write_text("print('hello')", encoding="utf-8")

            with mock.patch.object(validation_loop, "_git_binary", return_value=None):
                self.assertIsNone(
                    validation_loop._git_commit(code_dir, "test commit")
                )
                self.assertEqual(validation_loop._git_diff(code_dir), "")
                validation_loop._git_reset(code_dir, "deadbeef")

    def test_git_commit_excludes_agent_context_and_bytecode(self):
        git_bin = validation_loop._git_binary()
        if not git_bin:
            self.skipTest("git not available")
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir)
            (code_dir / "train.py").write_text("print('a')\n", encoding="utf-8")
            (code_dir / "AGENTS.md").write_text("old context\n", encoding="utf-8")
            pycache = code_dir / "__pycache__"
            pycache.mkdir()
            (pycache / "train.cpython-312.pyc").write_bytes(b"old")
            subprocess.run([git_bin, "init"], cwd=code_dir, check=True, capture_output=True)
            subprocess.run([git_bin, "config", "user.email", "test@example.com"], cwd=code_dir, check=True)
            subprocess.run([git_bin, "config", "user.name", "Test"], cwd=code_dir, check=True)
            subprocess.run([git_bin, "add", "-A"], cwd=code_dir, check=True, capture_output=True)
            subprocess.run([git_bin, "commit", "-m", "initial"], cwd=code_dir, check=True, capture_output=True)

            (code_dir / "train.py").write_text("print('b')\n", encoding="utf-8")
            (code_dir / "AGENTS.md").write_text("new context\n", encoding="utf-8")
            (pycache / "train.cpython-312.pyc").write_bytes(b"new")

            commit_hash = validation_loop._git_commit(code_dir, "method change")
            changed = subprocess.run(
                [git_bin, "show", "--name-only", "--format=", commit_hash],
                cwd=code_dir,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()

        self.assertEqual(changed, ["train.py"])

    def test_git_commit_returns_none_when_only_excluded_files_change(self):
        git_bin = validation_loop._git_binary()
        if not git_bin:
            self.skipTest("git not available")
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir)
            (code_dir / "train.py").write_text("print('a')\n", encoding="utf-8")
            (code_dir / "AGENTS.md").write_text("old context\n", encoding="utf-8")
            subprocess.run([git_bin, "init"], cwd=code_dir, check=True, capture_output=True)
            subprocess.run([git_bin, "config", "user.email", "test@example.com"], cwd=code_dir, check=True)
            subprocess.run([git_bin, "config", "user.name", "Test"], cwd=code_dir, check=True)
            subprocess.run([git_bin, "add", "-A"], cwd=code_dir, check=True, capture_output=True)
            subprocess.run([git_bin, "commit", "-m", "initial"], cwd=code_dir, check=True, capture_output=True)

            (code_dir / "AGENTS.md").write_text("new context\n", encoding="utf-8")

            commit_hash = validation_loop._git_commit(code_dir, "method change")

        self.assertIsNone(commit_hash)

    def test_git_diff_captures_latest_candidate_commit(self):
        git_bin = validation_loop._git_binary()
        if not git_bin:
            self.skipTest("git not available")
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir)
            (code_dir / "train.py").write_text("print('a')\n", encoding="utf-8")
            subprocess.run([git_bin, "init"], cwd=code_dir, check=True, capture_output=True)
            subprocess.run([git_bin, "config", "user.email", "test@example.com"], cwd=code_dir, check=True)
            subprocess.run([git_bin, "config", "user.name", "Test"], cwd=code_dir, check=True)
            subprocess.run([git_bin, "add", "-A"], cwd=code_dir, check=True, capture_output=True)
            subprocess.run([git_bin, "commit", "-m", "initial"], cwd=code_dir, check=True, capture_output=True)

            (code_dir / "train.py").write_text("print('candidate')\n", encoding="utf-8")
            commit_hash = validation_loop._git_commit(code_dir, "candidate")
            diff = validation_loop._git_diff(code_dir)

        self.assertIsNotNone(commit_hash)
        self.assertIn("+print('candidate')", diff)
        self.assertIn("-print('a')", diff)

    def test_find_train_file_prefers_nested_proxy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir)
            nested = code_dir / "src" / "qa"
            nested.mkdir(parents=True, exist_ok=True)
            target = nested / "inference.py"
            target.write_text("print('hello')", encoding="utf-8")

            resolved = validation_loop._find_train_file(
                code_dir, "src/qa/inference.py"
            )

        self.assertEqual(resolved, target)

    def test_run_validation_loop_blocks_non_formal_experiment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            (workdir / "code").mkdir()
            run = {
                "id": 7,
                "deep_insight_id": 3,
                "workdir": str(workdir),
                "proxy_config": '{"formal_experiment": false, "smoke_test_only": true}',
            }
            insight = {
                "id": 3,
                "tier": 2,
                "title": "Smoke",
                "proposed_method": '{"name": "M", "definition": "f(x)"}',
                "experimental_plan": '{"baselines": [], "datasets": [], "metrics": {"primary": "acc"}}',
            }

            with (
                mock.patch.object(validation_loop.db, "fetchone", side_effect=[run, insight]),
                mock.patch.object(validation_loop, "ALLOW_SMOKE_EXPERIMENT_VALIDATION", False),
                mock.patch.object(validation_loop.db, "execute") as execute,
                mock.patch.object(validation_loop.db, "commit"),
            ):
                result = validation_loop.run_validation_loop(7)

        self.assertEqual(result["verdict"], "blocked")
        self.assertEqual(result["reason"], "non_formal_experiment")
        execute.assert_called()

    def test_determine_final_verdict_marks_reproduction_only_runs(self):
        verdict = validation_loop._determine_final_verdict(
            baseline=1.0,
            best_value=1.0,
            direction="higher",
            criteria={"exciting": 0.8, "solid": 0.7},
            total_iters=0,
            total_kept=0,
            refute_min=30,
        )

        self.assertEqual(verdict, "reproduced")

    def test_determine_final_verdict_requires_real_improvement_for_confirmation(self):
        verdict = validation_loop._determine_final_verdict(
            baseline=1.0533,
            best_value=1.0533,
            direction="higher",
            criteria={"exciting": 0.79, "solid": 0.77},
            total_iters=1,
            total_kept=0,
            refute_min=30,
        )

        self.assertEqual(verdict, "inconclusive")

    def test_determine_final_verdict_accepts_benchmark_evidence(self):
        verdict = validation_loop._determine_final_verdict(
            baseline=0.77,
            best_value=0.80,
            direction="higher",
            criteria={"exciting": 0.79, "solid": 0.77},
            total_iters=0,
            total_kept=0,
            refute_min=30,
            benchmark_summary={
                "primary_metric": "utility",
                "candidate_method": "cggr",
                "best_method": "cggr",
                "num_seeds": 5,
                "per_method": {
                    "direct": {"utility": 0.71},
                    "adaptive_confidence": {"utility": 0.77},
                    "cggr": {"utility": 0.80},
                },
            },
        )

        self.assertEqual(verdict, "confirmed")

    def test_repo_snapshot_restore_recovers_multi_file_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir) / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            snapshot = Path(tmpdir) / "snapshot"
            (code_dir / "train.py").write_text("print('a')\n", encoding="utf-8")
            (code_dir / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

            validation_loop._snapshot_repo_tree(code_dir, snapshot)

            (code_dir / "train.py").write_text("print('b')\n", encoding="utf-8")
            (code_dir / "helper.py").unlink()
            (code_dir / "new_file.py").write_text("X = 2\n", encoding="utf-8")

            validation_loop._restore_repo_tree(snapshot, code_dir)

            self.assertEqual((code_dir / "train.py").read_text(encoding="utf-8"), "print('a')\n")
            self.assertEqual((code_dir / "helper.py").read_text(encoding="utf-8"), "VALUE = 1\n")
            self.assertFalse((code_dir / "new_file.py").exists())

    def test_launch_coding_agent_returns_codex_summary_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "train.py").write_text("print('baseline')\n", encoding="utf-8")
            spec = validation_loop.ExperimentSpec(
                deep_insight_id=1,
                experimental_plan={"baselines": [], "datasets": [], "metrics": {}},
                evidence_plan={"main_table": {"enabled": True}},
            )

            with (
                mock.patch.object(validation_loop.codex_executor, "codex_available", return_value=True),
                mock.patch.object(
                    validation_loop.codex_executor,
                    "run_codex_iteration",
                    return_value={
                        "ok": True,
                        "summary": "Codex changed repo files",
                        "artifact_paths": {"codex_last_message": "/tmp/last.json"},
                    },
                ),
                mock.patch.object(validation_loop, "_read_proxy_config", return_value={}),
            ):
                result = validation_loop._launch_coding_agent(
                    workdir,
                    code_dir,
                    1,
                    "Name: Method",
                    0.8,
                    0.7,
                    [],
                    spec=spec,
                    success_criteria={"metric_name": "acc"},
                    supervisor_plan={"mode": "bootstrap"},
                )

        self.assertEqual(result["executor"], "codex")
        self.assertIn("Codex", result["description"])
        self.assertIn("codex_last_message", result["artifact_paths"])

    def test_launch_coding_agent_does_not_legacy_fallback_after_codex_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "train.py").write_text("print('baseline')\n", encoding="utf-8")
            spec = validation_loop.ExperimentSpec(
                deep_insight_id=1,
                experimental_plan={"baselines": [], "datasets": [], "metrics": {}},
                evidence_plan={"main_table": {"enabled": True}},
            )

            with (
                mock.patch.object(validation_loop.codex_executor, "codex_available", return_value=True),
                mock.patch.object(
                    validation_loop.codex_executor,
                    "run_codex_iteration",
                    return_value={
                        "ok": False,
                        "stderr": "codex timed out",
                        "artifact_paths": {"codex_result": "/tmp/result.json"},
                    },
                ),
                mock.patch("agents.llm_client.call_llm") as call_llm,
                mock.patch.object(validation_loop, "_read_proxy_config", return_value={}),
            ):
                result = validation_loop._launch_coding_agent(
                    workdir,
                    code_dir,
                    2,
                    "Name: Method",
                    0.8,
                    0.7,
                    [],
                    spec=spec,
                    success_criteria={"metric_name": "acc"},
                    supervisor_plan={"mode": "redirect"},
                )

        self.assertEqual(result["executor"], "codex")
        self.assertTrue(result["code_generation_failed"])
        self.assertIn("codex timed out", result["description"])
        self.assertIn("codex_result", result["artifact_paths"])
        call_llm.assert_not_called()

    def test_resume_history_from_db_reconstructs_iteration_state(self):
        fairness_description = "x" * 120 + " benchmark_fairness_risk candidate-only canonicalizer"
        rows = [
            {"iteration_number": 4, "status": "keep", "metric_value": 1.1, "description": "first", "commit_hash": "abc"},
            {"iteration_number": 5, "status": "discard", "metric_value": 1.0, "description": fairness_description, "commit_hash": "def"},
            {"iteration_number": 6, "status": "keep", "metric_value": 1.2, "description": "third", "commit_hash": "fed"},
        ]

        with mock.patch.object(validation_loop.db, "fetchall", return_value=rows):
            history, iter_num, total_kept, best_commit = validation_loop._resume_history_from_db(7, 3)

        self.assertEqual(iter_num, 6)
        self.assertEqual(total_kept, 2)
        self.assertEqual(best_commit, "fed")
        self.assertIn("benchmark_fairness_risk", history[1]["description"])
        self.assertEqual(history[-1]["iteration"], 3)
        self.assertEqual(history[-1]["metric"], 1.2)

    def test_resume_history_from_db_prefers_coding_summary(self):
        description = validation_loop._iteration_db_description(
            result_judgement={
                "status": "discard",
                "summary": "Metric did not improve; discard the change.",
                "anomaly_type": "no_gain",
                "benchmark_semantic_warnings": [],
            },
            coding_summary="Tightened the zero-budget answer prompt in train.py",
            executor="codex",
        )
        rows = [
            {
                "iteration_number": 4,
                "status": "discard",
                "metric_value": 0.9,
                "description": description,
                "commit_hash": "abc",
            },
        ]

        with mock.patch.object(validation_loop.db, "fetchall", return_value=rows):
            history, _, _, _ = validation_loop._resume_history_from_db(7, 3)

        self.assertIn("Tightened the zero-budget answer prompt", history[0]["description"])
        self.assertIn("no_gain", history[0]["description"])


if __name__ == "__main__":
    unittest.main()
