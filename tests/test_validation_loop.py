import json
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

    def test_runner_contract_guard_allows_python_bool_literals_case_sensitively(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            spec_dir = workdir / "spec"
            code_dir.mkdir()
            spec_dir.mkdir()
            (spec_dir / "proxy_config.json").write_text(
                '{"real_benchmark_required": true}', encoding="utf-8"
            )
            train_py = code_dir / "train.py"
            train_py.write_text(
                'import json\nprint(json.dumps({"ok": True}, ensure_ascii=False))\n',
                encoding="utf-8",
            )

            violations = validation_loop._runner_contract_violations(
                workdir, code_dir, ["python", "train.py"]
            )

            train_py.write_text(
                'import json\nprint(json.dumps({"ok": True}, ensure_ascii=false))\n',
                encoding="utf-8",
            )
            bad_violations = validation_loop._runner_contract_violations(
                workdir, code_dir, ["python", "train.py"]
            )

        self.assertEqual(violations, [])
        self.assertIn("false", bad_violations[0])
        self.assertIn("False", bad_violations[0])

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
                "candidate_method": "candidate",
                "best_method": "candidate",
                "num_seeds": 5,
                "full_benchmark_completed": True,
                "raw_artifacts_complete": True,
                "claim_ledger_complete": True,
                "evaluator_id": "held-out-evaluator",
                "p_value": 0.01,
                "per_method": {
                    "direct": {"utility": 0.71},
                    "adaptive_confidence": {"utility": 0.77},
                    "candidate": {"utility": 0.80},
                },
            },
        )

        self.assertEqual(verdict, "supported")

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

    def test_launch_coding_agent_uses_resource_granted_role_route(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "train.py").write_text("print('baseline')\n", encoding="utf-8")
            spec = validation_loop.ExperimentSpec(
                run_id=77,
                deep_insight_id=1,
                experimental_plan={"baselines": [], "datasets": [], "metrics": {}},
                evidence_plan={"main_table": {"enabled": True}},
            )
            route = {
                "provider": "provider-a",
                "model": "model-a",
                "model_family": "family-a",
                "prompt_version": "validation-v1",
            }

            with (
                mock.patch.object(
                    validation_loop.db,
                    "fetchone",
                    return_value={
                        "agenda_id": 9,
                        "deep_insight_id": 1,
                        "resource_grant_id": 12,
                        "stage": "validation",
                    },
                ),
                mock.patch.object(
                    validation_loop,
                    "_read_proxy_config",
                    return_value={},
                ),
                mock.patch(
                    "agents.llm_client.configured_role_prompt_version",
                    return_value="validation-v1",
                ),
                mock.patch(
                    "agents.llm_client.call_llm_for_role",
                    return_value=(
                        "import math\n"
                        "VALUE = 1\n"
                        "print('candidate implementation with enough content', VALUE)\n",
                        31,
                        route,
                    ),
                ) as routed,
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

        self.assertEqual(result["executor"], "role_routed_llm")
        self.assertIn("llm_route", result["artifact_paths"])
        self.assertEqual(result["artifact_paths"]["llm_route"], route)
        self.assertEqual(routed.call_args.kwargs["resource_grant_id"], 12)
        self.assertEqual(routed.call_args.kwargs["role"], "proposer")

    def test_launch_coding_agent_fails_closed_when_role_route_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "train.py").write_text("print('baseline')\n", encoding="utf-8")
            spec = validation_loop.ExperimentSpec(
                run_id=77,
                deep_insight_id=1,
                experimental_plan={"baselines": [], "datasets": [], "metrics": {}},
                evidence_plan={"main_table": {"enabled": True}},
            )

            with (
                mock.patch.object(
                    validation_loop.db,
                    "fetchone",
                    return_value={
                        "agenda_id": 9,
                        "deep_insight_id": 1,
                        "resource_grant_id": 12,
                        "stage": "validation",
                    },
                ),
                mock.patch(
                    "agents.llm_client.configured_role_prompt_version",
                    return_value="validation-v1",
                ),
                mock.patch(
                    "agents.llm_client.call_llm_for_role",
                    side_effect=RuntimeError("provider unavailable"),
                ),
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

        self.assertEqual(result["executor"], "role_routed_llm")
        self.assertTrue(result["code_generation_failed"])
        self.assertIn("provider unavailable", result["description"])
        self.assertEqual(result["artifact_paths"], {})

    def test_reproduction_repair_records_resource_granted_route(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir()
            route = {
                "provider": "provider-a",
                "model": "model-a",
                "model_family": "family-a",
                "prompt_version": "validation-v1",
            }
            with (
                mock.patch.object(
                    validation_loop,
                    "_contract_context",
                    return_value=({}, {}, {}),
                ),
                mock.patch.object(
                    validation_loop.db,
                    "fetchone",
                    return_value={
                        "agenda_id": 9,
                        "deep_insight_id": 1,
                        "resource_grant_id": 12,
                        "stage": "validation",
                    },
                ),
                mock.patch(
                    "agents.llm_client.configured_role_prompt_version",
                    return_value="validation-v1",
                ),
                mock.patch(
                    "agents.llm_client.call_llm_json_for_role",
                    return_value=(
                        {
                            "summary": "repair import path",
                            "files": [
                                {
                                    "path": "train.py",
                                    "content": "print('repaired')\n",
                                }
                            ],
                        },
                        23,
                        route,
                    ),
                ) as routed,
            ):
                result = validation_loop._launch_reproduction_repair(
                    run_id=77,
                    workdir=workdir,
                    code_dir=code_dir,
                    repair_round=1,
                    baseline_command="python train.py",
                    metric_name="accuracy",
                    last_result={"status": "crash", "error": "missing import"},
                    environment_report={},
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["executor"], "role_routed_llm")
        self.assertEqual(result["llm_route"], route)
        self.assertEqual(routed.call_args.kwargs["resource_grant_id"], 12)

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


class ValidationLoopVerdictTests(unittest.TestCase):
    def test_no_candidate_diff_loop_is_inconclusive_not_refuted(self):
        history = [
            {
                "status": "discard",
                "metric": None,
                "result_judgement": {"anomaly_type": "no_candidate_diff"},
            }
            for _ in range(30)
        ]

        verdict = validation_loop._determine_final_verdict(
            baseline=0.7,
            best_value=0.7,
            direction="higher",
            criteria={},
            total_iters=len(history),
            total_kept=0,
            refute_min=30,
            automation_failed=validation_loop._hypothesis_testing_automation_failed(history),
        )

        self.assertEqual(verdict, "inconclusive")

    def test_benchmarked_no_gain_can_still_be_refuted(self):
        history = [
            {"status": "discard", "metric": 0.7, "result_judgement": {"anomaly_type": "no_gain"}}
            for _ in range(30)
        ]

        verdict = validation_loop._determine_final_verdict(
            baseline=0.7,
            best_value=0.7,
            direction="higher",
            criteria={},
            total_iters=len(history),
            total_kept=0,
            refute_min=30,
            automation_failed=validation_loop._hypothesis_testing_automation_failed(history),
        )

        self.assertEqual(verdict, "refuted")

    def test_recent_automation_failure_streak_counts_no_diff_only_until_real_metric(self):
        history = [
            {"status": "discard", "metric": 0.7, "result_judgement": {"anomaly_type": "no_gain"}},
            {"status": "discard", "metric": None, "result_judgement": {"anomaly_type": "no_candidate_diff"}},
            {"status": "discard", "metric": None, "result_judgement": {"anomaly_type": "pre_benchmark_guard"}},
        ]

        self.assertEqual(validation_loop._recent_automation_failure_streak(history), 2)

    def test_write_automation_failure_artifact_marks_not_scientific_verdict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = validation_loop._write_automation_failure_artifact(
                Path(tmpdir),
                run_id=9,
                insight_id=4,
                history=[
                    {
                        "status": "discard",
                        "metric": None,
                        "result_judgement": {"anomaly_type": "no_candidate_diff"},
                    }
                ],
                stop_reason="Automation failed: no benchmarked candidate method change.",
                method_desc="Method",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["failure_type"], "no_benchmarked_candidate_method_change")
        self.assertTrue(payload["not_scientific_verdict"])
        self.assertTrue(any("reforge" in action for action in payload["recommended_actions"]))

    def test_redesign_required_artifact_is_automation_failure(self):
        history = [
            {
                "status": "discard",
                "metric": None,
                "result_judgement": {"anomaly_type": "benchmark_mismatch_or_redesign_required"},
            }
            for _ in range(3)
        ]

        self.assertTrue(validation_loop._hypothesis_testing_automation_failed(history))

    def test_read_redesign_required_artifact_marks_not_scientific(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir()
            (code_dir / "EXPERIMENT_REDESIGN_REQUIRED.json").write_text(
                json.dumps(
                    {
                        "reason": "benchmark cannot exercise memory replay",
                        "mechanism_needed": "memory replay",
                        "benchmark_gap": "runner has no long-horizon state",
                    }
                ),
                encoding="utf-8",
            )

            payload = validation_loop._read_redesign_required_artifact(code_dir, workdir)

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertTrue(payload["not_scientific_verdict"])
        self.assertEqual(payload["recommended_route"], "reforge_or_benchmark_harness")

    def test_codex_agents_md_contains_mechanism_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir) / "code"
            code_dir.mkdir()

            path = validation_loop.codex_executor.write_iteration_agents_md(
                code_dir=code_dir,
                method_desc="Memory replay method",
                baseline=0.7,
                best_so_far=0.7,
                iteration=1,
                history=[],
                proxy={},
                success_criteria={"metric_name": "accuracy"},
                experimental_plan={},
                evidence_plan={},
                supervisor_plan={},
            )
            text = path.read_text(encoding="utf-8")

        self.assertIn("Mechanism Operationalization Contract", text)
        self.assertIn("EXPERIMENT_REDESIGN_REQUIRED.json", text)


if __name__ == "__main__":
    unittest.main()
