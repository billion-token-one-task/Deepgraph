import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents import experiment_forge
from agents.experiment_review import review_experiment_candidate


class GenerateScaffoldTests(unittest.TestCase):
    def test_autofill_experiment_contracts_fills_missing_review_fields(self):
        enriched = experiment_forge._autofill_experiment_contracts(
            {
                "id": 7,
                "tier": 2,
                "title": "Secure linguistic communication and linguistic steganography as measure-preserving coding",
                "proposed_method": None,
                "experimental_plan": {
                    "models": ["gpt2-medium", "roberta-base", "sentence-transformers/all-mpnet-base-v2"],
                    "datasets": ["WikiText-103", "CNN/DailyMail"],
                    "procedure": "Compare coders, measure BER and detector AUC, and report bits/token.",
                },
                "supporting_papers": ["ACF", "Discop"],
            }
        )

        self.assertTrue(enriched["proposed_method"]["definition"])
        self.assertGreaterEqual(len(enriched["experimental_plan"]["baselines"]), 2)
        self.assertEqual(enriched["experimental_plan"]["metrics"]["primary"], "bit_error_rate")
        self.assertEqual(enriched["experimental_plan"]["datasets"][0]["name"], "WikiText-103")

    def test_autofill_experiment_contracts_makes_gpu_smoke_plan_reviewable(self):
        enriched = experiment_forge._autofill_experiment_contracts(
            {
                "id": 16,
                "tier": 2,
                "title": "SSH GPU Smoke Validation Experiment Auto Experiment Run",
                "resource_class": "gpu_small",
                "proposed_method": {
                    "name": "remote_gpu_smoke",
                    "type": "systems_validation",
                    "definition": "Run a short CUDA-backed tensor workload and report device/VRAM telemetry.",
                },
                "experimental_plan": {
                    "baselines": [{"name": "remote_cuda_probe"}],
                    "metrics": [{"name": "gpu_probe_score"}],
                    "compute_budget": {"gpu_hours": 0.01},
                },
            }
        )

        self.assertGreaterEqual(len(enriched["experimental_plan"]["baselines"]), 2)
        self.assertEqual(enriched["experimental_plan"]["datasets"][0]["name"], "synthetic_remote_gpu_probe")
        self.assertEqual(enriched["experimental_plan"]["metrics"]["primary"], "gpu_probe_score")
        self.assertEqual(enriched["experimental_plan"]["compute_budget"]["total_gpu_hours"], 0.01)

        judgement = review_experiment_candidate(
            enriched,
            codebase={"url": "https://github.com/example/repo", "name": "repo", "main_train_file": "train.py", "main_eval_command": "python train.py"},
            entrypoint_available=True,
        )
        self.assertEqual(judgement.recommended_route, "formal")

    def test_generate_scaffold_accepts_evidence_plan(self):
        insight = {
            "proposed_method": {
                "name": "CGGR",
                "type": "hybrid",
                "one_line": "Route extra reasoning only when gain is positive.",
                "definition": "Estimate the counterfactual gain of more reasoning.",
            },
            "experimental_plan": {
                "baselines": ["baseline-a"],
                "datasets": ["dataset-a"],
                "metrics": {"primary": "accuracy"},
                "expected_results": {"delta": "+2"},
            },
            "evidence_plan": {
                "main_table": {"enabled": True, "priority": "required"},
                "visualization": {"enabled": False, "priority": "skip"},
            },
            "problem_statement": "Decide when extra reasoning is useful.",
            "existing_weakness": "Always-on reasoning wastes budget.",
        }
        codebase = {
            "url": "scratch",
            "name": "minimal",
            "main_train_file": "train.py",
            "main_eval_command": "python evaluate.py",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            captured = {}

            def _fake_call_llm_json(system: str, prompt: str):
                captured["prompt"] = prompt
                return (
                    {
                        "program_md": "# program",
                        "evaluate_py": "print('ok')",
                        "success_criteria": {"metric_name": "accuracy"},
                    },
                    17,
                )

            with mock.patch.object(
                experiment_forge, "call_llm_json", side_effect=_fake_call_llm_json
            ):
                scaffold = experiment_forge.generate_scaffold(
                    insight, codebase, workdir
                )

        self.assertEqual(scaffold["tokens"], 17)
        self.assertIn("Adaptive Evidence Plan", captured["prompt"])
        self.assertIn("Honor this plan", captured["prompt"])

    def test_setup_workspace_falls_back_to_archive_when_git_missing(self):
        codebase = {
            "url": "https://github.com/example/project",
            "name": "project",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            workroot = Path(tmpdir)

            def _fake_archive(url: str, code_dir: Path) -> bool:
                (code_dir / "train.py").write_text("print('hello')", encoding="utf-8")
                return True

            with (
                mock.patch.object(
                    experiment_forge,
                    "ensure_run_workspace",
                    return_value={
                        "run_root": workroot / "idea_7" / "experiment" / "runs" / "run_70",
                        "code_root": workroot / "idea_7" / "experiment" / "runs" / "run_70" / "code",
                        "results_root": workroot / "idea_7" / "experiment" / "runs" / "run_70" / "results",
                        "spec_root": workroot / "idea_7" / "experiment" / "runs" / "run_70" / "spec",
                        "codex_root": workroot / "idea_7" / "experiment" / "runs" / "run_70" / "codex",
                    },
                ),
                mock.patch.object(experiment_forge, "_git_binary", return_value=None),
                mock.patch.object(
                    experiment_forge, "_download_repo_archive", side_effect=_fake_archive
                ) as download_archive,
            ):
                workdir = experiment_forge.setup_workspace(7, 70, codebase)
                self.assertTrue((workdir / "code" / "train.py").exists())
                self.assertTrue((workdir / "spec").exists())
                self.assertTrue((workdir / "codex").exists())
                download_archive.assert_called_once()

    def test_build_proxy_config_carries_repo_execution_hints(self):
        proxy = experiment_forge.build_proxy_config(
            {"compute_budget": {"total_gpu_hours": 12}},
            codebase={
                "main_train_file": "src/qa/inference.py",
                "main_eval_command": "python src/qa/inference.py --dataset strategyqa",
            },
        )

        self.assertEqual(proxy["main_train_file"], "src/qa/inference.py")
        self.assertEqual(
            proxy["baseline_command"],
            "python src/qa/inference.py --dataset strategyqa",
        )

    def test_normalize_codebase_metadata_clears_placeholder_entrypoint_for_real_repo(self):
        normalized = experiment_forge._normalize_codebase_metadata(
            {
                "url": "https://github.com/example/project",
                "name": "project",
                "main_train_file": "scratch",
                "main_eval_command": "unknown",
            }
        )

        self.assertEqual(normalized["main_train_file"], "")
        self.assertEqual(normalized["main_eval_command"], "")

    def test_checkpoint_run_state_serializes_incremental_fields(self):
        with (
            mock.patch.object(experiment_forge.db, "execute") as execute,
            mock.patch.object(experiment_forge.db, "commit") as commit,
        ):
            experiment_forge._checkpoint_run_state(
                42,
                phase="review_decision_ready",
                workdir="/tmp/run_42",
                codebase={"url": "https://github.com/example/project", "name": "project"},
                proxy_config={"formal_experiment": True, "smoke_test_only": False},
                baseline_metric_name="accuracy",
            )

        sql, params = execute.call_args.args
        self.assertIn("phase=?", sql)
        self.assertEqual(params[-1], 42)
        self.assertIn("review_decision_ready", params)
        self.assertIn("/tmp/run_42", params)
        self.assertTrue(any("formal_experiment" in str(value) for value in params))
        commit.assert_called_once()

    def test_fallback_scaffold_produces_bootstrap_train_py(self):
        scaffold = experiment_forge._fallback_scaffold(
            {"name": "CGGR", "definition": "Adaptive reasoning gate."},
            {"metrics": {"primary": "cost_adjusted_utility"}},
            {"url": "scratch", "name": "minimal"},
        )

        self.assertIn("train_py", scaffold)
        self.assertIn("metric_value", scaffold["train_py"])

    def test_codebase_entrypoint_check_requires_expected_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            code_dir = Path(tmpdir)
            (code_dir / "README.md").write_text("repo", encoding="utf-8")

            self.assertFalse(
                experiment_forge._codebase_has_expected_entrypoint(
                    code_dir, {"main_train_file": "src/qa/inference.py"}
                )
            )

    def test_review_routes_scratch_to_smoke_only(self):
        judgement = review_experiment_candidate(
            {
                "id": 9,
                "tier": 2,
                "title": "CGGR",
                "resource_class": "cpu",
                "proposed_method": {"name": "CGGR", "definition": "Adaptive gate."},
                "experimental_plan": {
                    "baselines": [{"name": "A"}, {"name": "B"}],
                    "datasets": [{"name": "dataset-a"}],
                    "metrics": {"primary": "accuracy"},
                },
            },
            codebase={"url": "scratch", "name": "minimal"},
            entrypoint_available=False,
        )

        self.assertTrue(judgement.smoke_test_only)
        self.assertFalse(judgement.formal_experiment)


if __name__ == "__main__":
    unittest.main()
