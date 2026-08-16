"""Regression coverage for remote runner dependency and failure contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from meta_harness.backends.colab_cli import (
    ColabExecutionRequest,
    ColabExecutionResult,
    _runner_source,
)
from meta_harness.backends.colab_durable import _result_payload
from meta_harness.runner_capability import requirements_from_plan
from meta_harness.runner_materialization import materialize_runner_bundle
from orchestrator import colab_worker
from orchestrator import meta_compute_runtime


def _four_bit_preflight() -> dict[str, object]:
    return {
        "status": "passed",
        "id": 1,
        "adapter_id": "transformers_causal_lm_qa_v1",
        "adapter_version": "1.0.0",
        "dataset_revision": "dataset-revision",
        "model_revision": "model-revision",
        "requirements_json": json.dumps(
            {
                "task_protocol": "generative_qa",
                "dataset": {
                    "repository_id": "dataset",
                    "revision": "dataset-revision",
                    "split": "test",
                    "field_mapping": {"prompt": "question", "target": "answer"},
                },
                "model": {
                    "repository_id": "model",
                    "revision": "model-revision",
                    "framework": "transformers",
                    "task": "causal_lm",
                    "requires_cuda": True,
                    "quantization": "4bit",
                },
                "metric": {"name": "exact_match", "direction": "higher"},
                "candidate_hook": "candidate_prompt",
                "artifact_contract": ["final_results"],
                "preferred_backends": ["colab_gpu"],
            }
        ),
    }


class RunnerDependencyContractTests(unittest.TestCase):
    def test_colab_submission_keeps_portable_runner_artifacts_inside_code_root(self):
        with tempfile.TemporaryDirectory() as temp:
            workdir = Path(temp)
            code_dir = workdir / "code"
            code_dir.mkdir()
            (code_dir / "train.py").write_text("# runner", encoding="utf-8")
            run = {
                "id": 151,
                "agenda_id": 11,
                "deep_insight_id": 105,
                "resource_grant_id": 43,
                "workdir": str(workdir),
            }
            with (
                mock.patch.object(meta_compute_runtime.db, "fetchone", return_value=run),
                mock.patch.object(
                    meta_compute_runtime,
                    "_grant_from_row",
                    return_value=SimpleNamespace(
                        artifact_requirements=("final_results",), stage="pilot"
                    ),
                ),
                mock.patch.object(meta_compute_runtime, "submit_colab_work", return_value="job") as submit,
            ):
                assert meta_compute_runtime._submit_experiment_run_on_colab(
                    grant_row={}, experiment_run_id=151, timeout_seconds=60
                ) == "job"

        spec = submit.call_args.args[0]
        self.assertEqual(
            spec.command_tokens,
            (
                "python", "train.py", "--config", "execution_requirements.json",
                "--candidate-adapter", "candidate_adapter.py", "--output-dir", ".",
            ),
        )

    def test_colab_runner_collects_portable_runner_results_outside_code_dir(self):
        source = _runner_source(
            ColabExecutionRequest(
                agenda_id=11,
                idea_id=105,
                stage="pilot",
                resource_grant_id=43,
                idempotency_key="test",
                code_dir="/tmp/code",
                command_tokens=("python", "train.py"),
                environment={},
                timeout_seconds=60,
                artifact_paths=("final_results.json",),
                artifact_output_dir="/tmp/results",
            )
        )

        self.assertIn('root.parent / "results" / relative', source)
        self.assertIn("shutil.copy2(candidate, destination)", source)

    def test_gsm8k_minimal_plan_uses_its_required_main_config(self):
        requirements = requirements_from_plan(
            {
                "benchmark_targets": [
                    {
                        "hf_dataset": "openai/gsm8k",
                        "split": "test",
                        "task_type": "math_qa",
                    }
                ],
                "model_targets": [
                    {
                        "hf_model": "Qwen/Qwen2.5-0.5B-Instruct",
                        "backend": "transformers",
                        "requires_cuda": True,
                    }
                ],
                "metrics": {"primary": "exact_match"},
            }
        )

        self.assertEqual(requirements.dataset.config, "main")

    def test_four_bit_bundle_declares_all_conditional_runtime_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            materialize_runner_bundle(
                workdir=temp,
                preflight_row=_four_bit_preflight(),
                candidate_adapter_source=(
                    "CANDIDATE_METHOD = 'test'\n"
                    "def candidate_prompt(example, baseline):\n"
                    "    return baseline + ' test'\n"
                ),
            )
            dependencies = (Path(temp) / "code" / "requirements.txt").read_text().splitlines()
        self.assertIn("accelerate", dependencies)
        self.assertIn("bitsandbytes", dependencies)

    def test_cuda_bundle_declares_accelerate_without_quantization(self):
        preflight = _four_bit_preflight()
        requirements = json.loads(str(preflight["requirements_json"]))
        requirements["model"]["quantization"] = "none"
        preflight["requirements_json"] = json.dumps(requirements)
        with tempfile.TemporaryDirectory() as temp:
            materialize_runner_bundle(
                workdir=temp,
                preflight_row=preflight,
                candidate_adapter_source=(
                    "CANDIDATE_METHOD = 'test'\n"
                    "def candidate_prompt(example, baseline):\n"
                    "    return baseline + ' test'\n"
                ),
            )
            dependencies = (Path(temp) / "code" / "requirements.txt").read_text().splitlines()
        self.assertIn("accelerate", dependencies)
        self.assertNotIn("bitsandbytes", dependencies)

    def test_failed_colab_result_keeps_a_bounded_output_tail(self):
        payload = _result_payload(
            ColabExecutionResult(
                status="failed",
                returncode=2,
                stdout="a" * 6001,
                session="session",
                account_ref="account",
                gpu_type="T4",
                wall_seconds=1.0,
                artifact_manifest={},
                failure_reason="experiment_exit_2",
            )
        )
        self.assertEqual(payload["stdout_tail"], "a" * 6000)
        self.assertTrue(payload["stdout_truncated"])

    def test_terminal_colab_failure_marks_owning_run_terminal(self):
        row = {"experiment_run_id": 146, "agenda_id": 11, "idea_id": 105}
        with mock.patch.object(colab_worker.db, "execute") as execute, mock.patch.object(
            colab_worker.db, "commit"
        ) as commit:
            colab_worker._record_terminal_run_failure(
                row,
                SimpleNamespace(failure_reason="experiment_exit_2", returncode=2),
                SimpleNamespace(status="failed", failure_reason="experiment_exit_2"),
            )
        self.assertIn("SET status='failed'", execute.call_args.args[0])
        self.assertEqual(
            execute.call_args.args[1],
            ("colab_compute_failed:experiment_exit_2", 146, 11, 105),
        )
        commit.assert_called_once()

    def test_named_success_recovery_does_not_claim_or_submit_work(self):
        with mock.patch.object(colab_worker.db, "fetchone", return_value=None):
            self.assertFalse(
                colab_worker.recover_succeeded_run(
                    experiment_run_id=153, resource_grant_id=45
                )
            )


if __name__ == "__main__":
    unittest.main()
