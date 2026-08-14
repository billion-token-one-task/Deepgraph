from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

from meta_harness.failure_policy import (
    FailureContext,
    classify_failure,
    decide_recovery,
)
from meta_harness.runner_contract import (
    RunnerContractError,
    validate_final_results,
    verify_metric_from_artifacts,
)
from meta_harness.runner_materialization import (
    RunnerMaterializationError,
    materialize_runner_bundle,
)
from meta_harness.runners.generic_transformers import (
    GenericTransformersRunner,
    _load_candidate,
)


class RunnerContractTests(unittest.TestCase):
    def _artifacts(self, root: Path):
        raw = root / "raw_predictions.jsonl"
        rows = [
            {"method": "baseline", "prediction": "no", "target": "yes"},
            {"method": "baseline", "prediction": "yes", "target": "yes"},
            {"method": "candidate", "prediction": "yes", "target": "yes"},
            {"method": "candidate", "prediction": "no", "target": "no"},
        ]
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        for name in ("environment_manifest", "dataset_manifest", "model_manifest"):
            (root / f"{name}.json").write_text("{}", encoding="utf-8")
        payload = {
            "task_protocol": "sequence_classification",
            "dataset_id": "opaque/data",
            "dataset_revision": "a" * 40,
            "model_id": "opaque/model",
            "model_revision": "b" * 40,
            "seeds": [0],
            "num_examples": 2,
            "baseline_method": "baseline",
            "candidate_method": "candidate",
            "metric_name": "accuracy",
            "metric_direction": "higher",
            "metric_value": 1.0,
            "baseline_metric_value": 0.5,
            "best_metric_value": 1.0,
            "label_fallback_used": False,
            "per_method": {
                "baseline": {"accuracy": 0.5, "metric_value": 0.5},
                "candidate": {"accuracy": 1.0, "metric_value": 1.0},
            },
            "gpu_environment": {"available": True, "device_name": "test-gpu"},
            "artifacts": {
                "final_results": {"path": "final_results.json"},
                "raw_predictions": {"path": "raw_predictions.jsonl"},
                "environment_manifest": {"path": "environment_manifest.json"},
                "dataset_manifest": {"path": "dataset_manifest.json"},
                "model_manifest": {"path": "model_manifest.json"},
            },
            "artifact_hashes": {
                "raw_predictions": hashlib.sha256(raw.read_bytes()).hexdigest(),
            },
        }
        final = root / "final_results.json"
        final.write_text(json.dumps(payload), encoding="utf-8")
        return final, payload

    def test_metric_recomputes_from_raw_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            final, _payload = self._artifacts(Path(tmp))
            result = verify_metric_from_artifacts(final)
        self.assertEqual(result.baseline_value, 0.5)
        self.assertEqual(result.candidate_value, 1.0)
        self.assertEqual(result.recomputed_candidate, 1.0)

    def test_label_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _final, payload = self._artifacts(Path(tmp))
            payload["label_fallback_used"] = True
            with self.assertRaisesRegex(RunnerContractError, "label_fallback_forbidden"):
                validate_final_results(payload)


class FailurePolicyTests(unittest.TestCase):
    def _context(self, reason, retries=0):
        return FailureContext(
            reason_code=reason,
            detail="stable failure",
            code_hash="c" * 64,
            environment_hash="e" * 64,
            remaining_gpu_seconds=300,
            retry_count=retries,
        )

    def test_no_code_failure_never_invokes_llm_repair(self):
        for reason in (
            "network_transient",
            "model_download_timeout",
            "cuda_oom",
            "dataset_schema_mismatch",
            "dependency_missing",
            "metric_missing",
            "controller_lost",
        ):
            with self.subTest(reason=reason):
                decision = decide_recovery(self._context(reason), fingerprint_seen=False)
                self.assertFalse(decision.invoke_llm_repair)

    def test_same_fingerprint_is_not_executed_again(self):
        decision = decide_recovery(
            self._context("runner_contract_violation"),
            fingerprint_seen=True,
        )
        self.assertFalse(decision.retryable)
        self.assertEqual(decision.action, "defer_duplicate_fingerprint")

    def test_failure_fingerprint_ignores_volatile_diagnostic_prose(self):
        first = self._context("runner_contract_violation")
        second = FailureContext(
            reason_code=first.reason_code,
            detail="same failure at a different timestamp pid=9981",
            code_hash=first.code_hash,
            environment_hash=first.environment_hash,
            remaining_gpu_seconds=first.remaining_gpu_seconds,
        )
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_scientific_negative_is_an_outcome_not_a_crash(self):
        decision = decide_recovery(
            self._context("scientific_negative_result"),
            fingerprint_seen=False,
        )
        self.assertEqual(decision.action, "record_outcome")
        self.assertFalse(decision.retryable)

    def test_classifier_uses_stable_reason_codes(self):
        self.assertEqual(classify_failure(message="CUDA out of memory"), "cuda_oom")
        self.assertEqual(
            classify_failure(message="model download timeout"),
            "model_download_timeout",
        )
        self.assertEqual(
            classify_failure(message="", returncode=0, final_results_present=False),
            "metric_missing",
        )


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def manual_seed_all(_seed):
        return None

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def get_device_name(_index):
        return "contract-test-gpu"


class _FakeTorch:
    cuda = _FakeCuda()
    version = type("Version", (), {"cuda": "test"})()
    __version__ = "test"

    @staticmethod
    def manual_seed(_seed):
        return None


class _ContractRunner(GenericTransformersRunner):
    def __init__(self, *args, rows, **kwargs):
        super().__init__(*args, **kwargs)
        self._rows = rows

    def prepare(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_module, self.candidate_method, self.candidate_hook = _load_candidate(
            self.candidate_path, self.requirements.task_protocol
        )
        self.torch = _FakeTorch()

    def load_dataset(self):
        self.dataset_rows = list(self._rows)
        return self.dataset_rows

    def load_model(self):
        self.model = object()
        return self.model

    def _qa_prediction(self, prompt):
        return "correct" if str(prompt).startswith("candidate:") else "wrong"

    def _classification_prediction(self, text):
        return "1" if str(text).startswith("candidate:") else "0"


class GenericRunnerAndMaterializationTests(unittest.TestCase):
    def _requirements(self, protocol):
        if protocol == "generative_qa":
            return {
                "schema_version": "experiment_requirements_v1",
                "task_protocol": protocol,
                "candidate_hook": "candidate_prompt",
                "dataset": {
                    "repository_id": "opaque/qa-data",
                    "revision": "tag",
                    "split": "test",
                    "field_mapping": {"prompt": "question", "target": "answer"},
                },
                "model": {
                    "repository_id": "opaque/qa-model",
                    "revision": "tag",
                    "framework": "transformers",
                    "task": "causal_lm",
                    "requires_cuda": True,
                },
                "metric": {"name": "exact_match", "direction": "higher"},
                "dependencies": ["torch", "transformers", "datasets"],
                "seeds": [7],
                "sample_cap": 2,
                "artifact_contract": [
                    "final_results",
                    "raw_predictions",
                    "environment_manifest",
                    "dataset_manifest",
                    "model_manifest",
                ],
                "preferred_backends": ["ssh_gpu"],
            }
        return {
            "schema_version": "experiment_requirements_v1",
            "task_protocol": protocol,
            "candidate_hook": "candidate_text",
            "dataset": {
                "repository_id": "opaque/classification-data",
                "revision": "tag",
                "split": "test",
                "field_mapping": {"text": "sentence", "label": "class_id"},
            },
            "model": {
                "repository_id": "opaque/classification-model",
                "revision": "tag",
                "framework": "transformers",
                "task": "sequence_classification",
            },
            "metric": {"name": "accuracy", "direction": "higher"},
            "dependencies": ["torch", "transformers", "datasets"],
            "seeds": [11],
            "sample_cap": 2,
            "artifact_contract": [
                "final_results",
                "raw_predictions",
                "environment_manifest",
                "dataset_manifest",
                "model_manifest",
            ],
            "preferred_backends": ["cpu", "ssh_gpu"],
        }

    def _preflight(self, protocol):
        return {
            "id": 19,
            "status": "passed",
            "adapter_id": (
                "transformers_causal_lm_qa_v1"
                if protocol == "generative_qa"
                else "transformers_sequence_classification_v1"
            ),
            "adapter_version": "1.0.0",
            "dataset_revision": "a" * 40,
            "model_revision": "b" * 40,
            "requirements_json": json.dumps(self._requirements(protocol)),
        }

    def test_materialized_bundle_is_portable_and_revision_pinned(self):
        source = """
CANDIDATE_METHOD = "prefix_intervention"
def candidate_prompt(example, baseline_prompt):
    return "candidate:" + baseline_prompt
"""
        with tempfile.TemporaryDirectory() as tmp:
            bundle = materialize_runner_bundle(
                workdir=tmp,
                preflight_row=self._preflight("generative_qa"),
                candidate_adapter_source=source,
            )
            code = Path(tmp) / "code"
            config = json.loads((code / "execution_requirements.json").read_text())
            result = subprocess.run(
                [sys.executable, "train.py", "--help"],
                cwd=code,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(config["resolved_dataset_revision"], "a" * 40)
        self.assertEqual(config["resolved_model_revision"], "b" * 40)
        self.assertEqual(bundle["candidate_hook"], "candidate_prompt")

    def test_candidate_adapter_cannot_read_target(self):
        source = """
CANDIDATE_METHOD = "leaky"
def candidate_prompt(example, baseline_prompt):
    return baseline_prompt + str(example["answer"])
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                RunnerMaterializationError, "candidate_adapter_reads_target"
            ):
                materialize_runner_bundle(
                    workdir=tmp,
                    preflight_row=self._preflight("generative_qa"),
                    candidate_adapter_source=source,
                )

    def test_materializer_rejects_explicit_identity_adapter(self):
        source = """
CANDIDATE_METHOD = "identity"
def candidate_prompt(example, baseline_prompt):
    return baseline_prompt
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                RunnerMaterializationError, "candidate_adapter_identity"
            ):
                materialize_runner_bundle(
                    workdir=tmp,
                    preflight_row=self._preflight("generative_qa"),
                    candidate_adapter_source=source,
                )

    def test_materializer_rejects_wrong_hook_arity_before_compute(self):
        source = """
CANDIDATE_METHOD = "wrong_arity"
def candidate_prompt(prompt):
    return "candidate:" + prompt
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                RunnerMaterializationError, "candidate_hook_signature_invalid"
            ):
                materialize_runner_bundle(
                    workdir=tmp,
                    preflight_row=self._preflight("generative_qa"),
                    candidate_adapter_source=source,
                )

    def test_materializer_rejects_non_exact_hook_signature_before_compute(self):
        source = """
CANDIDATE_METHOD = "variadic_contract"
def candidate_prompt(example, baseline_prompt, *extra):
    return baseline_prompt
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                RunnerMaterializationError, "candidate_hook_signature_invalid"
            ):
                materialize_runner_bundle(
                    workdir=tmp,
                    preflight_row=self._preflight("generative_qa"),
                    candidate_adapter_source=source,
                )

    def test_materializer_rejects_async_hook_before_compute(self):
        source = """
CANDIDATE_METHOD = "async_contract"
async def candidate_prompt(example, baseline_prompt):
    return baseline_prompt + " candidate"
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                RunnerMaterializationError, "candidate_hook_missing:candidate_prompt"
            ):
                materialize_runner_bundle(
                    workdir=tmp,
                    preflight_row=self._preflight("generative_qa"),
                    candidate_adapter_source=source,
                )

    def test_materializer_rejects_positional_only_identity_adapter(self):
        source = """
CANDIDATE_METHOD = "posonly_identity"
def candidate_prompt(example, baseline_prompt, /):
    return baseline_prompt
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                RunnerMaterializationError, "candidate_adapter_identity"
            ):
                materialize_runner_bundle(
                    workdir=tmp,
                    preflight_row=self._preflight("generative_qa"),
                    candidate_adapter_source=source,
                )

    def test_materializer_rejects_decorated_hook_before_compute(self):
        source = """
CANDIDATE_METHOD = "decorated"
def wrapper(fn):
    return fn
@wrapper
def candidate_prompt(example, baseline_prompt):
    return baseline_prompt + " candidate"
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                RunnerMaterializationError, "candidate_hook_signature_invalid"
            ):
                materialize_runner_bundle(
                    workdir=tmp,
                    preflight_row=self._preflight("generative_qa"),
                    candidate_adapter_source=source,
                )

    def test_runner_rejects_runtime_all_sample_identity_adapter(self):
        adapter = """
CANDIDATE_METHOD = "aliased_identity"
def candidate_prompt(example, baseline_prompt):
    output = baseline_prompt + "   "
    return output
"""
        rows = [
            {"question": "q1", "answer": "correct"},
            {"question": "q2", "answer": "correct"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter_path = root / "candidate_adapter.py"
            adapter_path.write_text(adapter, encoding="utf-8")
            with self.assertRaisesRegex(
                RunnerContractError, "candidate_adapter_identity"
            ):
                _ContractRunner(
                    {
                        "requirements": self._requirements("generative_qa"),
                        "resolved_dataset_revision": "a" * 40,
                        "resolved_model_revision": "b" * 40,
                    },
                    candidate_adapter_path=adapter_path,
                    output_dir=root / "results",
                    rows=rows,
                ).run()

    def test_materializer_allows_selective_non_identity_adapter(self):
        source = """
CANDIDATE_METHOD = "selective_gate"
def candidate_prompt(example, baseline_prompt):
    if example.get("request_deliberation"):
        return baseline_prompt + "\\nVerify with an additional derivation."
    return baseline_prompt
"""
        with tempfile.TemporaryDirectory() as tmp:
            bundle = materialize_runner_bundle(
                workdir=tmp,
                preflight_row=self._preflight("generative_qa"),
                candidate_adapter_source=source,
            )

        self.assertEqual(bundle["candidate_hook"], "candidate_prompt")

    def test_two_protocols_emit_recomputable_real_metric_contract(self):
        cases = (
            (
                "generative_qa",
                "CANDIDATE_METHOD='qa_prefix'\n"
                "def candidate_prompt(example, baseline_prompt):\n"
                "    return 'candidate:' + baseline_prompt\n",
                [{"question": "q1", "answer": "correct"}, {"question": "q2", "answer": "correct"}],
            ),
            (
                "sequence_classification",
                "CANDIDATE_METHOD='classification_prefix'\n"
                "def candidate_text(example, baseline_text):\n"
                "    return 'candidate:' + baseline_text\n",
                [{"sentence": "x", "class_id": 1}, {"sentence": "y", "class_id": 1}],
            ),
        )
        for protocol, adapter, rows in cases:
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                adapter_path = root / "candidate_adapter.py"
                adapter_path.write_text(adapter, encoding="utf-8")
                config = {
                    "requirements": self._requirements(protocol),
                    "resolved_dataset_revision": "a" * 40,
                    "resolved_model_revision": "b" * 40,
                }
                result = _ContractRunner(
                    config,
                    candidate_adapter_path=adapter_path,
                    output_dir=root / "results",
                    rows=rows,
                ).run()
                verified = verify_metric_from_artifacts(root / "results" / "final_results.json")
                self.assertEqual(result["baseline_metric_value"], 0.0)
                self.assertEqual(result["best_metric_value"], 1.0)
                self.assertEqual(verified.recomputed_candidate, 1.0)
                self.assertFalse(result["label_fallback_used"])


if __name__ == "__main__":
    unittest.main()


class ContractFailureClassificationTests(unittest.TestCase):
    """A contract violation must not be classified as a missing metric.

    classify_failure returned metric_missing for anything that produced no
    final_results, and a run that breaks the runner or candidate-adapter
    contract dies long before it can write final_results. So every such
    failure took the metric_missing branch, which decide_recovery routes to a
    bare defer while runner_contract_violation routes to repair_code. Idea 105
    parked on that defer with the note "this failure does not require a code
    change" after its adapter was rejected for a contract breach.
    """

    ADAPTER_CONTRACT_FAILURES = (
        "capability_adapter_repair_failed:CapabilityScaffoldContractError:"
        "capability_adapter_repair_method_drift",
        "RunnerMaterializationError: candidate_hook_signature_invalid",
        "runner_contract_violation:candidate_adapter_required",
        "capability_scaffold_contract_missing:candidate_adapter_py",
    )

    def test_contract_failures_route_to_code_repair(self):
        for message in self.ADAPTER_CONTRACT_FAILURES:
            with self.subTest(message=message):
                reason = classify_failure(
                    message=message, returncode=None, final_results_present=False
                )
                self.assertEqual(reason, "runner_contract_violation")
                decision = decide_recovery(
                    FailureContext(
                        reason_code=reason,
                        detail="",
                        code_hash="code",
                        environment_hash="env",
                        remaining_gpu_seconds=7200.0,
                    ),
                    fingerprint_seen=False,
                )
                self.assertEqual(decision.action, "repair_code")
                self.assertTrue(decision.invoke_llm_repair)

    def test_a_genuinely_empty_run_is_still_metric_missing(self):
        self.assertEqual(
            classify_failure(
                message="process exited cleanly but produced no parsable output",
                returncode=0,
                final_results_present=False,
            ),
            "metric_missing",
        )
