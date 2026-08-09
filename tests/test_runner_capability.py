from __future__ import annotations

import unittest
from unittest import mock

from meta_harness.runner_capability import (
    DatasetRequirement,
    ExperimentRequirements,
    MetricRequirement,
    ModelRequirement,
    PreflightEngine,
    PreflightEnvironment,
    RepositoryMetadata,
    RunnerRegistry,
    requirements_from_plan,
)
from orchestrator import gpu_scheduler


class Probe:
    def __init__(self, *, datasets, models, dependencies=None):
        self.datasets = datasets
        self.models = models
        self.dependencies = set(dependencies or ("torch", "transformers", "datasets"))
        self.calls = []

    def dataset(self, repository_id, revision, config):
        self.calls.append(("dataset", repository_id, revision, config))
        return self.datasets.get(repository_id, RepositoryMetadata(False))

    def model(self, repository_id, revision):
        self.calls.append(("model", repository_id, revision))
        return self.models.get(repository_id, RepositoryMetadata(False))

    def dependency_available(self, name):
        self.calls.append(("dependency", name))
        return name in self.dependencies


ENVIRONMENT = PreflightEnvironment(
    enabled_backends=("ssh_gpu",),
    backend_vram_gb={"ssh_gpu": 40.0},
    network_available=True,
    disk_free_gb=100.0,
)


def qa_requirements(dataset="org/qa-corpus", model="org/generator"):
    return ExperimentRequirements(
        task_protocol="generative_qa",
        dataset=DatasetRequirement(
            repository_id=dataset,
            revision="stable",
            config="default",
            split="validation",
            field_mapping={"prompt": "query_text", "target": "gold_text"},
        ),
        model=ModelRequirement(
            repository_id=model,
            revision="release",
            task="causal_lm",
            min_vram_gb=8.0,
            requires_cuda=True,
        ),
        metric=MetricRequirement("exact_match", "higher"),
        candidate_hook="candidate_prompt",
        dependencies=("torch", "transformers", "datasets"),
        seeds=(3, 9),
        sample_cap=32,
        artifact_contract=(
            "final_results",
            "raw_predictions",
            "environment_manifest",
        ),
        preferred_backends=("ssh_gpu",),
    )


def classification_requirements():
    return ExperimentRequirements(
        task_protocol="sequence_classification",
        dataset=DatasetRequirement(
            repository_id="org/sentiment-corpus",
            revision="v2",
            split="test",
            field_mapping={"text": "sentence", "label": "class_id"},
        ),
        model=ModelRequirement(
            repository_id="org/classifier",
            revision="v4",
            task="sequence_classification",
            min_vram_gb=4.0,
            requires_cuda=True,
        ),
        metric=MetricRequirement("macro_f1", "higher"),
        candidate_hook="candidate_text",
        dependencies=("torch", "transformers", "datasets"),
        artifact_contract=(
            "final_results",
            "raw_predictions",
            "environment_manifest",
        ),
        preferred_backends=("ssh_gpu",),
    )


class RunnerCapabilityTests(unittest.TestCase):
    def test_registry_matches_two_protocols_without_repository_name_rules(self):
        registry = RunnerRegistry()
        qa = registry.matches(qa_requirements("opaque/a", "opaque/b"))
        classification = registry.matches(classification_requirements())

        self.assertEqual([item.adapter_id for item in qa], ["transformers_causal_lm_qa_v1"])
        self.assertEqual(
            [item.adapter_id for item in classification],
            ["transformers_sequence_classification_v1"],
        )

    def test_preflight_passes_real_metadata_and_resolves_revisions(self):
        probe = Probe(
            datasets={
                "org/qa-corpus": RepositoryMetadata(
                    True,
                    resolved_revision="a" * 40,
                    fields=("query_text", "gold_text", "id"),
                )
            },
            models={
                "org/generator": RepositoryMetadata(
                    True,
                    resolved_revision="b" * 40,
                    task="text_generation",
                    size_gb=6.0,
                )
            },
        )
        result = PreflightEngine(probe=probe).run(qa_requirements(), ENVIRONMENT)

        self.assertTrue(result.passed)
        self.assertEqual(result.adapter_id, "transformers_causal_lm_qa_v1")
        self.assertEqual(result.selected_backend, "ssh_gpu")
        self.assertEqual(result.dataset_revision, "a" * 40)
        self.assertEqual(result.model_revision, "b" * 40)

    def test_schema_mismatch_is_deferred_and_never_calls_compute(self):
        probe = Probe(
            datasets={
                "org/qa-corpus": RepositoryMetadata(
                    True,
                    resolved_revision="a" * 40,
                    fields=("unrelated",),
                )
            },
            models={
                "org/generator": RepositoryMetadata(
                    True,
                    resolved_revision="b" * 40,
                    task="text_generation",
                )
            },
        )
        result = PreflightEngine(probe=probe).run(qa_requirements(), ENVIRONMENT)

        self.assertEqual(result.status, "deferred")
        self.assertIn("dataset_schema_mismatch", result.reason_codes)
        self.assertFalse(any(call[0] == "gpu" for call in probe.calls))

    def test_missing_dependency_and_small_vram_are_structured(self):
        probe = Probe(
            datasets={
                "org/sentiment-corpus": RepositoryMetadata(
                    True,
                    resolved_revision="c" * 40,
                    fields=("sentence", "class_id"),
                )
            },
            models={
                "org/classifier": RepositoryMetadata(
                    True,
                    resolved_revision="d" * 40,
                    task="sequence_classification",
                )
            },
            dependencies=("torch", "datasets"),
        )
        environment = PreflightEnvironment(
            enabled_backends=("ssh_gpu",),
            backend_vram_gb={"ssh_gpu": 1.0},
            network_available=False,
            disk_free_gb=100.0,
        )
        result = PreflightEngine(probe=probe).run(
            classification_requirements(), environment
        )

        self.assertEqual(result.status, "deferred")
        self.assertIn("dependency_missing", result.reason_codes)
        self.assertIn("vram_insufficient", result.reason_codes)

    def test_plan_translation_uses_protocol_fields_not_dataset_identity(self):
        plan = {
            "benchmark_targets": [
                {
                    "hf_dataset": "arbitrary-owner/arbitrary-data",
                    "revision": "dataset-tag",
                    "config": "subset",
                    "split": "holdout",
                    "task_type": "classification",
                    "text_field": "body",
                    "label_field": "category",
                }
            ],
            "model_targets": [
                {
                    "hf_model": "another-owner/arbitrary-model",
                    "revision": "model-tag",
                    "backend": "transformers",
                    "task": "sequence_classification",
                    "requires_cuda": True,
                }
            ],
            "metrics": {"primary": "accuracy", "direction": "higher"},
            "minimum_seeds": 2,
        }
        requirements = requirements_from_plan(plan)

        self.assertEqual(requirements.task_protocol, "sequence_classification")
        self.assertEqual(
            requirements.dataset.field_mapping,
            {"text": "body", "label": "category"},
        )
        self.assertEqual(requirements.model.task, "sequence_classification")
        self.assertEqual(requirements.seeds, (0, 1))


class ComputePreflightGuardTests(unittest.TestCase):
    def test_production_guard_requires_passed_revision_bound_adapter(self):
        run = {"agenda_id": 2, "deep_insight_id": 3, "resource_grant_id": 5}
        with (
            mock.patch.object(gpu_scheduler.db, "_use_pg", return_value=True),
            mock.patch.object(
                gpu_scheduler.db,
                "fetchone",
                return_value={
                    "preflight_result_id": 7,
                    "status": "passed",
                    "adapter_id": "transformers.generative_qa.v1",
                    "dataset_revision": "a" * 40,
                    "model_revision": "b" * 40,
                },
            ),
        ):
            self.assertIsNone(gpu_scheduler._capability_preflight_blocker(run))

    def test_production_guard_quarantines_every_unbound_legacy_run(self):
        run = {"agenda_id": 2, "deep_insight_id": 3, "resource_grant_id": 5}
        with (
            mock.patch.object(gpu_scheduler.db, "_use_pg", return_value=True),
            mock.patch.object(gpu_scheduler.db, "fetchone", return_value=None),
        ):
            reason = gpu_scheduler._capability_preflight_blocker(run)
        self.assertIn("lacks a capability preflight", reason)


if __name__ == "__main__":
    unittest.main()
