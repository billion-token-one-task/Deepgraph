from __future__ import annotations

import unittest
from unittest import mock

from meta_harness.runner_capability import (
    CapabilityContractError,
    DatasetRequirement,
    ExperimentRequirements,
    MetricRequirement,
    ModelRequirement,
    PreflightEngine,
    PreflightEnvironment,
    RepositoryMetadata,
    RunnerRegistry,
    requirements_from_plan,
    validate_explicit_requirements_alignment,
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


class ExplicitRequirementsAlignmentTests(unittest.TestCase):
    def _plan(self):
        requirements = classification_requirements()
        return {
            "datasets": [
                {
                    "name": "Named benchmark",
                    "repository_id": requirements.dataset.repository_id,
                }
            ],
            "model_targets": [
                {"repository_id": requirements.model.repository_id}
            ],
            "metrics": {"primary": requirements.metric.name},
            "execution_requirements": requirements.to_dict(),
        }, requirements

    def test_explicit_contract_must_bind_the_same_scientific_identities(self):
        plan, requirements = self._plan()
        validate_explicit_requirements_alignment(plan, requirements)

    def test_unrelated_generic_contract_is_refused_before_preflight(self):
        plan, requirements = self._plan()
        plan["datasets"] = [
            {"name": "ScanNet Multi-View Pair/Triplet Benchmark"}
        ]
        plan["model_targets"] = []
        plan["metrics"] = {"primary": "Inlier classification F1-Score"}

        with self.assertRaisesRegex(
            CapabilityContractError,
            "execution_dataset_identity_unbound",
        ):
            validate_explicit_requirements_alignment(plan, requirements)

    def test_dataset_model_and_metric_drift_each_fail_closed(self):
        for mutation, reason in (
            (
                lambda plan: plan["datasets"][0].update(
                    repository_id="other/dataset"
                ),
                "execution_dataset_identity_mismatch",
            ),
            (
                lambda plan: plan["model_targets"][0].update(
                    repository_id="other/model"
                ),
                "execution_model_identity_mismatch",
            ),
            (
                lambda plan: plan["metrics"].update(primary="accuracy"),
                "execution_metric_identity_mismatch",
            ),
        ):
            with self.subTest(reason=reason):
                plan, requirements = self._plan()
                mutation(plan)
                with self.assertRaisesRegex(CapabilityContractError, reason):
                    validate_explicit_requirements_alignment(plan, requirements)


    def test_identities_written_the_way_a_plan_generator_writes_them(self):
        """A well-formed plan was refused for want of fields nobody writes.

        The generator names the dataset in ``datasets[].name`` with the config
        in parentheses, puts the model on ``baselines[].model``, and follows the
        metric with a parenthetical gloss. Every identity is present and
        correct; only the reader was looking elsewhere.
        """

        requirements = classification_requirements()
        plan = {
            "datasets": [
                {
                    "name": (
                        f"{requirements.dataset.repository_id} (subset-a)"
                    ),
                    "split": "validation: 100 samples",
                }
            ],
            "baselines": [
                {
                    "name": "Zero-Shot Direct Query",
                    "model": requirements.model.repository_id,
                }
            ],
            "metrics": {
                "primary": (
                    f"{requirements.metric.name} (parsed from the terminal"
                    " output token sequence)"
                )
            },
            "execution_requirements": requirements.to_dict(),
        }

        validate_explicit_requirements_alignment(plan, requirements)

    def test_prose_is_never_promoted_to_an_identity(self):
        """Reading the prose keys must not let free text bind a run."""

        requirements = classification_requirements()
        base = {
            "baselines": [{"model": requirements.model.repository_id}],
            "metrics": {"primary": requirements.metric.name},
            "execution_requirements": requirements.to_dict(),
        }
        for label, datasets in (
            ("bare prose", [{"name": "Zero-Shot Direct Query"}]),
            ("prose with a config", [{"name": "our internal set (hard)"}]),
        ):
            with self.subTest(label=label):
                plan = dict(base, datasets=datasets)
                with self.assertRaisesRegex(
                    CapabilityContractError,
                    "execution_dataset_identity_unbound",
                ):
                    validate_explicit_requirements_alignment(plan, requirements)


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


class MetricVocabularyTests(unittest.TestCase):
    """A candidate inside a runner's capabilities must not be refused over the
    spelling of its metric.

    Agenda 11 / idea 110 declared `exact_match_accuracy` on gsm8k with a
    causal_lm model - fully inside `transformers_causal_lm_qa_v1` - and was
    deferred because the registry spells the same measurement `exact_match`.
    """

    def test_metric_synonym_is_folded_onto_the_registry_vocabulary(self):
        plan = dict(
            qa_requirements().to_dict(),
            metric={"name": "exact_match_accuracy", "direction": "higher",
                    "required_prediction_fields": ["prediction", "target"]},
        )
        requirements = ExperimentRequirements.from_dict(plan)
        self.assertEqual(requirements.metric.name, "exact_match")
        self.assertEqual(
            [item.adapter_id for item in RunnerRegistry().matches(requirements)],
            ["transformers_causal_lm_qa_v1"],
        )

    def test_a_different_measurement_is_not_folded(self):
        """Only exact synonyms may be normalized; a real gap must still fail."""

        plan = dict(
            qa_requirements().to_dict(),
            metric={"name": "pass_at_1", "direction": "higher",
                    "required_prediction_fields": ["prediction", "target"]},
        )
        requirements = ExperimentRequirements.from_dict(plan)
        self.assertEqual(requirements.metric.name, "pass_at_1")
        self.assertEqual(RunnerRegistry().matches(requirements), ())


class PreflightDiagnosticsTests(unittest.TestCase):
    """Deferral reasons must point at the adapter that nearly matched.

    Unioning blockers across every adapter made a one-field metric mismatch
    look like five independent capability gaps - including
    `unsupported_task_protocol` for a protocol that did match - and sent a
    previous debugging pass after a runner-coverage problem that did not exist.
    """

    def test_deferral_reports_the_nearest_adapter_not_the_union(self):
        requirements = ExperimentRequirements.from_dict(
            dict(
                qa_requirements().to_dict(),
                metric={"name": "bleu", "direction": "higher",
                        "required_prediction_fields": ["prediction", "target"]},
            )
        )
        engine = PreflightEngine(
            registry=RunnerRegistry(),
            probe=Probe(datasets={}, models={}),
        )
        result = engine.run(requirements, ENVIRONMENT)

        self.assertEqual(result.reason_codes, ("metric_contract_unsupported",))
        self.assertEqual(result.checks["nearest_adapter"], "transformers_causal_lm_qa_v1")
        self.assertNotIn("unsupported_task_protocol", result.reason_codes)
        # The full picture stays available for auditing, just not as the verdict.
        self.assertIn(
            "unsupported_task_protocol",
            result.checks["adapter_blockers"]["transformers_sequence_classification_v1"],
        )


if __name__ == "__main__":
    unittest.main()


class PlanMetricAliasTests(unittest.TestCase):
    """A sanctioned synonym must survive both requirement construction paths.

    Idea 110 sat deferred on ``metric_contract_unsupported`` for days with a
    plan the causal-LM runner could otherwise execute: the alias table folded
    ``exact_match_accuracy`` when requirements were loaded from a stored row,
    but ``requirements_from_plan`` passed the raw spelling straight to the
    registry.
    """

    PLAN = {
        "benchmark_targets": [
            {
                "task_protocol": "generative_qa",
                "hf_dataset": "openai/gsm8k",
                "revision": "main",
                "split": "test",
                "question_field": "question",
                "answer_field": "answer",
            }
        ],
        "model_targets": [
            {
                "hf_model": "Qwen/Qwen2.5-Math-7B-Instruct",
                "revision": "main",
                "backend": "transformers",
                "task": "causal_lm",
            }
        ],
        "metrics": {"primary": "exact_match_accuracy"},
        "minimum_seeds": 1,
    }

    def test_plan_metric_synonym_is_folded_onto_the_registry_vocabulary(self):
        # A non-numeric dataset keeps this test about alias folding alone;
        # gsm8k would additionally trip the numeric-answer measurement floor
        # (tests/test_measurement_floors.py covers that mapping).
        plan = dict(self.PLAN)
        plan["benchmark_targets"] = [
            dict(self.PLAN["benchmark_targets"][0], hf_dataset="example/free-text-qa")
        ]
        requirements = requirements_from_plan(plan)
        self.assertEqual(requirements.metric.name, "exact_match")
        matches = RunnerRegistry().matches(requirements)
        self.assertTrue(
            matches, "a plan inside the runner's capabilities must find a runner"
        )

    def test_unknown_metric_still_reports_a_real_capability_gap(self):
        plan = dict(self.PLAN, metrics={"primary": "bleu"})
        requirements = requirements_from_plan(plan)
        self.assertEqual(requirements.metric.name, "bleu")
        self.assertIn(
            "metric_contract_unsupported",
            RunnerRegistry().all()[0].structural_blockers(requirements),
        )


class SampleCapDefaultTests(unittest.TestCase):
    """An unstated evaluation size must not authorise the whole split.

    sample_cap fell back to 0, which the runner reads as "evaluate everything".
    Idea 105's plan never named max_eval_examples, so its materialized bundle
    authorised all 1319 GSM8K test rows across three seeds and two methods --
    thousands of generations and hours of a scarce accelerator, from an
    omission rather than a decision.
    """

    PLAN = {
        "benchmark_targets": [
            {
                "task_protocol": "generative_qa",
                "hf_dataset": "openai/gsm8k",
                "revision": "main",
                "split": "test",
                "question_field": "question",
                "answer_field": "answer",
            }
        ],
        "model_targets": [
            {
                "hf_model": "Qwen/Qwen2.5-Math-7B-Instruct",
                "revision": "main",
                "backend": "transformers",
                "task": "causal_lm",
            }
        ],
        "metrics": {"primary": "exact_match"},
        "minimum_seeds": 1,
    }

    def test_an_unstated_size_falls_back_to_the_configured_ceiling(self):
        with mock.patch.dict(
            "sys.modules",
            {"config": mock.Mock(EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES=16)},
        ):
            requirements = requirements_from_plan(self.PLAN)
        self.assertEqual(requirements.sample_cap, 16)

    def test_an_explicit_size_still_wins(self):
        plan = dict(self.PLAN, max_eval_examples=250)
        with mock.patch.dict(
            "sys.modules",
            {"config": mock.Mock(EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES=16)},
        ):
            requirements = requirements_from_plan(plan)
        self.assertEqual(requirements.sample_cap, 250)


class TheHubAndTheContractNameTheSameHead(unittest.TestCase):
    """One of V1's two runners could never accept a model.

    Hugging Face publishes a classifier's pipeline_tag as
    "text-classification"; the runner contract calls that head
    "sequence_classification". With no synonym between them the contract was
    unsatisfiable in both directions -- declare the contract's spelling and the
    remote check refused it against the hub's, declare the hub's and the
    structural check refused it against the runner's. Measured 2026-08-27
    against three published classifiers, every one of which reports
    text_classification.
    """

    def _engine_and_env(self, published_task):
        probe = Probe(
            datasets={
                "org/sentiment-corpus": RepositoryMetadata(
                    True, resolved_revision="sha", fields=("sentence", "class_id")
                )
            },
            models={
                "org/classifier": RepositoryMetadata(
                    True, resolved_revision="sha", task=published_task, size_gb=0.5
                )
            },
        )
        return PreflightEngine(probe=probe), ENVIRONMENT

    def test_a_published_classifier_is_accepted(self):
        engine, env = self._engine_and_env("text_classification")
        result = engine.run(classification_requirements(), env)
        self.assertEqual(result.reason_codes, ())
        self.assertTrue(result.passed, result.reason_codes)

    def test_declaring_the_hubs_spelling_is_also_accepted(self):
        requirements = requirements_from_plan(
            {
                "datasets": [{"name": "org/sentiment-corpus"}],
                "baselines": [{"model": "org/classifier"}],
                "metrics": {"primary": "macro_f1"},
                "execution_requirements": {
                    "task_protocol": "sequence_classification",
                    "candidate_hook": "candidate_text",
                    "dataset": {
                        "repository_id": "org/sentiment-corpus",
                        "revision": "v2",
                        "split": "test",
                        "field_mapping": {"text": "sentence", "label": "class_id"},
                    },
                    "model": {
                        "repository_id": "org/classifier",
                        "revision": "v4",
                        "task": "text-classification",
                        "min_vram_gb": 4.0,
                    },
                    "metric": {"name": "macro_f1", "direction": "higher"},
                    "preferred_backends": ["ssh_gpu"],
                },
            }
        )
        self.assertEqual(requirements.model.task, "sequence_classification")
        self.assertTrue(RunnerRegistry().matches(requirements))

    def test_a_genuinely_different_head_is_still_refused(self):
        for published in ("token_classification", "fill_mask", "zero_shot_classification"):
            with self.subTest(published=published):
                engine, env = self._engine_and_env(published)
                result = engine.run(classification_requirements(), env)
                self.assertIn("model_task_mismatch", result.reason_codes)

    def test_the_generative_synonym_still_holds(self):
        probe = Probe(
            datasets={
                "org/qa-corpus": RepositoryMetadata(
                    True, resolved_revision="sha", fields=("query_text", "gold_text")
                )
            },
            models={
                "org/generator": RepositoryMetadata(
                    True, resolved_revision="sha", task="text_generation", size_gb=1.0
                )
            },
        )
        result = PreflightEngine(probe=probe).run(qa_requirements(), ENVIRONMENT)
        self.assertNotIn("model_task_mismatch", result.reason_codes)
