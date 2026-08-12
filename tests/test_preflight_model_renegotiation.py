from __future__ import annotations

import copy
import json
from unittest import mock

from agents import experiment_forge
from meta_harness import preflight_repository
from meta_harness.runner_capability import (
    PreflightEngine,
    PreflightEnvironment,
    PreflightResult,
)


ENVIRONMENT = PreflightEnvironment(
    enabled_backends=("ssh_gpu",),
    backend_vram_gb={"ssh_gpu": 40.0},
    network_available=True,
    disk_free_gb=100.0,
)


def _qa_plan(model: str) -> dict:
    return {
        "benchmark_targets": [
            {
                "name": "GSM8K",
                "hf_dataset": "openai/gsm8k",
                "config": "main",
                "split": "test",
                "task_type": "math_qa",
                "question_field": "question",
                "answer_field": "answer",
                "max_eval_examples": 16,
            }
        ],
        "model_targets": [
            {
                "name": model,
                "hf_model": model,
                "backend": "transformers",
                "role": "candidate_base_model",
                "load_in_4bit": True,
                "requires_cuda": True,
            }
        ],
        "metrics": {"primary": "exact_match", "direction": "higher"},
        "minimum_seeds": 1,
    }


def _resolved_probe_insight(old_model: str) -> dict:
    parsed = {
        "id": 105,
        "agenda_id": 11,
        "title": "Math reasoning probe",
        "problem_statement": "Test a reasoning method on GSM8K.",
        "resource_class": "gpu_large",
        "proposed_method": {
            "name": "ReasoningCandidate",
            "type": "reasoning",
            "definition": "Apply a bounded reasoning transformation.",
        },
        "evidence_plan": {},
    }
    plan = {
        "benchmark_design_status": "resolved",
        "benchmark_design_contract": {
            "status": "resolved",
            "source": "test-reviewed-contract",
        },
        "benchmark_targets": [
            {
                "name": "GSM8K",
                "hf_dataset": "openai/gsm8k",
                "config": "main",
                "split": "test",
                "task_type": "math_qa",
                "question_field": "question",
                "answer_field": "answer",
                "generated_runner_supported": True,
                "benchmark_role": "executable_probe",
                "formal_target_deferred": False,
            }
        ],
        "datasets": [{"name": "GSM8K"}],
        "baselines": [{"name": "Direct"}, {"name": "Always-CoT"}],
        "metrics": {"primary": "exact_match", "direction": "higher"},
        "minimum_seeds": 1,
    }
    with mock.patch.object(
        experiment_forge,
        "EXPERIMENT_REAL_LLM_MODEL",
        old_model,
    ):
        plan = experiment_forge._ensure_real_benchmark_plan(
            parsed,
            parsed["proposed_method"],
            plan,
            "gpu_large",
            resolve_datasets=False,
        )
        publication = experiment_forge._publication_evidence_contract(
            {**parsed, "experimental_plan": plan},
            plan,
            evidence_plan={},
            scaffold_kind="planned",
        )
    plan["publication_evidence_contract"] = publication
    plan["paper_intent"] = publication.get("paper_intent", {})
    return {**parsed, "experimental_plan": json.dumps(plan)}


def _mismatch() -> PreflightResult:
    return PreflightResult(
        status="deferred",
        reason_codes=("model_task_mismatch",),
        checks={"model_task": "image_text_to_text"},
        adapter_id="transformers_causal_lm_qa_v1",
        adapter_version="1.0.0",
        dataset_revision="dataset-revision",
        model_revision="vlm-revision",
    )


def _passed() -> PreflightResult:
    return PreflightResult(
        status="passed",
        reason_codes=(),
        checks={"model_task": "text_generation"},
        adapter_id="transformers_causal_lm_qa_v1",
        adapter_version="1.0.0",
        selected_backend="ssh_gpu",
        dataset_revision="dataset-revision",
        model_revision="text-model-revision",
    )


def test_stale_system_model_is_rederived_without_llm_and_contracts_stay_consistent():
    old_model = "Qwen/Qwen3.5-4B"
    replacement = "Qwen/Qwen3-4B-Instruct-2507"
    insight = _resolved_probe_insight(old_model)

    with (
        mock.patch.object(
            experiment_forge,
            "EXPERIMENT_REAL_LLM_MODEL",
            replacement,
        ),
        mock.patch.object(
            experiment_forge,
            "build_benchmark_design_contract",
            side_effect=AssertionError("resolved design must not invoke an LLM"),
        ),
        mock.patch.object(
            experiment_forge,
            "call_llm_json_for_role",
            side_effect=AssertionError("model renegotiation must not invoke an LLM"),
        ),
    ):
        candidate = experiment_forge.renegotiate_stale_model_requirement(
            insight,
            reason_codes=("model_task_mismatch",),
            checks={"model_task": "image_text_to_text"},
        )

    assert candidate is not None
    plan = candidate["plan"]
    assert [item["hf_model"] for item in plan["model_targets"]] == [replacement]
    assert plan["benchmark_protocol"]["model_policy"]["required_models"] == [replacement]
    assert plan["benchmark_protocol"]["full_benchmark_requirements"]["required_model_names"] == [replacement]
    assert plan["publication_evidence_contract"]["required_models"] == [replacement]
    assert plan["benchmark_execution"]["default_model"] == replacement
    history = plan["model_requirement_negotiation_history"][-1]
    assert history["previous_model_targets"][0]["hf_model"] == old_model
    assert history["replacement_model_targets"][0]["hf_model"] == replacement


def test_model_renegotiation_rejects_aggregate_reasons_and_explicit_requirements():
    insight = _resolved_probe_insight("Qwen/Qwen3.5-4B")
    with mock.patch.object(
        experiment_forge,
        "EXPERIMENT_REAL_LLM_MODEL",
        "Qwen/Qwen3-4B-Instruct-2507",
    ):
        assert experiment_forge.renegotiate_stale_model_requirement(
            insight,
            reason_codes=("model_task_mismatch", "disk_insufficient"),
            checks={"model_task": "image_text_to_text"},
        ) is None

        explicit = copy.deepcopy(insight)
        plan = json.loads(explicit["experimental_plan"])
        plan["execution_requirements"] = _qa_plan("owner/explicit-model")
        explicit["experimental_plan"] = json.dumps(plan)
        assert experiment_forge.renegotiate_stale_model_requirement(
            explicit,
            reason_codes=("model_task_mismatch",),
            checks={"model_task": "image_text_to_text"},
        ) is None


def test_repository_persists_only_after_replacement_passes_same_predicate():
    old_plan = _qa_plan("Qwen/Qwen3.5-4B")
    revised_plan = _qa_plan("Qwen/Qwen3-4B-Instruct-2507")
    raw_plan = json.dumps(old_plan)
    row = {"id": 105, "agenda_id": 11, "experimental_plan": raw_plan}
    engine = mock.Mock(spec=PreflightEngine)
    engine.run.side_effect = [_mismatch(), _passed()]
    cursor = mock.Mock(rowcount=1)
    repo = preflight_repository.CandidatePreflightRepository()
    candidate = {
        "status": "candidate",
        "plan": revised_plan,
        "previous_model": "Qwen/Qwen3.5-4B",
        "replacement_model": "Qwen/Qwen3-4B-Instruct-2507",
        "observed_model_task": "image_text_to_text",
    }

    with (
        mock.patch.object(preflight_repository.db, "fetchone", return_value=row),
        mock.patch.object(preflight_repository.db, "execute", return_value=cursor) as execute,
        mock.patch.object(preflight_repository.db, "commit"),
        mock.patch.object(preflight_repository.db, "rollback"),
        mock.patch.object(repo, "declare", side_effect=[7, 8]) as declare,
        mock.patch.object(repo, "record", side_effect=[15, 16]) as record,
        mock.patch.object(
            experiment_forge,
            "renegotiate_stale_model_requirement",
            return_value=candidate,
        ),
    ):
        result = repo.run_candidate(
            agenda_id=11,
            idea_id=105,
            engine=engine,
            environment=ENVIRONMENT,
        )

    assert result.passed
    assert result.preflight_result_id == 16
    assert result.checks["model_requirement_renegotiated"]["previous_model"] == "Qwen/Qwen3.5-4B"
    assert result.checks["model_requirement_renegotiated"]["replacement_model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert [call.args[0].model.repository_id for call in engine.run.call_args_list] == [
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3-4B-Instruct-2507",
    ]
    assert declare.call_count == 2
    assert record.call_count == 2
    assert execute.call_count == 1
    assert execute.call_args.args[1][1:] == (105, 11, raw_plan)


def test_repository_keeps_original_plan_when_replacement_still_defers():
    old_plan = _qa_plan("Qwen/Qwen3.5-4B")
    revised_plan = _qa_plan("Qwen/Qwen3-4B-Instruct-2507")
    row = {"id": 105, "agenda_id": 11, "experimental_plan": json.dumps(old_plan)}
    engine = mock.Mock(spec=PreflightEngine)
    engine.run.side_effect = [
        _mismatch(),
        PreflightResult(
            status="deferred",
            reason_codes=("model_unavailable",),
            checks={"model_task": ""},
        ),
    ]
    repo = preflight_repository.CandidatePreflightRepository()
    candidate = {
        "status": "candidate",
        "plan": revised_plan,
        "previous_model": "Qwen/Qwen3.5-4B",
        "replacement_model": "Qwen/Qwen3-4B-Instruct-2507",
        "observed_model_task": "image_text_to_text",
    }

    with (
        mock.patch.object(preflight_repository.db, "fetchone", return_value=row),
        mock.patch.object(preflight_repository.db, "execute") as execute,
        mock.patch.object(preflight_repository.db, "commit"),
        mock.patch.object(repo, "declare", return_value=7) as declare,
        mock.patch.object(repo, "record", return_value=15) as record,
        mock.patch.object(
            experiment_forge,
            "renegotiate_stale_model_requirement",
            return_value=candidate,
        ),
    ):
        result = repo.run_candidate(
            agenda_id=11,
            idea_id=105,
            engine=engine,
            environment=ENVIRONMENT,
        )

    assert result.status == "deferred"
    assert result.preflight_result_id == 15
    assert declare.call_count == 1
    assert record.call_count == 1
    execute.assert_not_called()


def test_repository_fails_closed_on_concurrent_plan_change():
    old_plan = _qa_plan("Qwen/Qwen3.5-4B")
    revised_plan = _qa_plan("Qwen/Qwen3-4B-Instruct-2507")
    row = {"id": 105, "agenda_id": 11, "experimental_plan": json.dumps(old_plan)}
    engine = mock.Mock(spec=PreflightEngine)
    engine.run.side_effect = [_mismatch(), _passed()]
    repo = preflight_repository.CandidatePreflightRepository()
    candidate = {
        "status": "candidate",
        "plan": revised_plan,
        "previous_model": "Qwen/Qwen3.5-4B",
        "replacement_model": "Qwen/Qwen3-4B-Instruct-2507",
        "observed_model_task": "image_text_to_text",
    }

    with (
        mock.patch.object(preflight_repository.db, "fetchone", return_value=row),
        mock.patch.object(
            preflight_repository.db,
            "execute",
            return_value=mock.Mock(rowcount=0),
        ),
        mock.patch.object(preflight_repository.db, "commit"),
        mock.patch.object(preflight_repository.db, "rollback") as rollback,
        mock.patch.object(repo, "declare", return_value=7) as declare,
        mock.patch.object(repo, "record", return_value=15) as record,
        mock.patch.object(
            experiment_forge,
            "renegotiate_stale_model_requirement",
            return_value=candidate,
        ),
    ):
        result = repo.run_candidate(
            agenda_id=11,
            idea_id=105,
            engine=engine,
            environment=ENVIRONMENT,
        )

    assert result.status == "deferred"
    assert result.preflight_result_id == 15
    assert declare.call_count == 1
    assert record.call_count == 1
    rollback.assert_called_once()


def test_repository_returns_original_result_when_renegotiator_errors():
    old_plan = _qa_plan("Qwen/Qwen3.5-4B")
    row = {"id": 105, "agenda_id": 11, "experimental_plan": json.dumps(old_plan)}
    engine = mock.Mock(spec=PreflightEngine)
    engine.run.return_value = _mismatch()
    repo = preflight_repository.CandidatePreflightRepository()

    with (
        mock.patch.object(preflight_repository.db, "fetchone", return_value=row),
        mock.patch.object(preflight_repository.db, "execute") as execute,
        mock.patch.object(preflight_repository.db, "commit"),
        mock.patch.object(repo, "declare", return_value=7),
        mock.patch.object(repo, "record", return_value=15),
        mock.patch.object(
            experiment_forge,
            "renegotiate_stale_model_requirement",
            side_effect=ValueError("malformed historical plan"),
        ),
    ):
        result = repo.run_candidate(
            agenda_id=11,
            idea_id=105,
            engine=engine,
            environment=ENVIRONMENT,
        )

    assert result.status == "deferred"
    assert result.preflight_result_id == 15
    execute.assert_not_called()
