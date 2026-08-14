"""Structured runner capabilities and grant-before-compute preflight.

All matching is performed on declared protocols, schema roles, model tasks,
metrics, resources, and artifact contracts.  Dataset and model repository names
are opaque identifiers and never participate in authorization decisions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


PREFLIGHT_PASSED = "passed"
PREFLIGHT_DEFERRED = "deferred"
PREFLIGHT_FAILED = "failed"


class CapabilityContractError(ValueError):
    pass


# Plan generators and the runner registry grew their metric vocabularies
# independently, so a candidate that is fully inside a runner's capabilities
# gets refused over spelling alone. Only exact synonyms belong here: each key
# must denote the *same* measurement as its value, never a related one. A
# rename that changes what is measured (pass@k, bleu, rouge, ...) is a real
# capability gap and must keep failing preflight.
METRIC_NAME_ALIASES: Mapping[str, str] = {
    "acc": "accuracy",
    "accuracy_score": "accuracy",
    "em": "exact_match",
    "exact_match_accuracy": "exact_match",
    "exact_match_rate": "exact_match",
    "exact_match_score": "exact_match",
    "f1_macro": "macro_f1",
    "f1_score": "f1",
    "macro_f1_score": "macro_f1",
    "numeric_accuracy_score": "numeric_accuracy",
}


def canonical_metric_name(name: str) -> str:
    """Fold a metric name onto the registry's vocabulary. Unknown names pass
    through untouched so a genuine capability gap still surfaces."""

    normalized = str(name or "").strip().lower()
    return METRIC_NAME_ALIASES.get(normalized, normalized)


@dataclass(frozen=True)
class DatasetRequirement:
    repository_id: str
    revision: str = "main"
    config: str = ""
    split: str = "test"
    field_mapping: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequirement:
    repository_id: str
    revision: str = "main"
    framework: str = "transformers"
    task: str = "causal_lm"
    min_vram_gb: float = 0.0
    requires_cuda: bool = False
    quantization: str = "none"


@dataclass(frozen=True)
class MetricRequirement:
    name: str
    direction: str = "higher"
    required_prediction_fields: tuple[str, ...] = ("prediction", "target")


@dataclass(frozen=True)
class ExperimentRequirements:
    task_protocol: str
    dataset: DatasetRequirement
    model: ModelRequirement
    metric: MetricRequirement
    candidate_hook: str = ""
    dependencies: tuple[str, ...] = ()
    network_required: bool = True
    min_disk_gb: float = 1.0
    seeds: tuple[int, ...] = (0,)
    sample_cap: int | None = None
    artifact_contract: tuple[str, ...] = (
        "final_results",
        "raw_predictions",
        "environment_manifest",
    )
    preferred_backends: tuple[str, ...] = (
        "ssh_gpu",
        "local_gpu",
        "colab_gpu",
    )
    schema_version: str = "experiment_requirements_v1"

    def validate(self) -> None:
        if self.schema_version != "experiment_requirements_v1":
            raise CapabilityContractError("requirements_schema_unsupported")
        if not self.task_protocol.strip():
            raise CapabilityContractError("task_protocol_required")
        if not self.dataset.repository_id.strip():
            raise CapabilityContractError("dataset_repository_required")
        if not self.dataset.revision.strip():
            raise CapabilityContractError("dataset_revision_required")
        if not self.dataset.split.strip():
            raise CapabilityContractError("dataset_split_required")
        if not self.dataset.field_mapping or any(
            not str(role).strip() or not str(column).strip()
            for role, column in self.dataset.field_mapping.items()
        ):
            raise CapabilityContractError("dataset_field_mapping_required")
        if not self.model.repository_id.strip() or not self.model.revision.strip():
            raise CapabilityContractError("model_repository_and_revision_required")
        if not self.model.framework.strip() or not self.model.task.strip():
            raise CapabilityContractError("model_contract_required")
        if self.model.min_vram_gb < 0 or self.min_disk_gb < 0:
            raise CapabilityContractError("resource_requirements_invalid")
        if not self.metric.name.strip() or self.metric.direction not in {
            "higher",
            "lower",
        }:
            raise CapabilityContractError("metric_contract_invalid")
        expected_hook = {
            "generative_qa": "candidate_prompt",
            "sequence_classification": "candidate_text",
        }.get(self.task_protocol)
        if expected_hook and self.candidate_hook != expected_hook:
            raise CapabilityContractError("candidate_hook_contract_invalid")
        if not self.candidate_hook.strip():
            raise CapabilityContractError("candidate_hook_required")
        if not self.seeds or any(int(seed) < 0 for seed in self.seeds):
            raise CapabilityContractError("seed_contract_invalid")
        if self.sample_cap is not None and int(self.sample_cap) <= 0:
            raise CapabilityContractError("sample_cap_invalid")
        if not self.artifact_contract or "final_results" not in self.artifact_contract:
            raise CapabilityContractError("final_results_artifact_required")
        if not self.preferred_backends:
            raise CapabilityContractError("preferred_backend_required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def canonical_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentRequirements":
        dataset = value.get("dataset") or {}
        model = value.get("model") or {}
        metric = value.get("metric") or {}
        result = cls(
            task_protocol=str(value.get("task_protocol") or ""),
            dataset=DatasetRequirement(
                repository_id=str(dataset.get("repository_id") or ""),
                revision=str(dataset.get("revision") or "main"),
                config=str(dataset.get("config") or ""),
                split=str(dataset.get("split") or "test"),
                field_mapping={
                    str(key): str(item)
                    for key, item in dict(dataset.get("field_mapping") or {}).items()
                },
            ),
            model=ModelRequirement(
                repository_id=str(model.get("repository_id") or ""),
                revision=str(model.get("revision") or "main"),
                framework=str(model.get("framework") or "transformers"),
                task=str(model.get("task") or "causal_lm"),
                min_vram_gb=float(model.get("min_vram_gb") or 0.0),
                requires_cuda=bool(model.get("requires_cuda")),
                quantization=str(model.get("quantization") or "none"),
            ),
            metric=MetricRequirement(
                name=canonical_metric_name(str(metric.get("name") or "")),
                direction=str(metric.get("direction") or "higher").lower(),
                required_prediction_fields=tuple(
                    str(item)
                    for item in metric.get(
                        "required_prediction_fields",
                        ("prediction", "target"),
                    )
                ),
            ),
            candidate_hook=str(
                value.get("candidate_hook")
                or {
                    "generative_qa": "candidate_prompt",
                    "sequence_classification": "candidate_text",
                }.get(str(value.get("task_protocol") or ""), "")
            ),
            dependencies=tuple(str(item) for item in value.get("dependencies", ())),
            network_required=bool(value.get("network_required", True)),
            min_disk_gb=float(value.get("min_disk_gb") or 0.0),
            seeds=tuple(int(item) for item in value.get("seeds", (0,))),
            sample_cap=(
                int(value["sample_cap"])
                if value.get("sample_cap") not in (None, "", 0)
                else None
            ),
            artifact_contract=tuple(
                str(item) for item in value.get("artifact_contract", ())
            )
            or cls.artifact_contract,
            preferred_backends=tuple(
                str(item) for item in value.get("preferred_backends", ())
            )
            or cls.preferred_backends,
            schema_version=str(
                value.get("schema_version") or "experiment_requirements_v1"
            ),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class RunnerCapability:
    adapter_id: str
    version: str
    task_protocols: tuple[str, ...]
    candidate_hooks: tuple[str, ...]
    dataset_roles: tuple[str, ...]
    model_frameworks: tuple[str, ...]
    model_tasks: tuple[str, ...]
    metric_names: tuple[str, ...]
    dependencies: tuple[str, ...]
    can_install_dependencies: bool
    network_required: bool
    min_disk_gb: float
    min_vram_gb: float
    supports_seed: bool
    supports_sample_cap: bool
    output_artifacts: tuple[str, ...]
    backends: tuple[str, ...]

    def structural_blockers(
        self, requirements: ExperimentRequirements
    ) -> tuple[str, ...]:
        blockers: list[str] = []
        if requirements.task_protocol not in self.task_protocols:
            blockers.append("unsupported_task_protocol")
        if requirements.candidate_hook not in self.candidate_hooks:
            blockers.append("candidate_hook_unsupported")
        if not set(requirements.dataset.field_mapping).issuperset(
            self.dataset_roles
        ):
            blockers.append("dataset_schema_role_mismatch")
        if requirements.model.framework not in self.model_frameworks:
            blockers.append("model_framework_mismatch")
        if requirements.model.task not in self.model_tasks:
            blockers.append("model_task_mismatch")
        if requirements.metric.name not in self.metric_names:
            blockers.append("metric_contract_unsupported")
        if len(requirements.seeds) > 1 and not self.supports_seed:
            blockers.append("seed_control_unsupported")
        if requirements.sample_cap is not None and not self.supports_sample_cap:
            blockers.append("sample_cap_unsupported")
        if not set(requirements.artifact_contract).issubset(self.output_artifacts):
            blockers.append("artifact_contract_mismatch")
        if not set(requirements.preferred_backends).intersection(self.backends):
            blockers.append("backend_contract_mismatch")
        return tuple(blockers)


class RunnerRegistry:
    def __init__(self, capabilities: Sequence[RunnerCapability] | None = None):
        selected = tuple(capabilities or default_runner_capabilities())
        if len({item.adapter_id for item in selected}) != len(selected):
            raise CapabilityContractError("duplicate_runner_adapter")
        self._capabilities = selected

    def all(self) -> tuple[RunnerCapability, ...]:
        return self._capabilities

    def matches(
        self, requirements: ExperimentRequirements
    ) -> tuple[RunnerCapability, ...]:
        requirements.validate()
        return tuple(
            capability
            for capability in self._capabilities
            if not capability.structural_blockers(requirements)
        )


def default_runner_capabilities() -> tuple[RunnerCapability, ...]:
    common_outputs = (
        "final_results",
        "raw_predictions",
        "environment_manifest",
        "dataset_manifest",
        "model_manifest",
    )
    common_backends = ("local_gpu", "ssh_gpu", "colab_gpu")
    return (
        RunnerCapability(
            adapter_id="transformers_causal_lm_qa_v1",
            version="1.0.0",
            task_protocols=("generative_qa",),
            candidate_hooks=("candidate_prompt",),
            dataset_roles=("prompt", "target"),
            model_frameworks=("transformers",),
            model_tasks=("causal_lm", "text_generation"),
            metric_names=("exact_match", "numeric_accuracy", "accuracy"),
            dependencies=("torch", "transformers", "datasets"),
            can_install_dependencies=True,
            network_required=True,
            min_disk_gb=4.0,
            min_vram_gb=4.0,
            supports_seed=True,
            supports_sample_cap=True,
            output_artifacts=common_outputs,
            backends=common_backends,
        ),
        RunnerCapability(
            adapter_id="transformers_sequence_classification_v1",
            version="1.0.0",
            task_protocols=("sequence_classification",),
            candidate_hooks=("candidate_text",),
            dataset_roles=("text", "label"),
            model_frameworks=("transformers",),
            model_tasks=("sequence_classification",),
            metric_names=("accuracy", "f1", "macro_f1"),
            dependencies=("torch", "transformers", "datasets"),
            can_install_dependencies=True,
            network_required=True,
            min_disk_gb=2.0,
            min_vram_gb=0.0,
            supports_seed=True,
            supports_sample_cap=True,
            output_artifacts=common_outputs,
            backends=("cpu",) + common_backends,
        ),
    )


@dataclass(frozen=True)
class RepositoryMetadata:
    available: bool
    resolved_revision: str = ""
    fields: tuple[str, ...] = ()
    task: str = ""
    size_gb: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class PreflightEnvironment:
    enabled_backends: tuple[str, ...]
    backend_vram_gb: Mapping[str, float]
    network_available: bool
    disk_free_gb: float


class MetadataProbe(Protocol):
    def dataset(
        self, repository_id: str, revision: str, config: str
    ) -> RepositoryMetadata: ...

    def model(self, repository_id: str, revision: str) -> RepositoryMetadata: ...

    def dependency_available(self, name: str) -> bool: ...


class HuggingFaceMetadataProbe:
    """Cheap controller-side repository metadata probe; never loads weights."""

    def __init__(self, *, timeout_seconds: int = 15):
        self.timeout_seconds = max(1, int(timeout_seconds))

    def _json(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "deepgraph-preflight-v1"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else {}

    def dataset(
        self, repository_id: str, revision: str, config: str
    ) -> RepositoryMetadata:
        quoted = urllib.parse.quote(repository_id.strip(), safe="/")
        revision_quoted = urllib.parse.quote(revision.strip(), safe="")
        try:
            payload = self._json(
                f"https://huggingface.co/api/datasets/{quoted}/revision/{revision_quoted}"
            )
        except Exception as exc:
            return RepositoryMetadata(False, reason=f"{type(exc).__name__}")
        fields: set[str] = set()
        card_data = payload.get("cardData") or {}
        dataset_info = card_data.get("dataset_info") if isinstance(card_data, dict) else {}
        candidates = dataset_info if isinstance(dataset_info, list) else [dataset_info]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            if config and str(item.get("config_name") or "") not in {"", config}:
                continue
            features = item.get("features") or []
            if isinstance(features, dict):
                fields.update(str(key) for key in features)
            elif isinstance(features, list):
                fields.update(
                    str(feature.get("name"))
                    for feature in features
                    if isinstance(feature, dict) and feature.get("name")
                )
        return RepositoryMetadata(
            available=True,
            resolved_revision=str(payload.get("sha") or ""),
            fields=tuple(sorted(fields)),
        )

    def model(self, repository_id: str, revision: str) -> RepositoryMetadata:
        quoted = urllib.parse.quote(repository_id.strip(), safe="/")
        revision_quoted = urllib.parse.quote(revision.strip(), safe="")
        try:
            payload = self._json(
                f"https://huggingface.co/api/models/{quoted}/revision/{revision_quoted}"
            )
        except Exception as exc:
            return RepositoryMetadata(False, reason=f"{type(exc).__name__}")
        tags = {str(item) for item in payload.get("tags", [])}
        task = str(payload.get("pipeline_tag") or "")
        siblings = payload.get("siblings") or []
        total_bytes = sum(
            int((item.get("lfs") or {}).get("size") or item.get("size") or 0)
            for item in siblings
            if isinstance(item, dict)
        )
        if not task:
            if "text-generation" in tags:
                task = "text_generation"
            elif "text-classification" in tags:
                task = "sequence_classification"
        return RepositoryMetadata(
            available=True,
            resolved_revision=str(payload.get("sha") or ""),
            task=task.replace("-", "_"),
            size_gb=total_bytes / (1024**3),
        )

    def dependency_available(self, name: str) -> bool:
        return importlib.util.find_spec(str(name).replace("-", "_")) is not None


@dataclass(frozen=True)
class PreflightResult:
    status: str
    reason_codes: tuple[str, ...]
    checks: Mapping[str, Any]
    adapter_id: str | None = None
    adapter_version: str | None = None
    selected_backend: str | None = None
    dataset_revision: str | None = None
    model_revision: str | None = None
    preflight_result_id: int | None = None

    @property
    def passed(self) -> bool:
        return self.status == PREFLIGHT_PASSED


class PreflightEngine:
    def __init__(
        self,
        *,
        registry: RunnerRegistry | None = None,
        probe: MetadataProbe | None = None,
    ):
        self.registry = registry or RunnerRegistry()
        self.probe = probe or HuggingFaceMetadataProbe()

    def run(
        self,
        requirements: ExperimentRequirements,
        environment: PreflightEnvironment,
    ) -> PreflightResult:
        try:
            requirements.validate()
        except CapabilityContractError as exc:
            return PreflightResult(
                PREFLIGHT_FAILED,
                (str(exc),),
                {"contract": "invalid"},
            )
        matches = self.registry.matches(requirements)
        if not matches:
            # Report the closest adapter's blockers, not the union across every
            # adapter. The union mixes in reasons from adapters that were never
            # applicable - a generative_qa candidate collected
            # `unsupported_task_protocol` from the classification adapter - and
            # reads as a capability gap when the real gap is one field.
            ranked = sorted(
                (
                    (capability.adapter_id, capability.structural_blockers(requirements))
                    for capability in self.registry.all()
                ),
                key=lambda item: (len(item[1]), item[0]),
            )
            nearest_id, nearest = ranked[0] if ranked else ("", ())
            return PreflightResult(
                PREFLIGHT_DEFERRED,
                tuple(sorted(nearest) or ["runner_unavailable"]),
                {
                    "runner_match": False,
                    "nearest_adapter": nearest_id,
                    "adapter_blockers": {
                        adapter_id: sorted(reasons) for adapter_id, reasons in ranked
                    },
                },
            )
        dataset = self.probe.dataset(
            requirements.dataset.repository_id,
            requirements.dataset.revision,
            requirements.dataset.config,
        )
        model = self.probe.model(
            requirements.model.repository_id,
            requirements.model.revision,
        )
        reasons: list[str] = []
        checks: dict[str, Any] = {
            "dataset_available": dataset.available,
            "dataset_resolved_revision": dataset.resolved_revision,
            "dataset_fields": list(dataset.fields),
            "model_available": model.available,
            "model_resolved_revision": model.resolved_revision,
            "model_task": model.task,
            "model_size_gb": model.size_gb,
        }
        if not dataset.available:
            reasons.append("dataset_unavailable")
        elif not dataset.resolved_revision:
            reasons.append("dataset_revision_unresolved")
        if dataset.available and dataset.fields:
            missing_fields = sorted(
                set(requirements.dataset.field_mapping.values())
                - set(dataset.fields)
            )
            checks["dataset_missing_fields"] = missing_fields
            if missing_fields:
                reasons.append("dataset_schema_mismatch")
        elif dataset.available:
            reasons.append("dataset_schema_unverified")
        if not model.available:
            reasons.append("model_unavailable")
        elif not model.resolved_revision:
            reasons.append("model_revision_unresolved")
        if model.available and model.task and model.task not in {
            requirements.model.task,
            requirements.model.task.replace("causal_lm", "text_generation"),
        }:
            reasons.append("model_task_mismatch")
        adapter = matches[0]
        required_dependencies = tuple(
            dict.fromkeys(adapter.dependencies + requirements.dependencies)
        )
        missing_dependencies = [
            name
            for name in required_dependencies
            if not self.probe.dependency_available(name)
        ]
        checks["missing_dependencies"] = missing_dependencies
        checks["dependencies_install_planned"] = bool(
            missing_dependencies and adapter.can_install_dependencies
        )
        if missing_dependencies and (
            not adapter.can_install_dependencies
            or not environment.network_available
        ):
            reasons.append("dependency_missing")
        disk_required = max(
            requirements.min_disk_gb,
            adapter.min_disk_gb,
            model.size_gb,
        )
        checks["disk_required_gb"] = disk_required
        checks["disk_free_gb"] = environment.disk_free_gb
        if environment.disk_free_gb + 1e-9 < disk_required:
            reasons.append("disk_insufficient")
        if (requirements.network_required or adapter.network_required) and not (
            environment.network_available
        ):
            reasons.append("network_unavailable")
        eligible_backends = [
            backend
            for backend in requirements.preferred_backends
            if backend in adapter.backends and backend in environment.enabled_backends
        ]
        vram_required = max(requirements.model.min_vram_gb, adapter.min_vram_gb)
        if requirements.model.requires_cuda:
            eligible_backends = [
                backend for backend in eligible_backends if backend != "cpu"
            ]
        sized_backends = [
            backend
            for backend in eligible_backends
            if float(environment.backend_vram_gb.get(backend, 0.0)) + 1e-9
            >= vram_required
        ]
        checks["eligible_backends"] = eligible_backends
        checks["vram_required_gb"] = vram_required
        checks["sized_backends"] = sized_backends
        if not eligible_backends:
            reasons.append("backend_unavailable")
        elif not sized_backends:
            reasons.append("vram_insufficient")
        reasons = list(dict.fromkeys(reasons))
        if reasons:
            return PreflightResult(
                PREFLIGHT_DEFERRED,
                tuple(reasons),
                checks,
                adapter_id=adapter.adapter_id,
                adapter_version=adapter.version,
                dataset_revision=dataset.resolved_revision or None,
                model_revision=model.resolved_revision or None,
            )
        return PreflightResult(
            PREFLIGHT_PASSED,
            (),
            checks,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.version,
            selected_backend=sized_backends[0],
            dataset_revision=dataset.resolved_revision,
            model_revision=model.resolved_revision,
        )


def local_preflight_environment(
    *,
    enabled_backends: Sequence[str],
    backend_vram_gb: Mapping[str, float],
    network_available: bool = True,
    path: str | Path = ".",
) -> PreflightEnvironment:
    free_bytes = shutil.disk_usage(Path(path)).free
    return PreflightEnvironment(
        enabled_backends=tuple(str(item) for item in enabled_backends),
        backend_vram_gb={str(key): float(value) for key, value in backend_vram_gb.items()},
        network_available=bool(network_available),
        disk_free_gb=free_bytes / (1024**3),
    )


def requirements_from_plan(plan: Mapping[str, Any]) -> ExperimentRequirements:
    """Translate a candidate design without inspecting repository names."""
    explicit = plan.get("execution_requirements")
    if isinstance(explicit, Mapping):
        return ExperimentRequirements.from_dict(explicit)
    targets = [
        item for item in plan.get("benchmark_targets", []) if isinstance(item, Mapping)
    ]
    models = [item for item in plan.get("model_targets", []) if isinstance(item, Mapping)]
    if not targets or not models:
        raise CapabilityContractError("candidate_execution_requirements_missing")
    target = targets[0]
    model = models[0]
    task_type = str(
        target.get("task_protocol") or target.get("task_type") or ""
    ).lower()
    protocol_map = {
        "math_qa": "generative_qa",
        "multihop_qa": "generative_qa",
        "boolean_qa": "generative_qa",
        "qa": "generative_qa",
        "text_classification": "sequence_classification",
        "classification": "sequence_classification",
        "retrieval": "retrieval_ranking",
    }
    protocol = protocol_map.get(task_type, task_type)
    if protocol == "generative_qa":
        field_mapping = {
            "prompt": str(target.get("question_field") or "question"),
            "target": str(target.get("answer_field") or "answer"),
        }
        model_task = str(model.get("task") or "causal_lm")
    elif protocol == "sequence_classification":
        field_mapping = {
            "text": str(target.get("text_field") or "text"),
            "label": str(target.get("label_field") or "label"),
        }
        model_task = str(model.get("task") or "sequence_classification")
    elif protocol == "retrieval_ranking":
        field_mapping = {
            "query": str(target.get("query_field") or "query"),
            "document": str(target.get("document_field") or "document"),
            "relevance": str(target.get("relevance_field") or "relevance"),
        }
        model_task = str(model.get("task") or "embedding")
    else:
        field_mapping = {
            str(key): str(value)
            for key, value in dict(target.get("field_mapping") or {}).items()
        }
        model_task = str(model.get("task") or "")
    metrics = plan.get("metrics") or {}
    metric_name = (
        metrics.get("primary") if isinstance(metrics, Mapping) else None
    ) or target.get("primary_metric")
    protocol_block = plan.get("benchmark_protocol") or {}
    if not metric_name and isinstance(protocol_block, Mapping):
        metric_policy = protocol_block.get("metric_policy") or {}
        if isinstance(metric_policy, Mapping):
            metric_name = metric_policy.get("primary_metric")
    seeds = tuple(
        int(item)
        for item in (
            ((protocol_block.get("seed_policy") or {}).get("seed_values") or [])
            if isinstance(protocol_block, Mapping)
            else []
        )
    ) or tuple(range(max(1, int(plan.get("minimum_seeds") or 1))))
    sample_cap = int(target.get("max_eval_examples") or plan.get("max_eval_examples") or 0)
    artifact_contract = tuple(
        str(item)
        for item in (
            (
                (protocol_block.get("full_benchmark_requirements") or {}).get(
                    "required_artifacts"
                )
                or []
            )
            if isinstance(protocol_block, Mapping)
            else []
        )
    )
    # The adapter contract uses logical artifacts; additional publication
    # files remain experiment outputs but do not redefine runner admission.
    logical_artifacts = (
        "final_results",
        "raw_predictions",
        "environment_manifest",
        "dataset_manifest",
        "model_manifest",
    )
    result = ExperimentRequirements(
        task_protocol=protocol,
        dataset=DatasetRequirement(
            repository_id=str(
                target.get("hf_dataset") or target.get("dataset_id") or ""
            ),
            revision=str(target.get("revision") or "main"),
            config=str(target.get("config") or ""),
            split=str(target.get("split") or "test"),
            field_mapping=field_mapping,
        ),
        model=ModelRequirement(
            repository_id=str(model.get("hf_model") or model.get("model_id") or ""),
            revision=str(model.get("revision") or "main"),
            framework=str(model.get("backend") or "transformers"),
            task=model_task,
            min_vram_gb=float(model.get("min_vram_gb") or 0.0),
            requires_cuda=bool(model.get("requires_cuda")),
            quantization=(
                "4bit" if model.get("load_in_4bit") else str(model.get("quantization") or "none")
            ),
        ),
        metric=MetricRequirement(
            # from_dict() folds the alias table but this path did not, so a
            # plan whose metric was a sanctioned synonym was refused for
            # spelling while the identical requirements loaded from a stored
            # row passed. Unknown names still fall through untouched and keep
            # surfacing as a real capability gap.
            name=canonical_metric_name(str(metric_name or "")),
            direction=str(
                (metrics.get("direction") if isinstance(metrics, Mapping) else None)
                or "higher"
            ).lower(),
        ),
        candidate_hook={
            "generative_qa": "candidate_prompt",
            "sequence_classification": "candidate_text",
        }.get(protocol, str(plan.get("candidate_hook") or "")),
        dependencies=("torch", "transformers", "datasets"),
        network_required=True,
        min_disk_gb=float((plan.get("compute_budget") or {}).get("disk_gb") or 4.0),
        seeds=seeds,
        sample_cap=sample_cap or None,
        artifact_contract=logical_artifacts,
        preferred_backends=("ssh_gpu", "local_gpu", "colab_gpu"),
    )
    result.validate()
    return result
