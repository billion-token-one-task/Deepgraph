"""Structured pipeline contracts and adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from contracts.base import (
    ContractRecord,
    ContractValidationError,
    coerce_optional_float,
    coerce_optional_int,
    dedupe_strings,
    ensure_dict,
    ensure_list,
    ensure_string_list,
    require_non_empty,
)


DEEP_INSIGHT_JSON_FIELDS = {
    "field_a",
    "field_b",
    "predictions",
    "falsification",
    "proposed_method",
    "experimental_plan",
    "related_work_positioning",
    "supporting_papers",
    "source_node_ids",
    "source_paper_ids",
    "signal_mix",
    "evidence_packet",
    "evidence_plan",
    "adversarial_critique",
    "source_signal_ids",
    "novelty_report",
    "exemplars_used",
}


def _copy_mapping(payload: Mapping[str, Any] | dict[str, Any] | None) -> dict[str, Any]:
    return dict(payload or {})


@dataclass
class StructuredPaperRecord(ContractRecord):
    paper_id: str = ""
    title: str = ""
    source_url: str = ""
    ingestion_metadata: dict[str, Any] = field(default_factory=dict)
    full_text: str = ""
    paper_overview: dict[str, Any] = field(default_factory=dict)
    taxonomy_nodes: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    methods: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    knowledge_graph: dict[str, Any] = field(default_factory=dict)
    processing_stage: str = "extracted"
    token_usage: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        require_non_empty("paper_id", self.paper_id)
        require_non_empty("title", self.title)

    @classmethod
    def from_processing_state(
        cls,
        *,
        paper: Mapping[str, Any],
        full_text: str,
        extraction: Mapping[str, Any] | None = None,
        contradictions: list[dict[str, Any]] | None = None,
        processing_stage: str = "extracted",
        token_usage: Mapping[str, Any] | None = None,
    ) -> "StructuredPaperRecord":
        extraction = extraction or {}
        return cls(
            paper_id=str(paper.get("id") or ""),
            title=str(paper.get("title") or ""),
            source_url=str(paper.get("source_url") or paper.get("pdf_url") or ""),
            ingestion_metadata={
                "published_date": paper.get("published_date"),
                "authors": ensure_string_list(paper.get("authors")),
                "status": paper.get("status"),
            },
            full_text=str(full_text or ""),
            paper_overview=ensure_dict(extraction.get("paper_overview")),
            taxonomy_nodes=[
                {
                    "node_id": str(row.get("node_id") or ""),
                    "confidence": coerce_optional_float(row.get("confidence")) or 1.0,
                }
                for row in ensure_list(extraction.get("taxonomy_nodes"))
                if str(row.get("node_id") or "").strip()
            ],
            claims=[dict(row) for row in ensure_list(extraction.get("claims")) if isinstance(row, dict)],
            methods=[dict(row) for row in ensure_list(extraction.get("methods")) if isinstance(row, dict)],
            results=[dict(row) for row in ensure_list(extraction.get("results")) if isinstance(row, dict)],
            contradictions=[dict(row) for row in contradictions or [] if isinstance(row, dict)],
            knowledge_graph=ensure_dict(extraction.get("knowledge_graph")),
            processing_stage=processing_stage,
            token_usage=_copy_mapping(token_usage),
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        full_text = payload.pop("full_text", "")
        payload["full_text_length"] = len(str(full_text or ""))
        return {"structured_paper_record": payload}

    def event_payload(self) -> dict[str, Any]:
        return {
            "structured_paper_record": {
                "schema_version": self.schema_version,
                "contract_type": self.contract_type(),
                "paper_id": self.paper_id,
                "title": self.title,
                "processing_stage": self.processing_stage,
                "taxonomy_nodes": self.taxonomy_nodes,
                "claim_count": len(self.claims),
                "result_count": len(self.results),
                "contradiction_count": len(self.contradictions),
            }
        }


@dataclass
class DiscoverySignalBundle(ContractRecord):
    tier: int = 0
    stage: str = "signal_harvest"
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        if self.tier not in (1, 2):
            raise ContractValidationError("DiscoverySignalBundle tier must be 1 or 2")

    @classmethod
    def from_payload(
        cls,
        *,
        tier: int,
        payload: Mapping[str, Any] | None,
        metadata: Mapping[str, Any] | None = None,
        stage: str = "signal_harvest",
    ) -> "DiscoverySignalBundle":
        return cls(
            tier=tier,
            stage=stage,
            payload=_copy_mapping(payload),
            metadata=_copy_mapping(metadata),
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __contains__(self, key: object) -> bool:
        return key in self.payload


@dataclass
class DeepInsightSpec(ContractRecord):
    insight_id: int | None = None
    tier: int = 0
    status: str = "candidate"
    title: str = ""
    problem_statement: str = ""
    existing_weakness: str = ""
    formal_structure: str = ""
    transformation: str = ""
    mechanism_type: str = ""
    submission_status: str = "not_started"
    novelty_status: str = "unchecked"
    resource_class: str = ""
    experimentability: str = ""
    evidence_summary: str = ""
    field_a: dict[str, Any] = field(default_factory=dict)
    field_b: dict[str, Any] = field(default_factory=dict)
    predictions: list[Any] = field(default_factory=list)
    falsification: dict[str, Any] = field(default_factory=dict)
    proposed_method: dict[str, Any] = field(default_factory=dict)
    experimental_plan: dict[str, Any] = field(default_factory=dict)
    related_work_positioning: dict[str, Any] = field(default_factory=dict)
    supporting_papers: list[str] = field(default_factory=list)
    source_paper_ids: list[str] = field(default_factory=list)
    source_node_ids: list[str] = field(default_factory=list)
    source_signal_ids: list[str] = field(default_factory=list)
    signal_mix: list[str] = field(default_factory=list)
    evidence_packet: dict[str, Any] = field(default_factory=dict)
    evidence_plan: dict[str, Any] = field(default_factory=dict)
    adversarial_score: float | None = None
    adversarial_critique: dict[str, Any] = field(default_factory=dict)
    novelty_report: dict[str, Any] = field(default_factory=dict)
    exemplars_used: list[Any] = field(default_factory=list)
    llm_model: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        require_non_empty("title", self.title)
        if self.tier not in (1, 2):
            raise ContractValidationError("DeepInsightSpec tier must be 1 or 2")

    @classmethod
    def from_raw(cls, payload: Mapping[str, Any]) -> "DeepInsightSpec":
        raw = dict(payload)
        known = set(cls.__dataclass_fields__.keys())
        known.discard("schema_version")
        known.discard("extras")
        extras = {key: value for key, value in raw.items() if key not in known and key != "contract_type"}
        return cls(
            schema_version=str(raw.get("schema_version") or cls().schema_version),
            insight_id=coerce_optional_int(raw.get("insight_id") or raw.get("id")),
            tier=coerce_optional_int(raw.get("tier")) or 0,
            status=str(raw.get("status") or "candidate"),
            title=str(raw.get("title") or ""),
            problem_statement=str(raw.get("problem_statement") or ""),
            existing_weakness=str(raw.get("existing_weakness") or ""),
            formal_structure=str(raw.get("formal_structure") or ""),
            transformation=str(raw.get("transformation") or ""),
            mechanism_type=str(raw.get("mechanism_type") or ""),
            submission_status=str(raw.get("submission_status") or "not_started"),
            novelty_status=str(raw.get("novelty_status") or "unchecked"),
            resource_class=str(raw.get("resource_class") or ""),
            experimentability=str(raw.get("experimentability") or ""),
            evidence_summary=str(raw.get("evidence_summary") or ""),
            field_a=ensure_dict(raw.get("field_a")),
            field_b=ensure_dict(raw.get("field_b")),
            predictions=ensure_list(raw.get("predictions")),
            falsification=ensure_dict(raw.get("falsification")),
            proposed_method=ensure_dict(raw.get("proposed_method")),
            experimental_plan=ensure_dict(raw.get("experimental_plan")),
            related_work_positioning=ensure_dict(raw.get("related_work_positioning")),
            supporting_papers=ensure_string_list(raw.get("supporting_papers")),
            source_paper_ids=ensure_string_list(raw.get("source_paper_ids")),
            source_node_ids=ensure_string_list(raw.get("source_node_ids")),
            source_signal_ids=ensure_string_list(raw.get("source_signal_ids")),
            signal_mix=ensure_string_list(raw.get("signal_mix")),
            evidence_packet=ensure_dict(raw.get("evidence_packet")),
            evidence_plan=ensure_dict(raw.get("evidence_plan")),
            adversarial_score=coerce_optional_float(raw.get("adversarial_score")),
            adversarial_critique=ensure_dict(raw.get("adversarial_critique")),
            novelty_report=ensure_dict(raw.get("novelty_report")),
            exemplars_used=ensure_list(raw.get("exemplars_used")),
            llm_model=str(raw.get("llm_model") or raw.get("model_version") or ""),
            extras=extras,
        )

    def to_storage_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self.extras)
        payload.update(
            {
                "id": self.insight_id,
                "tier": self.tier,
                "status": self.status,
                "title": self.title,
                "problem_statement": self.problem_statement,
                "existing_weakness": self.existing_weakness,
                "formal_structure": self.formal_structure,
                "transformation": self.transformation,
                "mechanism_type": self.mechanism_type,
                "submission_status": self.submission_status,
                "novelty_status": self.novelty_status,
                "resource_class": self.resource_class,
                "experimentability": self.experimentability,
                "evidence_summary": self.evidence_summary,
                "field_a": self.field_a,
                "field_b": self.field_b,
                "predictions": self.predictions,
                "falsification": self.falsification,
                "proposed_method": self.proposed_method,
                "experimental_plan": self.experimental_plan,
                "related_work_positioning": self.related_work_positioning,
                "supporting_papers": self.supporting_papers,
                "source_paper_ids": self.source_paper_ids,
                "source_node_ids": self.source_node_ids,
                "source_signal_ids": self.source_signal_ids,
                "signal_mix": self.signal_mix,
                "evidence_packet": self.evidence_packet,
                "evidence_plan": self.evidence_plan,
                "adversarial_score": self.adversarial_score,
                "adversarial_critique": self.adversarial_critique,
                "novelty_report": self.novelty_report,
                "exemplars_used": self.exemplars_used,
                "llm_model": self.llm_model,
            }
        )
        for key in DEEP_INSIGHT_JSON_FIELDS:
            if key in payload:
                payload[key] = payload[key]
        return payload

    def storage_json_dict(self) -> dict[str, Any]:
        payload = self.to_storage_dict()
        for key in DEEP_INSIGHT_JSON_FIELDS:
            if key in payload:
                payload[key] = ensure_dict(payload[key]) if key not in {
                    "supporting_papers",
                    "source_paper_ids",
                    "source_node_ids",
                    "source_signal_ids",
                    "signal_mix",
                    "predictions",
                    "exemplars_used",
                } else ensure_list(payload[key])
        return payload


@dataclass
class ExperimentJudgement(ContractRecord):
    deep_insight_id: int | None = None
    recommended_route: str = "blocked"
    formal_experiment: bool = False
    smoke_test_only: bool = False
    summary: str = ""
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    baseline_review: dict[str, Any] = field(default_factory=dict)
    scale_review: dict[str, Any] = field(default_factory=dict)
    alignment_review: dict[str, Any] = field(default_factory=dict)
    environment_review: dict[str, Any] = field(default_factory=dict)
    codebase_review: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        if self.recommended_route not in {"formal", "smoke_test", "blocked"}:
            raise ContractValidationError("ExperimentJudgement route must be formal/smoke_test/blocked")
        if self.formal_experiment and self.smoke_test_only:
            raise ContractValidationError("ExperimentJudgement cannot be formal and smoke_test_only simultaneously")

    @classmethod
    def from_raw(cls, payload: Mapping[str, Any] | None) -> "ExperimentJudgement":
        payload = payload or {}
        return cls(
            schema_version=str(payload.get("schema_version") or cls().schema_version),
            deep_insight_id=coerce_optional_int(payload.get("deep_insight_id")),
            recommended_route=str(payload.get("recommended_route") or "blocked"),
            formal_experiment=bool(payload.get("formal_experiment")),
            smoke_test_only=bool(payload.get("smoke_test_only")),
            summary=str(payload.get("summary") or ""),
            blockers=ensure_string_list(payload.get("blockers")),
            warnings=ensure_string_list(payload.get("warnings")),
            baseline_review=ensure_dict(payload.get("baseline_review")),
            scale_review=ensure_dict(payload.get("scale_review")),
            alignment_review=ensure_dict(payload.get("alignment_review")),
            environment_review=ensure_dict(payload.get("environment_review")),
            codebase_review=ensure_dict(payload.get("codebase_review")),
        )


@dataclass
class ExperimentSpec(ContractRecord):
    run_id: int | None = None
    deep_insight_id: int | None = None
    insight_title: str = ""
    workdir: str = ""
    formal_experiment: bool = False
    smoke_test_only: bool = False
    resource_class: str = ""
    codebase: dict[str, Any] = field(default_factory=dict)
    proposed_method: dict[str, Any] = field(default_factory=dict)
    experimental_plan: dict[str, Any] = field(default_factory=dict)
    evidence_plan: dict[str, Any] = field(default_factory=dict)
    success_criteria: dict[str, Any] = field(default_factory=dict)
    proxy_config: dict[str, Any] = field(default_factory=dict)
    judgement: ExperimentJudgement = field(default_factory=ExperimentJudgement)
    artifact_paths: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        if self.deep_insight_id is None:
            raise ContractValidationError("ExperimentSpec missing deep_insight_id")
        if self.formal_experiment and not self.judgement.formal_experiment:
            raise ContractValidationError("ExperimentSpec formal flag must agree with judgement")

    @classmethod
    def from_sources(
        cls,
        *,
        run_id: int | None,
        insight: DeepInsightSpec,
        workdir: str,
        codebase: Mapping[str, Any],
        judgement: ExperimentJudgement,
        success_criteria: Mapping[str, Any] | None,
        proxy_config: Mapping[str, Any] | None,
        artifact_paths: Mapping[str, Any] | None = None,
    ) -> "ExperimentSpec":
        return cls(
            run_id=run_id,
            deep_insight_id=insight.insight_id,
            insight_title=insight.title,
            workdir=workdir,
            formal_experiment=judgement.formal_experiment,
            smoke_test_only=judgement.smoke_test_only,
            resource_class=insight.resource_class,
            codebase=_copy_mapping(codebase),
            proposed_method=dict(insight.proposed_method),
            experimental_plan=dict(insight.experimental_plan),
            evidence_plan=dict(insight.evidence_plan),
            success_criteria=_copy_mapping(success_criteria),
            proxy_config=_copy_mapping(proxy_config),
            judgement=judgement,
            artifact_paths=_copy_mapping(artifact_paths),
        )

    @classmethod
    def from_run_row(
        cls,
        run: Mapping[str, Any],
        insight: DeepInsightSpec,
        *,
        success_criteria: Mapping[str, Any] | None = None,
        proxy_config: Mapping[str, Any] | None = None,
    ) -> "ExperimentSpec":
        proxy_config = _copy_mapping(proxy_config)
        judgement = ExperimentJudgement.from_raw(proxy_config.get("experiment_judgement"))
        return cls(
            run_id=coerce_optional_int(run.get("id")),
            deep_insight_id=coerce_optional_int(run.get("deep_insight_id")) or insight.insight_id,
            insight_title=insight.title,
            workdir=str(run.get("workdir") or ""),
            formal_experiment=bool(proxy_config.get("formal_experiment")),
            smoke_test_only=bool(proxy_config.get("smoke_test_only")),
            resource_class=str(run.get("resource_class") or insight.resource_class or ""),
            codebase={
                "url": run.get("codebase_url"),
                "name": run.get("codebase_ref"),
            },
            proposed_method=dict(insight.proposed_method),
            experimental_plan=dict(insight.experimental_plan),
            evidence_plan=dict(insight.evidence_plan),
            success_criteria=_copy_mapping(success_criteria),
            proxy_config=proxy_config,
            judgement=judgement,
            artifact_paths={},
        )


@dataclass
class ExperimentIterationPacket(ContractRecord):
    run_id: int | None = None
    iteration_number: int = 0
    phase: str = ""
    status: str = ""
    description: str = ""
    metric_name: str = "metric"
    metric_value: float | None = None
    baseline_value: float | None = None
    best_value_before: float | None = None
    best_value_after: float | None = None
    environment_report: dict[str, Any] = field(default_factory=dict)
    judge_report: dict[str, Any] = field(default_factory=dict)
    execution_report: dict[str, Any] = field(default_factory=dict)
    result_judgement: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, Any] = field(default_factory=dict)
    commit_hash: str = ""
    code_diff: str = ""

    def validate(self) -> None:
        super().validate()
        if self.iteration_number <= 0:
            raise ContractValidationError("ExperimentIterationPacket iteration_number must be > 0")
        require_non_empty("phase", self.phase)


@dataclass
class ExperimentResultPacket(ContractRecord):
    run_id: int | None = None
    deep_insight_id: int | None = None
    formal_experiment: bool = False
    smoke_test_only: bool = False
    metric_name: str = "metric"
    metric_direction: str = "higher"
    verdict: str = "inconclusive"
    baseline: float | None = None
    best: float | None = None
    effect_size: float | None = None
    effect_pct: float | None = None
    p_value: float | None = None
    confidence: float | None = None
    total_iterations: int = 0
    kept_iterations: int = 0
    crash_count: int = 0
    reproduction_metrics: list[float] = field(default_factory=list)
    hypothesis_iterations: list[dict[str, Any]] = field(default_factory=list)
    best_iteration: dict[str, Any] = field(default_factory=dict)
    claim_text: str = ""
    source_paper_ids: list[str] = field(default_factory=list)
    source_node_ids: list[str] = field(default_factory=list)
    benchmark_summary: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        if self.run_id is None:
            raise ContractValidationError("ExperimentResultPacket missing run_id")
        if self.deep_insight_id is None:
            raise ContractValidationError("ExperimentResultPacket missing deep_insight_id")

    def require_formal_manuscript_ready(self) -> None:
        self.validate()
        if not self.formal_experiment or self.smoke_test_only:
            raise ContractValidationError("Non-formal experiment result cannot enter manuscript path")
        if not self.claim_text:
            raise ContractValidationError("ExperimentResultPacket missing claim_text")


@dataclass
class ManuscriptInputState(ContractRecord):
    run_id: int | None = None
    deep_insight_id: int | None = None
    formal_experiment: bool = False
    smoke_test_only: bool = False
    title: str = ""
    problem_statement: str = ""
    existing_weakness: str = ""
    method_name: str = ""
    method_summary: str = ""
    method_payload: dict[str, Any] = field(default_factory=dict)
    mechanism_type: str = ""
    resource_class: str = ""
    baseline_metric_name: str = "metric"
    baseline_metric_value: float | None = None
    best_metric_value: float | None = None
    effect_pct: float | None = None
    verdict: str = "inconclusive"
    claims: list[dict[str, Any]] = field(default_factory=list)
    iterations: list[dict[str, Any]] = field(default_factory=list)
    best_iteration: dict[str, Any] = field(default_factory=dict)
    datasets: list[Any] = field(default_factory=list)
    baselines: list[Any] = field(default_factory=list)
    paper_outline: dict[str, Any] = field(default_factory=dict)
    contributions: list[str] = field(default_factory=list)
    supporting_papers: list[str] = field(default_factory=list)
    source_paper_ids: list[str] = field(default_factory=list)
    source_node_ids: list[str] = field(default_factory=list)
    citation_seed_paper_ids: list[str] = field(default_factory=list)
    evidence_summary: str = ""
    evidence_packet: dict[str, Any] = field(default_factory=dict)
    evidence_plan: dict[str, Any] = field(default_factory=dict)
    experimental_plan: dict[str, Any] = field(default_factory=dict)
    submission_keywords: list[Any] = field(default_factory=list)
    result_packet: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        require_non_empty("title", self.title)
        require_non_empty("method_name", self.method_name)

    def require_submission_ready(self) -> None:
        self.validate()
        if not self.formal_experiment or self.smoke_test_only:
            raise ContractValidationError("Only formal experiment states may generate submission bundles")
        if not self.result_packet:
            raise ContractValidationError("ManuscriptInputState missing ExperimentResultPacket")
        verdict = str(self.result_packet.get("verdict") or self.verdict or "").strip().lower()
        benchmark_summary = self.result_packet.get("benchmark_summary")
        if verdict == "reproduced":
            if not isinstance(benchmark_summary, dict) or not benchmark_summary.get("per_method"):
                raise ContractValidationError("Reproduction-only runs cannot enter manuscript generation")
        hypothesis_iterations = self.result_packet.get("hypothesis_iterations")
        if isinstance(hypothesis_iterations, list) and not hypothesis_iterations and not (
            isinstance(benchmark_summary, dict) and benchmark_summary.get("per_method")
        ):
            raise ContractValidationError("ManuscriptInputState missing hypothesis-testing evidence")
        if not self.citation_seed_paper_ids:
            raise ContractValidationError("ManuscriptInputState missing citation_seed_paper_ids")
        if not self.claims:
            raise ContractValidationError("ManuscriptInputState missing claim records")


def parse_deep_insight_spec(payload: Mapping[str, Any]) -> DeepInsightSpec:
    return DeepInsightSpec.from_raw(payload)


def merge_signal_bundle_metadata(bundle: DiscoverySignalBundle, **metadata: Any) -> DiscoverySignalBundle:
    merged = dict(bundle.metadata)
    merged.update(metadata)
    return DiscoverySignalBundle(
        schema_version=bundle.schema_version,
        tier=bundle.tier,
        stage=bundle.stage,
        payload=dict(bundle.payload),
        metadata=merged,
    )


def normalize_deep_insight_storage(spec: DeepInsightSpec) -> dict[str, Any]:
    spec.validate()
    payload = spec.to_storage_dict()
    for key in ("supporting_papers", "source_paper_ids", "source_node_ids", "source_signal_ids", "signal_mix", "predictions", "exemplars_used"):
        payload[key] = ensure_list(payload.get(key))
    for key in DEEP_INSIGHT_JSON_FIELDS - {"supporting_papers", "source_paper_ids", "source_node_ids", "source_signal_ids", "signal_mix", "predictions", "exemplars_used"}:
        payload[key] = ensure_dict(payload.get(key))
    if payload.get("adversarial_score") is None:
        payload["adversarial_score"] = spec.adversarial_score
    payload["supporting_papers"] = dedupe_strings(payload.get("supporting_papers", []))
    payload["source_paper_ids"] = dedupe_strings(payload.get("source_paper_ids", []))
    payload["source_node_ids"] = dedupe_strings(payload.get("source_node_ids", []))
    payload["source_signal_ids"] = dedupe_strings(payload.get("source_signal_ids", []))
    payload["signal_mix"] = dedupe_strings(payload.get("signal_mix", []))
    return payload
