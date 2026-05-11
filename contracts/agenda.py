"""Contracts for agenda-driven autonomous research loop (issue #9).

Adds five structured dataclasses on top of contracts.base.ContractRecord:
- ResearchAgenda             : the configurable agenda spec (YAML/JSON input)
- AgendaSelection            : the chosen deep_insight + scoring rationale
- AgendaReview               : reviewer verdict on the produced manuscript bundle
- AgendaRevisionPlan         : next-step experiments derived from review
- LoopInspectionSnapshot     : aggregated end-to-end loop view for dashboard
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.base import (
    ContractRecord,
    ContractValidationError,
    coerce_optional_float,
    coerce_optional_int,
    ensure_dict,
    ensure_list,
    ensure_string_list,
    require_non_empty,
)


VALID_REVIEW_RECOMMENDATIONS = {
    "accept",
    "minor_revision",
    "major_revision",
    "reject",
}

VALID_SELECTION_STATUS = {
    "pending",
    "launched",
    "blocked",
    "completed",
    "failed",
}


@dataclass
class ResearchAgenda(ContractRecord):
    """Configurable research agenda used to bias deep_insight selection.

    Schema mirrors the agenda_v1 sample in issue #9:
        version, name, focus[], prefer{}, reject{}, required_output{}
    """

    version: str = "v1"
    name: str = ""
    description: str = ""
    focus: list[str] = field(default_factory=list)
    prefer: dict[str, Any] = field(default_factory=dict)
    reject: dict[str, Any] = field(default_factory=dict)
    required_output: dict[str, Any] = field(default_factory=dict)
    raw_config: dict[str, Any] = field(default_factory=dict)
    agenda_id: int | None = None
    is_active: bool = True

    def validate(self) -> None:
        super().validate()
        require_non_empty("name", self.name)
        require_non_empty("version", self.version)
        self.focus = ensure_string_list(self.focus)
        self.prefer = ensure_dict(self.prefer)
        self.reject = ensure_dict(self.reject)
        self.required_output = ensure_dict(self.required_output)
        self.raw_config = ensure_dict(self.raw_config)
        if not self.focus and not self.prefer:
            raise ContractValidationError(
                "ResearchAgenda needs at least one focus keyword or prefer rule"
            )


@dataclass
class AgendaSelection(ContractRecord):
    """A single selection decision produced by the agenda selector."""

    agenda_id: int = 0
    selected_insight_id: int | None = None
    score: float | None = None
    rationale: str = ""
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    scoring_breakdown: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    auto_research_job_id: int | None = None
    experiment_run_id: int | None = None
    manuscript_run_id: int | None = None
    submission_bundle_id: int | None = None
    error_message: str = ""
    selection_id: int | None = None

    def validate(self) -> None:
        super().validate()
        if not self.agenda_id:
            raise ContractValidationError("AgendaSelection requires agenda_id")
        if self.status not in VALID_SELECTION_STATUS:
            raise ContractValidationError(
                f"AgendaSelection status '{self.status}' not in {sorted(VALID_SELECTION_STATUS)}"
            )
        self.score = coerce_optional_float(self.score)
        self.selected_insight_id = coerce_optional_int(self.selected_insight_id)
        self.auto_research_job_id = coerce_optional_int(self.auto_research_job_id)
        self.experiment_run_id = coerce_optional_int(self.experiment_run_id)
        self.manuscript_run_id = coerce_optional_int(self.manuscript_run_id)
        self.submission_bundle_id = coerce_optional_int(self.submission_bundle_id)
        self.rejected_candidates = ensure_list(self.rejected_candidates)
        self.scoring_breakdown = ensure_dict(self.scoring_breakdown)


@dataclass
class AgendaReview(ContractRecord):
    """Reviewer verdict on a completed submission bundle."""

    selection_id: int = 0
    submission_bundle_id: int | None = None
    manuscript_run_id: int | None = None
    reviewer: str = "internal_evidence_gate"
    recommendation: str = "minor_revision"
    confidence: float | None = None
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    required_revisions: list[str] = field(default_factory=list)
    evidence_blockers: list[dict[str, Any]] = field(default_factory=list)
    raw_review: dict[str, Any] = field(default_factory=dict)
    review_id: int | None = None

    def validate(self) -> None:
        super().validate()
        if not self.selection_id:
            raise ContractValidationError("AgendaReview requires selection_id")
        if self.recommendation not in VALID_REVIEW_RECOMMENDATIONS:
            raise ContractValidationError(
                f"AgendaReview recommendation '{self.recommendation}' not in "
                f"{sorted(VALID_REVIEW_RECOMMENDATIONS)}"
            )
        require_non_empty("reviewer", self.reviewer)
        self.confidence = coerce_optional_float(self.confidence)
        self.strengths = ensure_string_list(self.strengths)
        self.weaknesses = ensure_string_list(self.weaknesses)
        self.required_revisions = ensure_string_list(self.required_revisions)
        self.evidence_blockers = ensure_list(self.evidence_blockers)
        self.raw_review = ensure_dict(self.raw_review)


@dataclass
class AgendaRevisionPlan(ContractRecord):
    """Structured next-step plan derived from review feedback."""

    selection_id: int = 0
    review_id: int = 0
    rationale: str = ""
    next_experiments: list[dict[str, Any]] = field(default_factory=list)
    status: str = "proposed"
    plan_id: int | None = None

    def validate(self) -> None:
        super().validate()
        if not self.selection_id:
            raise ContractValidationError("AgendaRevisionPlan requires selection_id")
        if not self.review_id:
            raise ContractValidationError("AgendaRevisionPlan requires review_id")
        self.next_experiments = ensure_list(self.next_experiments)
        if not self.next_experiments and not self.rationale.strip():
            raise ContractValidationError(
                "AgendaRevisionPlan requires either next_experiments or rationale"
            )


@dataclass
class LoopInspectionSnapshot(ContractRecord):
    """Aggregated end-to-end loop view returned by /api/research_agenda/loop/<id>."""

    selection: dict[str, Any] = field(default_factory=dict)
    agenda: dict[str, Any] = field(default_factory=dict)
    insight: dict[str, Any] = field(default_factory=dict)
    auto_research_job: dict[str, Any] = field(default_factory=dict)
    experiment_run: dict[str, Any] = field(default_factory=dict)
    evidence_gate: dict[str, Any] = field(default_factory=dict)
    manuscript_run: dict[str, Any] = field(default_factory=dict)
    submission_bundle: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    revision_plan: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        if not self.selection:
            raise ContractValidationError("LoopInspectionSnapshot requires selection")
        for name in (
            "selection",
            "agenda",
            "insight",
            "auto_research_job",
            "experiment_run",
            "evidence_gate",
            "manuscript_run",
            "submission_bundle",
            "review",
            "revision_plan",
        ):
            setattr(self, name, ensure_dict(getattr(self, name)))
