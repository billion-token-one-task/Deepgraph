"""Contracts for agenda-scoped research and hard budget accounting."""

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


VALID_AGENDA_STATUS = {"active", "paused_budget", "paused_manual", "closed"}
VALID_SELECTION_STATUS = {
    "pending",
    "awaiting_portfolio_decision",
    "granted",
    "running",
    "blocked",
    "completed",
    "failed",
}
VALID_BACKLOG_POLICIES = {"explicit_import_only", "new_only"}


@dataclass
class ResearchAgenda(ContractRecord):
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
    submitter: str = ""
    token_budget: int | None = None
    token_spent: int = 0
    token_reserved: int = 0
    gpu_hours_budget: float = 0.0
    gpu_hours_spent: float = 0.0
    gpu_hours_reserved: float = 0.0
    max_concurrency: int = 1
    backend_allowlist: list[str] = field(default_factory=lambda: ["cpu", "llm"])
    backlog_policy: str = "explicit_import_only"
    status: str = "active"

    def validate(self) -> None:
        super().validate()
        require_non_empty("name", self.name)
        require_non_empty("version", self.version)
        self.focus = ensure_string_list(self.focus)
        self.prefer = ensure_dict(self.prefer)
        self.reject = ensure_dict(self.reject)
        self.required_output = ensure_dict(self.required_output)
        self.raw_config = ensure_dict(self.raw_config)
        self.backend_allowlist = ensure_string_list(self.backend_allowlist)
        self.token_budget = coerce_optional_int(self.token_budget)
        self.token_spent = coerce_optional_int(self.token_spent) or 0
        self.token_reserved = coerce_optional_int(self.token_reserved) or 0
        self.max_concurrency = coerce_optional_int(self.max_concurrency) or 0
        self.gpu_hours_budget = coerce_optional_float(self.gpu_hours_budget) or 0.0
        self.gpu_hours_spent = coerce_optional_float(self.gpu_hours_spent) or 0.0
        self.gpu_hours_reserved = coerce_optional_float(self.gpu_hours_reserved) or 0.0
        if not self.focus and not self.prefer:
            raise ContractValidationError(
                "ResearchAgenda needs at least one focus keyword or prefer rule"
            )
        if self.token_budget is None or self.token_budget <= 0:
            raise ContractValidationError(
                "ResearchAgenda token_budget must be a positive hard cap"
            )
        if self.token_spent < 0 or self.token_reserved < 0:
            raise ContractValidationError("agenda token accounting cannot be negative")
        if self.gpu_hours_budget < 0:
            raise ContractValidationError(
                "gpu_hours_budget must be zero (GPU disabled) or a positive cap"
            )
        if self.gpu_hours_spent < 0 or self.gpu_hours_reserved < 0:
            raise ContractValidationError("agenda GPU accounting cannot be negative")
        if self.max_concurrency <= 0:
            raise ContractValidationError("max_concurrency must be positive")
        if not self.backend_allowlist:
            raise ContractValidationError("backend_allowlist cannot be empty")
        if self.status not in VALID_AGENDA_STATUS:
            raise ContractValidationError(
                f"agenda status must be one of {sorted(VALID_AGENDA_STATUS)}"
            )
        if self.backlog_policy not in VALID_BACKLOG_POLICIES:
            raise ContractValidationError(
                f"backlog_policy must be one of {sorted(VALID_BACKLOG_POLICIES)}"
            )


@dataclass
class AgendaSelection(ContractRecord):
    agenda_id: int = 0
    selected_insight_id: int | None = None
    score: float | None = None
    rationale: str = ""
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    scoring_breakdown: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    auto_research_job_id: int | None = None
    selection_id: int | None = None

    def validate(self) -> None:
        super().validate()
        if self.agenda_id <= 0:
            raise ContractValidationError("AgendaSelection requires agenda_id")
        if self.status not in VALID_SELECTION_STATUS:
            raise ContractValidationError(
                f"selection status must be one of {sorted(VALID_SELECTION_STATUS)}"
            )
        self.selected_insight_id = coerce_optional_int(self.selected_insight_id)
        self.auto_research_job_id = coerce_optional_int(self.auto_research_job_id)
        self.score = coerce_optional_float(self.score)
        self.rejected_candidates = ensure_list(self.rejected_candidates)
        self.scoring_breakdown = ensure_dict(self.scoring_breakdown)


@dataclass
class BudgetReservation(ContractRecord):
    reservation_id: int = 0
    agenda_id: int = 0
    operation: str = ""
    idempotency_key: str = ""
    token_cap: int = 0
    gpu_hours_cap: float = 0.0
    status: str = "reserved"

    def validate(self) -> None:
        super().validate()
        if self.agenda_id <= 0:
            raise ContractValidationError("BudgetReservation requires agenda_id")
        require_non_empty("operation", self.operation)
        require_non_empty("idempotency_key", self.idempotency_key)
        if self.token_cap < 0 or self.gpu_hours_cap < 0:
            raise ContractValidationError("reservation caps cannot be negative")
        if self.token_cap == 0 and self.gpu_hours_cap == 0:
            raise ContractValidationError("reservation must cap at least one resource")
        if self.status not in {"reserved", "settled", "released"}:
            raise ContractValidationError("invalid reservation status")
