"""Pure scientific-evidence and presentation-integrity contracts.

This module has no database, configuration, network, or application imports so
it can be evaluated safely in an isolated test lane.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from contracts.base import ContractRecord, ContractValidationError


SCIENTIFIC_VERDICTS = {"supported", "refuted", "inconclusive", "invalid"}
CLAIM_STRENGTH_ORDER = {
    "none": 0,
    "descriptive": 1,
    "pilot_observation": 2,
    "bounded_supported": 3,
    "general_superiority": 4,
}


@dataclass
class EvidenceDecisionInput(ContractRecord):
    verdict: str = "inconclusive"
    p_value: float | None = None
    alpha: float = 0.05
    metric_value: float | None = None
    baseline_value: float | None = None
    full_benchmark_complete: bool = False
    raw_artifacts_complete: bool = False
    claim_ledger_complete: bool = False
    evaluator_id: str = ""
    failure_reason: str = ""

    def validate(self) -> None:
        super().validate()
        if self.verdict not in SCIENTIFIC_VERDICTS:
            raise ContractValidationError(
                f"verdict must be one of {sorted(SCIENTIFIC_VERDICTS)}"
            )
        if not 0 < float(self.alpha) < 1:
            raise ContractValidationError("alpha must be between 0 and 1")
        for name in ("p_value", "metric_value", "baseline_value"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ContractValidationError(f"{name} must be finite")
        if self.p_value is not None and not 0 <= float(self.p_value) <= 1:
            raise ContractValidationError("p_value must be between 0 and 1")


@dataclass
class EvidenceDecision(ContractRecord):
    significant: bool = False
    positive_claim_allowed: bool = False
    confirmation_allowed: bool = False
    max_claim_strength: str = "none"
    blockers: list[str] = field(default_factory=list)

    def validate(self) -> None:
        super().validate()
        if self.max_claim_strength not in CLAIM_STRENGTH_ORDER:
            raise ContractValidationError("unknown max_claim_strength")
        if self.confirmation_allowed and not self.positive_claim_allowed:
            raise ContractValidationError(
                "confirmation cannot be allowed when positive claims are blocked"
            )


def decide_evidence(payload: EvidenceDecisionInput) -> EvidenceDecision:
    """Apply fail-closed M1/M4 evidence rules."""
    payload.validate()
    blockers: list[str] = []

    if payload.verdict == "refuted":
        blockers.append("evaluator_refuted")
    elif payload.verdict == "inconclusive":
        blockers.append("evaluator_inconclusive")
    elif payload.verdict == "invalid":
        blockers.append("evaluator_invalid")

    if payload.metric_value is None:
        blockers.append("metric_missing")
    if payload.baseline_value is None:
        blockers.append("baseline_missing")
    elif float(payload.baseline_value) == 0:
        blockers.append("baseline_zero")
    if not payload.full_benchmark_complete:
        blockers.append("full_benchmark_incomplete")
    if not payload.raw_artifacts_complete:
        blockers.append("raw_artifacts_incomplete")
    if not payload.claim_ledger_complete:
        blockers.append("claim_ledger_incomplete")
    if not payload.evaluator_id.strip():
        blockers.append("independent_evaluator_missing")
    if payload.p_value is None:
        blockers.append("p_value_missing")

    significant = (
        payload.p_value is not None
        and float(payload.p_value) < float(payload.alpha)
        and payload.verdict == "supported"
    )
    if payload.p_value is not None and not significant:
        blockers.append("not_significant")

    positive_allowed = payload.verdict == "supported"
    complete = not any(
        blocker
        in {
            "metric_missing",
            "baseline_missing",
            "baseline_zero",
            "full_benchmark_incomplete",
            "raw_artifacts_incomplete",
            "claim_ledger_incomplete",
            "independent_evaluator_missing",
        }
        for blocker in blockers
    )
    confirmation_allowed = positive_allowed and complete and significant

    if payload.verdict == "refuted":
        max_strength = "none"
        positive_allowed = False
        confirmation_allowed = False
    elif confirmation_allowed:
        max_strength = "bounded_supported"
    elif positive_allowed and payload.metric_value is not None:
        max_strength = "descriptive"
    else:
        max_strength = "none"

    decision = EvidenceDecision(
        significant=significant,
        positive_claim_allowed=positive_allowed,
        confirmation_allowed=confirmation_allowed,
        max_claim_strength=max_strength,
        blockers=list(dict.fromkeys(blockers)),
    )
    decision.validate()
    return decision


_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?%?"
)
_STRENGTH_MARKERS = (
    ("general_superiority", ("state-of-the-art", "sota", "universally superior", "general superiority")),
    ("bounded_supported", ("statistically significant", "confirmed improvement", "evidence supports")),
    ("pilot_observation", ("pilot suggests", "preliminary improvement")),
    ("descriptive", ("point estimate", "observed", "measured")),
)


def numeric_tokens(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(str(text or "")))


def inferred_claim_strength(text: str) -> str:
    lower = str(text or "").lower()
    for strength, markers in _STRENGTH_MARKERS:
        if any(marker in lower for marker in markers):
            return strength
    return "none"


@dataclass
class PresentationAudit(ContractRecord):
    passed: bool = False
    introduced_numbers: list[str] = field(default_factory=list)
    source_claim_strength: str = "none"
    rendered_claim_strength: str = "none"
    blockers: list[str] = field(default_factory=list)


def audit_presentation_transform(source: str, rendered: str) -> PresentationAudit:
    """Ensure layout/polish cannot create numbers or strengthen a claim."""
    introduced = sorted(numeric_tokens(rendered) - numeric_tokens(source))
    source_strength = inferred_claim_strength(source)
    rendered_strength = inferred_claim_strength(rendered)
    blockers: list[str] = []
    if introduced:
        blockers.append("presentation_introduced_numeric_tokens")
    if CLAIM_STRENGTH_ORDER[rendered_strength] > CLAIM_STRENGTH_ORDER[source_strength]:
        blockers.append("presentation_strengthened_claim")
    return PresentationAudit(
        passed=not blockers,
        introduced_numbers=introduced,
        source_claim_strength=source_strength,
        rendered_claim_strength=rendered_strength,
        blockers=blockers,
    )
