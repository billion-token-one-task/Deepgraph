"""Topic-quality gate: three questions before compute, then surprise buys the
next lane.

Ported behaviour from the historical ``9d24d29`` ("选题三问闸门 + 惊讶度驱动算力
分配,并解禁诚实负结果") onto the current Agenda / Frontier / ResourceGrant
contracts. The historical commit is *not* an ancestor of master and is not
cherry-picked: only the behaviour below is carried over, and two of its
decisions are deliberately reversed.

Carried over
------------
闸一 (zero compute, before anything is spent) asks three questions per
candidate:

1. what do we predict will happen, and how confident are we?
2. do both outcomes lead to the same next action?
3. has this already been published?

Failing any one of them means the experiment cannot pay for itself.

闸二 measures passing in *bits of surprise*: a pilot passes by refuting the
prediction, not by running. A 0.60-confidence prediction that comes true is
0.74 bits and does not clear a 1.0-bit bar; its refutation is 1.32 bits and
does.

闸三's compute ladder: every candidate starts on ``pilot`` with a small slice
of its planned budget and buys the next lane by producing bits.

Deliberately reversed here
--------------------------
* **No LLM inside the gate.** The historical version elicited the prediction
  with an unscoped model call and *passed the candidate through when the model
  was unavailable*. Under the current contracts an ungranted LLM call is
  exactly the failure this recovery is meant to remove, and "provider down =>
  everything passes" is a silent fallback. A candidate with no recorded
  prediction is parked with an auditable reason instead.
* **Agenda-scoped and deterministic.** Every decision is a pure function of the
  candidate row, the agenda, and the policy, so the same inputs always produce
  the same reason codes.

Honest negative results stay valid evidence: nothing in this module rejects a
candidate for predicting, or later producing, a negative outcome. Only absent,
undecidable, or already-known questions are rejected.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from agents.agenda_relevance import (
    agenda_scope_terms,
    candidate_scope_text,
    insight_in_scope,
)
from contracts.agenda import ResearchAgenda


POLICY_VERSION = "topic_gate_v2"

# Stage ladder. A candidate never starts above ``pilot``.
STAGES = ("pilot", "confirm", "full")
STAGE_BUDGET_FRACTION = {"pilot": 0.10, "confirm": 0.35, "full": 1.0}

# Reason codes are an operator contract: they are persisted, shown in the
# dashboard and asserted in tests. Do not rename one without a migration note.
REASON_SCOPE_MISMATCH = "topic_gate_agenda_scope_mismatch"
REASON_REJECT_KEYWORD = "topic_gate_agenda_reject_keyword"
REASON_GENERIC = "topic_gate_generic_statement"
REASON_MISSING_PREDICTION = "topic_gate_prediction_missing"
REASON_ANSWER_KNOWN = "topic_gate_answer_already_known"
REASON_NO_DECISION_RELEVANCE = "topic_gate_no_decision_relevance"
REASON_ALREADY_PUBLISHED = "topic_gate_already_published"
REASON_DUPLICATE_OR_OBSOLETE = "topic_gate_duplicate_or_obsolete"
REASON_NOT_FALSIFIABLE = "topic_gate_not_falsifiable"
REASON_NO_CHEAP_DECISIVE_EXPERIMENT = "topic_gate_no_decisive_low_cost_experiment"
REASON_EXPECTED_INFORMATION_TOO_LOW = "topic_gate_expected_information_too_low"

_NO_ACTION_MARKERS = {
    "",
    "none",
    "nothing",
    "no action",
    "no-op",
    "n/a",
    "na",
    "tbd",
    "unknown",
    "不做",
    "无",
    "没有",
}
_PUBLISHED_YES = {"yes", "true", "published", "exists"}
_OBSOLETE_STATUSES = {"exists", "duplicate", "obsolete", "solved"}


class TopicGateError(ValueError):
    """Raised when the gate is handed an unusable policy or candidate."""


@dataclass(frozen=True)
class TopicGatePolicy:
    """Thresholds. Every default is a deliberate refusal to spend."""

    max_confidence: float = 0.90
    surprise_bits: float = 1.0
    min_expected_bits: float = 0.25
    min_statement_chars: int = 40
    max_pilot_tokens: int = 20_000
    max_pilot_gpu_hours: float = 0.0
    max_pilot_wall_hours: float = 24.0

    def validate(self) -> None:
        if not 0 < self.max_confidence <= 1:
            raise TopicGateError("max_confidence must be within (0, 1]")
        if self.surprise_bits <= 0:
            raise TopicGateError("surprise_bits must be positive")
        if self.min_expected_bits < 0:
            raise TopicGateError("min_expected_bits cannot be negative")
        if self.min_statement_chars < 0:
            raise TopicGateError("min_statement_chars cannot be negative")
        if self.max_pilot_tokens <= 0:
            raise TopicGateError("max_pilot_tokens must be positive")
        if self.max_pilot_gpu_hours < 0 or self.max_pilot_wall_hours <= 0:
            raise TopicGateError("pilot cost caps are invalid")


def policy_from_config() -> TopicGatePolicy:
    """Build the active policy from configuration thresholds."""
    from config import (
        TOPIC_GATE_MAX_CONFIDENCE,
        TOPIC_GATE_MAX_PILOT_GPU_HOURS,
        TOPIC_GATE_MAX_PILOT_TOKENS,
        TOPIC_GATE_MAX_PILOT_WALL_HOURS,
        TOPIC_GATE_MIN_EXPECTED_BITS,
        TOPIC_GATE_MIN_STATEMENT_CHARS,
        TOPIC_GATE_SURPRISE_BITS,
    )

    policy = TopicGatePolicy(
        max_confidence=float(TOPIC_GATE_MAX_CONFIDENCE),
        surprise_bits=float(TOPIC_GATE_SURPRISE_BITS),
        min_expected_bits=float(TOPIC_GATE_MIN_EXPECTED_BITS),
        min_statement_chars=int(TOPIC_GATE_MIN_STATEMENT_CHARS),
        max_pilot_tokens=int(TOPIC_GATE_MAX_PILOT_TOKENS),
        max_pilot_gpu_hours=float(TOPIC_GATE_MAX_PILOT_GPU_HOURS),
        max_pilot_wall_hours=float(TOPIC_GATE_MAX_PILOT_WALL_HOURS),
    )
    policy.validate()
    return policy


@dataclass(frozen=True)
class TopicGateDecision:
    passed: bool
    reason_codes: tuple[str, ...] = ()
    blockers: tuple[dict[str, str], ...] = ()
    expected_bits: float = 0.0
    refute_bits: float = 0.0
    confidence: float | None = None
    policy_version: str = POLICY_VERSION
    prediction: dict[str, Any] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "blockers": [dict(item) for item in self.blockers],
            "expected_bits": self.expected_bits,
            "refute_bits": self.refute_bits,
            "confidence": self.confidence,
            "policy_version": self.policy_version,
        }


# ---------- information-theory helpers ----------


def surprisal_bits(probability: float) -> float:
    """Bits of surprise from observing an event we gave ``probability``."""
    bounded = min(max(float(probability), 1e-6), 1.0)
    return -math.log2(bounded)


def binary_entropy(probability: float) -> float:
    """Expected bits from a yes/no question we answer correctly with ``p``."""
    bounded = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return -(bounded * math.log2(bounded) + (1 - bounded) * math.log2(1 - bounded))


# ---------- normalisation ----------


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalized_action(value: Any) -> str:
    text = _text(value).lower()
    for char in ".。,，;；!！?？ \t\n":
        text = text.replace(char, " ")
    return " ".join(text.split())


def _mapping(raw: Any) -> Mapping[str, Any] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else None
        except json.JSONDecodeError:
            return None
    return raw if isinstance(raw, Mapping) else None


def gate_record(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Read the candidate's recorded gate answers. Never invents them."""
    record = _mapping(candidate.get("topic_gate_json"))
    if record is None:
        record = _mapping(candidate.get("topic_gate"))
    return dict(record or {})


def normalize_prediction(raw: Any) -> dict[str, Any] | None:
    """Coerce a stored prediction record into the gate's shape."""
    mapping = _mapping(raw)
    if mapping is None:
        return None
    try:
        confidence = float(mapping.get("confidence"))
    except (TypeError, ValueError):
        confidence = None
    return {
        "predicted_outcome": _text(
            mapping.get("predicted_outcome") or mapping.get("outcome")
        ),
        "confidence": confidence,
        "action_if_confirmed": _text(mapping.get("action_if_confirmed")),
        "action_if_refuted": _text(mapping.get("action_if_refuted")),
        "already_published": _text(mapping.get("already_published")).lower() or "unsure",
        "already_published_evidence": _text(mapping.get("already_published_evidence")),
    }


# ---------- 闸一: three questions, zero compute ----------


def _scope_blockers(
    candidate: Mapping[str, Any],
    agenda: ResearchAgenda,
    policy: TopicGatePolicy,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if not insight_in_scope(dict(candidate), agenda):
        blockers.append(
            {
                "question": "agenda_scope",
                "code": REASON_SCOPE_MISMATCH,
                "reason": "candidate is not bound to this agenda's scope",
            }
        )
        return blockers
    text = candidate_scope_text(dict(candidate))
    for phrase in sorted(str(value) for value in (agenda.reject or {}).get("keywords") or []):
        if phrase.lower() in text:
            blockers.append(
                {
                    "question": "agenda_scope",
                    "code": REASON_REJECT_KEYWORD,
                    "reason": f"agenda reject keyword matched:{phrase}",
                }
            )
    statement = _text(candidate.get("problem_statement")) or _text(
        candidate.get("title")
    )
    scope_hits = [term for term in agenda_scope_terms(agenda) if term in text]
    if len(statement) < policy.min_statement_chars or not scope_hits:
        blockers.append(
            {
                "question": "specificity",
                "code": REASON_GENERIC,
                "reason": (
                    "problem statement is too generic to be decisive "
                    f"({len(statement)} chars, {len(scope_hits)} scope terms)"
                ),
            }
        )
    return blockers


def _prior_work_blockers(
    candidate: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    novelty = _text(candidate.get("novelty_status")).lower()
    if novelty in _OBSOLETE_STATUSES:
        blockers.append(
            {
                "question": "prior_work",
                "code": REASON_DUPLICATE_OR_OBSOLETE,
                "reason": f"novelty_status={novelty}",
            }
        )
    if prediction["already_published"] in _PUBLISHED_YES:
        evidence = prediction["already_published_evidence"] or "no citation given"
        blockers.append(
            {
                "question": "prior_work",
                "code": REASON_ALREADY_PUBLISHED,
                "reason": f"already published ({evidence})",
            }
        )
    return blockers


def _falsifiability_blockers(
    record: Mapping[str, Any],
    policy: TopicGatePolicy,
) -> list[dict[str, str]]:
    """A decisive, cheap experiment must exist before anything is granted."""
    experiment = _mapping(record.get("minimum_falsification_experiment")) or {}
    blockers: list[dict[str, str]] = []
    missing = [
        field_name
        for field_name in ("metric", "baseline", "decisive_comparison")
        if not _text(experiment.get(field_name))
    ]
    if missing:
        blockers.append(
            {
                "question": "falsifiability",
                "code": REASON_NOT_FALSIFIABLE,
                "reason": "minimum falsification experiment missing:"
                + ",".join(sorted(missing)),
            }
        )
        return blockers
    cost = _mapping(experiment.get("estimated_cost")) or {}
    try:
        tokens = int(float(cost.get("tokens") or 0))
        gpu_hours = float(cost.get("gpu_hours") or 0)
        wall_hours = float(cost.get("wall_hours") or 0)
    except (TypeError, ValueError):
        blockers.append(
            {
                "question": "falsifiability",
                "code": REASON_NO_CHEAP_DECISIVE_EXPERIMENT,
                "reason": "estimated cost is not numeric",
            }
        )
        return blockers
    if tokens <= 0 and gpu_hours <= 0 and wall_hours <= 0:
        blockers.append(
            {
                "question": "falsifiability",
                "code": REASON_NO_CHEAP_DECISIVE_EXPERIMENT,
                "reason": "no estimated cost recorded for the decisive experiment",
            }
        )
        return blockers
    over = []
    if tokens > policy.max_pilot_tokens:
        over.append(f"tokens={tokens}>{policy.max_pilot_tokens}")
    if gpu_hours > policy.max_pilot_gpu_hours:
        over.append(f"gpu_hours={gpu_hours}>{policy.max_pilot_gpu_hours}")
    if wall_hours > policy.max_pilot_wall_hours:
        over.append(f"wall_hours={wall_hours}>{policy.max_pilot_wall_hours}")
    if over:
        blockers.append(
            {
                "question": "falsifiability",
                "code": REASON_NO_CHEAP_DECISIVE_EXPERIMENT,
                "reason": "decisive experiment is not cheap enough for a pilot:"
                + ",".join(over),
            }
        )
    return blockers


def screen_candidate(
    candidate: Mapping[str, Any],
    agenda: ResearchAgenda,
    *,
    policy: TopicGatePolicy | None = None,
) -> TopicGateDecision:
    """闸一 for one agenda-bound candidate. Pure: no LLM, no database."""
    active_policy = policy or TopicGatePolicy()
    active_policy.validate()
    agenda.validate()

    blockers: list[dict[str, str]] = _scope_blockers(candidate, agenda, active_policy)
    record = gate_record(candidate)
    prediction = normalize_prediction(record.get("prediction"))

    if (
        not prediction
        or not prediction["predicted_outcome"]
        or prediction["confidence"] is None
    ):
        blockers.append(
            {
                "question": "prediction",
                "code": REASON_MISSING_PREDICTION,
                "reason": (
                    "no recorded prediction and confidence; the gate never "
                    "elicits one with an ungranted model call"
                ),
            }
        )
        return _decision(blockers, prediction, 0.0, 0.0, None)

    confidence = min(max(float(prediction["confidence"]), 0.0), 1.0)
    expected_bits = round(binary_entropy(confidence), 4)
    refute_bits = round(surprisal_bits(1.0 - confidence), 4)

    if confidence >= active_policy.max_confidence:
        blockers.append(
            {
                "question": "prediction",
                "code": REASON_ANSWER_KNOWN,
                "reason": (
                    f"confidence={confidence:.2f} >= "
                    f"{active_policy.max_confidence:.2f}: the answer is already "
                    "known, running it buys no information"
                ),
            }
        )

    action_confirmed = _normalized_action(prediction["action_if_confirmed"])
    action_refuted = _normalized_action(prediction["action_if_refuted"])
    if not action_confirmed or not action_refuted:
        blockers.append(
            {
                "question": "decision_relevance",
                "code": REASON_NO_DECISION_RELEVANCE,
                "reason": "next action undeclared for at least one outcome",
            }
        )
    elif action_confirmed == action_refuted or (
        action_confirmed in _NO_ACTION_MARKERS and action_refuted in _NO_ACTION_MARKERS
    ):
        blockers.append(
            {
                "question": "decision_relevance",
                "code": REASON_NO_DECISION_RELEVANCE,
                "reason": "both outcomes lead to the same next action",
            }
        )

    blockers.extend(_prior_work_blockers(candidate, prediction))
    blockers.extend(_falsifiability_blockers(record, active_policy))

    if expected_bits < active_policy.min_expected_bits:
        blockers.append(
            {
                "question": "information_value",
                "code": REASON_EXPECTED_INFORMATION_TOO_LOW,
                "reason": (
                    f"expected_bits={expected_bits:.2f} < "
                    f"{active_policy.min_expected_bits:.2f}"
                ),
            }
        )

    return _decision(blockers, prediction, expected_bits, refute_bits, confidence)


def _decision(
    blockers: list[dict[str, str]],
    prediction: dict[str, Any] | None,
    expected_bits: float,
    refute_bits: float,
    confidence: float | None,
) -> TopicGateDecision:
    ordered = sorted(blockers, key=lambda item: (item["code"], item["reason"]))
    codes: list[str] = []
    for blocker in ordered:
        if blocker["code"] not in codes:
            codes.append(blocker["code"])
    return TopicGateDecision(
        passed=not ordered,
        reason_codes=tuple(codes),
        blockers=tuple(ordered),
        expected_bits=expected_bits,
        refute_bits=refute_bits,
        confidence=confidence,
        prediction=prediction,
    )


# ---------- 闸二: the pilot passes by refuting, not by running ----------


def observed_surprise_bits(
    *,
    confidence: float,
    outcome: str,
) -> float:
    """Bits produced by an observed pilot outcome under the prior confidence."""
    bounded = min(max(float(confidence), 0.0), 1.0)
    normalized = _text(outcome).lower()
    if normalized == "refuted":
        return round(surprisal_bits(1.0 - bounded), 4)
    if normalized == "confirmed":
        return round(surprisal_bits(bounded), 4)
    return 0.0


def escalation_verdict(
    prediction: Any,
    observation: Mapping[str, Any],
    *,
    policy: TopicGatePolicy | None = None,
) -> dict[str, Any]:
    """闸二: may this candidate buy the next compute lane?

    ``observation`` keys: ``ran`` (bool), ``outcome``
    (confirmed|refuted|inconclusive), ``attribution_control``
    (passed|failed|missing).

    A refuted prediction is a *pass*, not a failure: honest negative results are
    evidence. Only unattributable or undecided pilots are invalid.
    """
    active_policy = policy or TopicGatePolicy()
    active_policy.validate()
    normalized = normalize_prediction(prediction) or {}
    confidence = normalized.get("confidence")
    confidence = 0.5 if confidence is None else min(max(float(confidence), 0.0), 1.0)
    outcome = _text(observation.get("outcome")).lower()
    control = _text(observation.get("attribution_control")).lower() or "missing"

    if not observation.get("ran", False):
        return {
            "verdict": "invalid",
            "surprise_bits": 0.0,
            "reason_codes": ("pilot_did_not_run",),
        }
    if control == "failed":
        return {
            "verdict": "invalid",
            "surprise_bits": 0.0,
            "reason_codes": ("effect_not_attributable",),
        }
    if outcome not in {"confirmed", "refuted"}:
        return {
            "verdict": "inconclusive",
            "surprise_bits": 0.0,
            "reason_codes": ("prediction_not_decided",),
        }

    bits = observed_surprise_bits(confidence=confidence, outcome=outcome)
    reasons = [f"prediction_{outcome}", f"surprise_bits={bits:.2f}"]
    if control == "missing":
        reasons.append("attribution_control_not_run")
    if bits >= active_policy.surprise_bits:
        return {
            "verdict": "escalate",
            "surprise_bits": bits,
            "reason_codes": tuple(reasons),
        }
    reasons.append("running_is_not_passing")
    return {"verdict": "stop", "surprise_bits": bits, "reason_codes": tuple(reasons)}


# ---------- compute follows surprise ----------


def stage_token_cap(planned_tokens: int, *, stage: str) -> int:
    """Cap a stage's token request to its slice of the planned budget."""
    if stage not in STAGE_BUDGET_FRACTION:
        raise TopicGateError(f"unknown stage:{stage}")
    planned = int(planned_tokens)
    if planned <= 0:
        raise TopicGateError("planned token budget must be positive")
    return max(1, int(planned * STAGE_BUDGET_FRACTION[stage]))


def next_stage(stage: str) -> str | None:
    """The lane a candidate may buy next, or None at the top of the ladder."""
    if stage not in STAGES:
        raise TopicGateError(f"unknown stage:{stage}")
    index = STAGES.index(stage)
    return STAGES[index + 1] if index + 1 < len(STAGES) else None
