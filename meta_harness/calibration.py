"""Transparent calibration reports from predicted packets and actual outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

from contracts.meta_harness import IdeaDecisionPacket, OutcomeRecord


@dataclass(frozen=True)
class CalibrationRow:
    idea_id: int
    predicted_success: float
    observed_success: float
    predicted_tokens: float
    actual_tokens: int
    predicted_gpu_hours: float
    actual_gpu_hours: float
    predicted_impact: float
    observed_effect: float | None


def join_predictions(
    decisions: Iterable[IdeaDecisionPacket],
    outcomes: Iterable[OutcomeRecord],
) -> list[CalibrationRow]:
    by_key = {}
    for packet in decisions:
        packet.validate()
        by_key[(packet.agenda_id, packet.idea_id)] = packet
    rows: list[CalibrationRow] = []
    for outcome in outcomes:
        outcome.validate()
        packet = by_key.get((outcome.agenda_id, outcome.idea_id))
        if packet is None:
            continue
        rows.append(
            CalibrationRow(
                idea_id=outcome.idea_id,
                predicted_success=packet.success_probability.value,
                observed_success=1.0 if outcome.verdict == "supported" else 0.0,
                predicted_tokens=packet.expected_token_cost.value,
                actual_tokens=outcome.actual_tokens,
                predicted_gpu_hours=packet.expected_gpu_cost.value,
                actual_gpu_hours=outcome.actual_gpu_hours,
                predicted_impact=packet.expected_impact.value,
                observed_effect=outcome.effect,
            )
        )
    return rows


def build_calibration_report(
    decisions: Iterable[IdeaDecisionPacket],
    outcomes: Iterable[OutcomeRecord],
) -> dict:
    rows = join_predictions(decisions, outcomes)
    if not rows:
        return {
            "sample_count": 0,
            "status": "insufficient_data",
            "policy_update_allowed": False,
        }
    brier = sum(
        (row.predicted_success - row.observed_success) ** 2 for row in rows
    ) / len(rows)
    token_mae = sum(
        abs(row.predicted_tokens - row.actual_tokens) for row in rows
    ) / len(rows)
    gpu_mae = sum(
        abs(row.predicted_gpu_hours - row.actual_gpu_hours) for row in rows
    ) / len(rows)
    effect_rows = [row for row in rows if row.observed_effect is not None]
    impact_rmse = (
        sqrt(
            sum(
                (row.predicted_impact - float(row.observed_effect)) ** 2
                for row in effect_rows
            )
            / len(effect_rows)
        )
        if effect_rows
        else None
    )
    return {
        "sample_count": len(rows),
        "status": "reported",
        "success_brier_score": round(brier, 8),
        "token_cost_mae": round(token_mae, 4),
        "gpu_hours_mae": round(gpu_mae, 6),
        "impact_rmse": round(impact_rmse, 8) if impact_rmse is not None else None,
        "policy_update_allowed": False,
        "policy_update_reason": "report_only_until_reviewer_approval",
        "rows": [row.__dict__ for row in rows],
    }
