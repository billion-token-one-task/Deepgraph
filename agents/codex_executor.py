"""Codex CLI-backed repo editing for experiment iterations."""

from __future__ import annotations

import json



def _recent_no_candidate_diff_streak(history: list[dict], limit: int = 12) -> int:
    streak = 0
    for row in reversed(history[-limit:]):
        judgement = row.get("result_judgement") if isinstance(row.get("result_judgement"), dict) else {}
        text = json.dumps(row, ensure_ascii=False).lower()
        anomaly = str(judgement.get("anomaly_type") or "").strip().lower()
        if anomaly == "no_candidate_diff" or "no candidate code diff" in text:
            streak += 1
        else:
            break
    return streak
