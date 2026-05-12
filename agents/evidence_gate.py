"""Evidence gate for the agenda loop (issue #9).

Issue #9 explicitly requires:

  "evidence gate 产出 pass/block 报告，并明确列出 blockers."
  "manuscript bundle 只有在 evidence gate 允许时才生成."
  "新测试至少覆盖 ... manuscript/review allowed 与 blocked 两种行为."

This module is the gate — separate from the reviewer adapter — that runs
BEFORE manuscript creation. It consumes the structured outputs of a real
experiment_run (experiment_result_packet, experimental_claims) and emits
a persisted decision artifact:

    {
      "status": "pass" | "block",
      "blockers": [{"requirement": "...", "reason": "..."}],
      "metrics_summary": {...},
      "packet_path": "...",
      "rule_set": "agenda_v1_default"
    }

Default rule set (``agenda_v1_default``) blocks when ANY of:
- no experiment_run is linked to the selection
- experiment_run is not completed
- no confirmed experimental_claim exists
- a refuted claim exists for a metric the agenda's ``required_output`` lists
- no experiment_result_packet.json artifact found on disk
- result packet missing required keys: config, softmax_attention (or
  baseline), linear_attention (or candidate), delta

Public API:
    evaluate_gate(selection_id) -> dict          # in-memory decision
    run_gate(selection_id) -> dict               # decide + persist
    get_latest_gate(selection_id) -> dict | None
    get_gate(gate_id) -> dict | None
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from db import database as db


# ---------- data assembly ----------


def _selection_row(selection_id: int) -> dict[str, Any]:
    row = db.fetchone("SELECT * FROM agenda_selections WHERE id=?", (selection_id,))
    if not row:
        raise ValueError(f"selection {selection_id} not found")
    return row


def _experiment_run(run_id: int | None) -> dict[str, Any] | None:
    if not run_id:
        return None
    return db.fetchone(
        """
        SELECT id, deep_insight_id, status, phase, hypothesis_verdict,
               baseline_metric_name, baseline_metric_value, best_metric_value,
               effect_size, effect_pct, workdir, error_message
        FROM experiment_runs WHERE id=?
        """,
        (run_id,),
    )


def _claims_for_run(run_id: int | None) -> list[dict[str, Any]]:
    if not run_id:
        return []
    return db.fetchall(
        """
        SELECT id, claim_text, claim_type, verdict, effect_size, confidence,
               p_value, supporting_data
        FROM experimental_claims WHERE run_id=? ORDER BY id
        """,
        (run_id,),
    )


def _agenda_required_output(agenda_id: int | None) -> dict[str, Any]:
    if not agenda_id:
        return {}
    row = db.fetchone(
        "SELECT required_output_json FROM research_agendas WHERE id=?",
        (agenda_id,),
    )
    if not row:
        return {}
    raw = row.get("required_output_json")
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}
    return raw or {}


def _load_packet(workdir: str | None) -> tuple[dict[str, Any] | None, str | None]:
    if not workdir:
        return None, None
    path = Path(workdir) / "experiment_result_packet.json"
    if not path.exists():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), str(path)
    except (json.JSONDecodeError, OSError):
        return None, str(path)


# ---------- gate rules ----------


REQUIRED_PACKET_KEYS = ("config", "softmax_attention", "linear_attention", "delta")


def _evaluate_default_rules(
    selection: Mapping[str, Any],
    experiment: Mapping[str, Any] | None,
    claims: list[dict[str, Any]],
    packet: Mapping[str, Any] | None,
    packet_path: str | None,
    agenda_required: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []

    if not experiment:
        blockers.append({"requirement": "experiment_run", "reason": "not_linked"})
        return _finalize(blockers, experiment, claims, packet, packet_path)

    if (experiment.get("status") or "").lower() != "completed":
        blockers.append({
            "requirement": "experiment_run.status=completed",
            "reason": f"current={experiment.get('status')}",
        })

    verdicts = [(c.get("verdict") or "").lower() for c in claims]
    confirmed = sum(1 for v in verdicts if v == "confirmed")
    refuted = sum(1 for v in verdicts if v == "refuted")
    if confirmed == 0:
        blockers.append({
            "requirement": "experimental_claims.confirmed>=1",
            "reason": f"confirmed={confirmed}",
        })

    # agenda_v1_default unconditionally requires the experiment_result_packet
    # on disk. (Future rule sets may make this conditional on agenda_required
    # declaring "experiment_result_packet" or "evidence_manifest" keys.)
    if not packet:
        blockers.append({
            "requirement": "experiment_result_packet.json",
            "reason": "missing_or_unreadable",
            "looked_at": packet_path or "(no workdir)",
        })
    else:
        missing_keys = [k for k in REQUIRED_PACKET_KEYS if k not in packet]
        if missing_keys:
            blockers.append({
                "requirement": "experiment_result_packet keys",
                "reason": f"missing_keys={missing_keys}",
            })

    # A refuted claim about the primary effect blocks publication.
    if refuted > 0:
        blockers.append({
            "requirement": "no_refuted_primary_claims",
            "reason": f"refuted_count={refuted}",
        })

    return _finalize(blockers, experiment, claims, packet, packet_path)


def _finalize(
    blockers: list[dict[str, Any]],
    experiment: Mapping[str, Any] | None,
    claims: list[dict[str, Any]],
    packet: Mapping[str, Any] | None,
    packet_path: str | None,
) -> dict[str, Any]:
    counts = {"confirmed": 0, "refuted": 0, "inconclusive": 0, "other": 0}
    for c in claims:
        v = (c.get("verdict") or "").lower()
        counts[v if v in counts else "other"] += 1
    status = "block" if blockers else "pass"
    metrics_summary: dict[str, Any] = {
        "claim_counts": counts,
        "experiment_status": (experiment or {}).get("status"),
        "hypothesis_verdict": (experiment or {}).get("hypothesis_verdict"),
        "effect_size": (experiment or {}).get("effect_size"),
        "effect_pct": (experiment or {}).get("effect_pct"),
    }
    if packet:
        delta = packet.get("delta") or {}
        metrics_summary["latency_speedup_x"] = delta.get("latency_speedup_x")
        metrics_summary["approximation_relative_error"] = delta.get("relative_error")
    return {
        "status": status,
        "blockers": blockers,
        "metrics_summary": metrics_summary,
        "packet_path": packet_path,
        "rule_set": "agenda_v1_default",
    }


# ---------- public API ----------


def evaluate_gate(selection_id: int) -> dict[str, Any]:
    """Compute (but do not persist) the gate decision."""
    sel = _selection_row(selection_id)
    exp = _experiment_run(sel.get("experiment_run_id"))
    claims = _claims_for_run(sel.get("experiment_run_id"))
    packet, packet_path = _load_packet((exp or {}).get("workdir"))
    required = _agenda_required_output(sel.get("agenda_id"))
    return _evaluate_default_rules(sel, exp, claims, packet, packet_path, required)


def run_gate(selection_id: int) -> dict[str, Any]:
    """Compute the gate decision and persist it to ``agenda_evidence_gates``."""
    sel = _selection_row(selection_id)
    decision = evaluate_gate(selection_id)
    gate_id = db.insert_returning_id(
        """
        INSERT INTO agenda_evidence_gates
            (selection_id, experiment_run_id, status, blockers_json,
             metrics_summary_json, packet_path, rule_set)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        (
            int(selection_id),
            sel.get("experiment_run_id"),
            decision["status"],
            json.dumps(decision["blockers"], ensure_ascii=False),
            json.dumps(decision["metrics_summary"], ensure_ascii=False),
            decision.get("packet_path"),
            decision["rule_set"],
        ),
    )
    db.commit()
    decision["id"] = gate_id
    return decision


def _row_to_gate(row: Mapping[str, Any]) -> dict[str, Any]:
    def _decode(field: str, default: Any) -> Any:
        v = row.get(field)
        if isinstance(v, str):
            try:
                return json.loads(v) if v else default
            except json.JSONDecodeError:
                return default
        return v if v is not None else default

    return {
        "id": row.get("id"),
        "selection_id": row.get("selection_id"),
        "experiment_run_id": row.get("experiment_run_id"),
        "status": row.get("status"),
        "blockers": _decode("blockers_json", []),
        "metrics_summary": _decode("metrics_summary_json", {}),
        "packet_path": row.get("packet_path"),
        "rule_set": row.get("rule_set"),
        "created_at": row.get("created_at"),
    }


def get_gate(gate_id: int) -> dict[str, Any] | None:
    row = db.fetchone("SELECT * FROM agenda_evidence_gates WHERE id=?", (gate_id,))
    return _row_to_gate(row) if row else None


def get_latest_gate(selection_id: int) -> dict[str, Any] | None:
    row = db.fetchone(
        """
        SELECT * FROM agenda_evidence_gates
        WHERE selection_id=? ORDER BY created_at DESC, id DESC LIMIT 1
        """,
        (selection_id,),
    )
    return _row_to_gate(row) if row else None
