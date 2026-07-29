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

Default rule set (``agenda_v2_default``) blocks when ANY of:
- no experiment_run is linked to the selection
- experiment_run is not completed
- no experimental_claim was decided either way (nothing confirmed, nothing
  refuted) — an undecided run has nothing to write up
- a refuted claim covers a metric the agenda's ``required_output`` promised a
  confirmed result on: the manuscript's claim and the data disagree
- no experiment_result_packet.json artifact found on disk
- result packet missing the keys the agenda declares (default: config,
  baseline, candidate, delta; domain aliases such as softmax_attention /
  linear_attention still satisfy baseline / candidate)

A refuted claim on its own is a finding, not a failure. The decision carries
``claim_stance`` (positive | negative | mixed) so the manuscript step writes
the result the data actually supports instead of the one we hoped for.

``required_output`` keys this gate reads (all optional):
    packet_keys: [...]          # what the result packet must contain
    confirmed_metrics: [...]    # metrics whose refutation blocks publication
    relative_error_max: 0.10    # only applied to approximation-style packets

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


DEFAULT_PACKET_KEYS = ("config", "baseline", "candidate", "delta")

# A packet may name its two arms in domain terms. The gate asks for roles, not
# for the vocabulary of the first experiment we ever ran through it.
PACKET_KEY_ALIASES = {
    "baseline": ("baseline", "control", "reference", "softmax_attention"),
    "candidate": ("candidate", "treatment", "proposed", "linear_attention"),
}

# Magnitude threshold for approximation-style packets: an approximation with
# > 10% relative error against its reference is inconclusive and must not flow
# into manuscript generation. Audit finding (PR #10 review): without this rule
# the gate was greenlighting bundles for results the reviewer simultaneously
# flagged as inconclusive (rel_err=0.767 → status=pass in the acceptance run).
# Only applied when the packet reports delta.relative_error at all.
RELATIVE_ERROR_MAX = 0.10

RULE_SET = "agenda_v2_default"


def _required_packet_keys(agenda_required: Mapping[str, Any]) -> tuple[str, ...]:
    declared = agenda_required.get("packet_keys")
    if isinstance(declared, (list, tuple)) and declared:
        return tuple(str(k).strip() for k in declared if str(k).strip())
    return DEFAULT_PACKET_KEYS


def _missing_packet_keys(packet: Mapping[str, Any], keys: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for key in keys:
        accepted = PACKET_KEY_ALIASES.get(key, (key,))
        if not any(alias in packet for alias in accepted):
            missing.append(key)
    return missing


def _relative_error_max(agenda_required: Mapping[str, Any]) -> float:
    raw = agenda_required.get("relative_error_max")
    try:
        return float(raw) if raw is not None else RELATIVE_ERROR_MAX
    except (TypeError, ValueError):
        return RELATIVE_ERROR_MAX


def _promised_metrics(agenda_required: Mapping[str, Any]) -> list[str]:
    raw = agenda_required.get("confirmed_metrics")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(m).strip().lower() for m in raw if str(m).strip()]


def _claim_mentions(claim: Mapping[str, Any], metric: str) -> bool:
    haystack = f"{claim.get('claim_text') or ''} {claim.get('supporting_data') or ''}".lower()
    return metric in haystack


def _claim_stance(confirmed: int, refuted: int) -> str:
    if confirmed and refuted:
        return "mixed"
    if refuted:
        return "negative"
    if confirmed:
        return "positive"
    return "undecided"


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
    # A refuted claim is a result. A run that decided nothing is not.
    if confirmed + refuted == 0:
        blockers.append({
            "requirement": "experimental_claims.decided>=1",
            "reason": f"confirmed={confirmed}, refuted={refuted}",
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
        missing_keys = _missing_packet_keys(packet, _required_packet_keys(agenda_required))
        if missing_keys:
            blockers.append({
                "requirement": "experiment_result_packet keys",
                "reason": f"missing_keys={missing_keys}",
            })
        # Magnitude check: even when every required key is present, an
        # approximation's error must be small enough that the result is not
        # self-evidently inconclusive.
        delta = packet.get("delta") or {}
        rel_err = delta.get("relative_error")
        rel_err_max = _relative_error_max(agenda_required)
        if isinstance(rel_err, (int, float)) and rel_err > rel_err_max:
            blockers.append({
                "requirement": f"delta.relative_error<={rel_err_max}",
                "reason": f"observed={rel_err}",
            })

    # A refuted claim is publishable as a negative result. What is not
    # publishable is a manuscript promising a confirmed result on a metric the
    # data refuted — there the claim and the evidence point opposite ways.
    promised = _promised_metrics(agenda_required)
    if promised:
        broken = sorted(
            {
                metric
                for c in claims
                if (c.get("verdict") or "").lower() == "refuted"
                for metric in promised
                if _claim_mentions(c, metric)
            }
        )
        if broken:
            blockers.append({
                "requirement": "no_refuted_promised_metrics",
                "reason": f"refuted_metrics={broken}",
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
    stance = _claim_stance(counts["confirmed"], counts["refuted"])
    metrics_summary: dict[str, Any] = {
        "claim_counts": counts,
        "claim_stance": stance,
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
        # What the manuscript is allowed to claim, so a negative result is
        # written up as a negative result rather than as a failed positive one.
        "claim_stance": stance,
        "rule_set": RULE_SET,
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

    metrics_summary = _decode("metrics_summary_json", {})
    return {
        "id": row.get("id"),
        "selection_id": row.get("selection_id"),
        "experiment_run_id": row.get("experiment_run_id"),
        "status": row.get("status"),
        "blockers": _decode("blockers_json", []),
        "metrics_summary": metrics_summary,
        "claim_stance": (metrics_summary or {}).get("claim_stance"),
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
