"""The evidence behind one verdict, read live from the ledger.

This is the data layer the homepage drill-down runs on: open a conclusion in
the Evidence tab and every rung it renders, every hash it shows and every file
it offers for download is served from here.

Three things it deliberately does NOT do, because the whole value of the
drill-down is that a reader can check it:

  * It never recomputes a verdict. Verdicts come from
    scientific_decision_records, the only table a verdict is written to, and
    each carries the verdict_hash the auditor signed it with.
  * It never narrows or widens a query to make a finding look better. A run's
    measurement at every stage is returned, so the held-out number sits beside
    the benchmark number instead of behind it.
  * It has no mutating endpoint and no operator affordance. Mutations belong
    to the operator-authenticated /api/meta-harness/v1 blueprint.

Two things are removed on the way out, and neither is evidence for anything:
the compute vendor baked into some actor names, and the hardware class a grant
requested. The ledger keeps every real value; this is a display decision, made
in one place so it cannot be quietly undone. What is NOT removed is
`retrospective_review:operator`, which says a human entered a rung by hand --
the most important caveat a reader can have about a row.

Bilingual copy lives here as (zh, en) pairs. The dashboard's i18n.js owns the
strings its own template renders; these belong to the payload, travel with it,
and cannot drift out of sync with the field they describe.

It began as a standalone exhibit page at /judge, which was retired on
2026-08-28 once the homepage could do the same thing in the place a reader
already was.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

from flask import Blueprint, abort, jsonify, send_file

from config import IDEA_WORKSPACE_DIR
from db import database as db
from web.provenance_routes import _scrub_text

blueprint = Blueprint("judge_demo", __name__)

# The candidate ideas in agenda 14 were transcribed from published papers by an
# operator, not invented by the system: model_version records which. Any
# exhibit whose candidate carries this value is a reproduction under audit, and
# the page says so next to the verdict rather than in a footnote.
OPERATOR_FROZEN_MODEL_VERSION = "operator_frozen_no_llm"

# Ledger actor names, as shown to a reader.
#
# Two different things are being done here, and conflating them would be a
# mistake in opposite directions:
#
#   * `colab_terminal_handoff_v1` names a compute vendor in a string that is
#     otherwise about WHEN a rung was written. The vendor is not evidence for
#     any claim on this page, and this page is public, so the display layer
#     shows the stage instead. The ledger keeps the real value; nothing is
#     rewritten in the database.
#   * `retrospective_review:operator` says a HUMAN entered that rung rather
#     than the system reaching it on its own. That is not infrastructure, it is
#     the single most important caveat a reader can have about a row, and it
#     stays visible and says "operator" in both languages.
#
# Unknown actors fall through unchanged rather than being silently blanked: an
# actor this map has not seen is something a reader should still see.
ACTOR_LABELS = {
    "forge": ("预注册", "Pre-registration"),
    "colab_terminal_handoff_v1": ("算力交接", "Compute handoff"),
    "settled_compute_handoff_v1": ("算力交接 (已结算)", "Compute handoff (settled)"),
    "evidence_audit_v1": ("证据审计", "Evidence audit"),
    "ai-reviewer-v1": ("AI 评审", "AI reviewer"),
    "retrospective_review:operator": ("人工回溯录入", "Operator, entered retrospectively"),
}

# Any actor whose name carries one of these is a vendor or hardware identity
# and must not reach a public page even if ACTOR_LABELS has not been taught
# about it yet. Checked as a substring, case-insensitively.
INFRASTRUCTURE_TOKENS = (
    "colab", "aws", "gcp", "azure", "nvidia", "a10g", "a100", "t4", "v100",
    "h100", "g5.", "ec2", "runpod", "lambda-labs", "vast.ai",
)


def actor_labels(actor: str) -> tuple[str, str]:
    """(zh, en) for a ledger actor, with infrastructure identity removed."""
    known = ACTOR_LABELS.get(actor)
    if known:
        return known
    lowered = str(actor or "").lower()
    if any(token in lowered for token in INFRASTRUCTURE_TOKENS):
        return ("算力交接", "Compute handoff")
    return (actor, actor)


# The rungs an experiment run must climb, in order. Sourced from
# contracts.meta_harness.EVIDENCE_STATES; spelled out here with the plain
# language a reader outside the project needs, and with the gate that has to
# pass before each one is written.
#
# (state, label_zh, label_en, why_zh, why_en)
LADDER = [
    ("planned", "预注册", "Pre-registration",
     "候选的方法、指标、基准切片和成功阈值在跑之前写死并存档; 之后改不了",
     "The method, metric, benchmark slice and success thresholds are frozen "
     "and filed before anything runs, and cannot be edited afterwards"),
    ("sanity_passed", "小样本试跑", "Pilot",
     "先用小切片证明这套代码能跑出非空预测, 再申请全量预算",
     "A small slice first, to prove the code produces non-empty predictions "
     "before it may ask for a full benchmark budget"),
    ("full_benchmark_complete", "全量基准", "Full benchmark",
     "候选和基线在同一份 revision-pinned 数据、同一份预算下各跑一遍",
     "Candidate and baseline each run once, on the same revision-pinned data "
     "and under the same budget"),
    ("evidence_audited", "证据审计", "Evidence audit",
     "留出集必须自证是留出集; 置换检验出 p 值; 跨厂商的独立评审读原始预测",
     "The holdout must prove it is a holdout; a permutation test yields the "
     "p-value; an independent evaluator from a different vendor reads the raw "
     "predictions"),
    ("scientifically_decided", "判决", "Verdict",
     "supported / refuted / inconclusive 之一, 连同 verdict_hash 写进不可变账本",
     "One of supported / refuted / inconclusive, written to the immutable "
     "ledger together with its verdict_hash"),
    ("manuscript_allowed", "允许成稿", "Manuscript allowed",
     "只有走完上面全部台阶的判决才允许写成论文",
     "Only a verdict that climbed every rung above may be written up"),
]

# How a verdict reads in a sentence, in both languages.
#
# Two rules, both learned the hard way on this page.
#
# Not "accuracy improved": the metric is whatever the run's benchmark contract
# pinned, and hardcoding one metric's name makes the sentence wrong the first
# time a run measures latency or cost instead.
#
# And no jargon. An earlier draft said "预注册的预测成立" / "the pre-registered
# prediction held". Nobody arriving on this page knows what pre-registration
# is, and a reader who has to look up a word in the headline has already
# stopped reading. These say the same thing in words that need no gloss: the
# candidate said in advance that it would work, and then it was measured. What
# it was measured AGAINST is the control value and the thresholds, and those
# are on the row itself rather than hidden behind the sentence.
VERDICT_PHRASE = {
    "supported": ("符合预期效果", "Matched the effect it predicted"),
    "refuted": ("达不到预期效果", "Did not reach the effect it predicted"),
    "inconclusive": ("不确定是否有效", "Not clear whether it works"),
    "invalid": ("这一趟没测出可用结果", "This run produced no usable measurement"),
}


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    try:
        return [dict(row) for row in db.fetchall(sql, params)]
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        return []


def _one(sql: str, params: tuple = ()) -> dict:
    rows = _rows(sql, params)
    return rows[0] if rows else {}


def _loads(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}


def _ts(value: Any) -> str:
    """One clock, one format.

    experiment_runs.created_at comes back naive and the transition rows come
    back tz-aware, so an unformatted ladder mixed "06:09:48" with
    "07:06:22+00:00" and invited the reader to wonder which was local. Both are
    UTC; the page says so once, in the ladder header.
    """
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ", timespec="seconds")[:19]
    return str(value)[:19]


def _exhibit(run_id: int) -> dict | None:
    """Assemble one run's whole ladder from the tables that recorded it.

    Returns None when the run does not exist, so a page built before a demo
    run finishes renders the exhibits it does have instead of a 500.
    """
    run = _one(
        "SELECT id, deep_insight_id, agenda_id, status, phase,"
        " scientific_evidence_state, hypothesis_verdict, baseline_metric_name,"
        " baseline_metric_value, best_metric_value, effect_size, effect_pct,"
        " resource_grant_id, created_at, completed_at"
        " FROM experiment_runs WHERE id=?",
        (run_id,),
    )
    if not run:
        return None

    idea = _one(
        "SELECT id, title, model_version, agenda_id FROM deep_insights WHERE id=?",
        (run["deep_insight_id"],),
    )
    audit = _one(
        "SELECT id, raw_artifacts_hash, claim_ledger_hash, benchmark_contract_hash,"
        " evaluator_ref, evaluator_hash, holdout_ref, holdout_hash, created_at"
        " FROM evidence_audit_records WHERE experiment_run_id=?"
        " ORDER BY id DESC LIMIT 1",
        (run_id,),
    )
    decision = _one(
        "SELECT id, verdict, verdict_hash, evidence_decision_json, created_at"
        " FROM scientific_decision_records WHERE experiment_run_id=?"
        " ORDER BY id DESC LIMIT 1",
        (run_id,),
    )
    outcome = _one(
        "SELECT id, verdict, effect, baseline, actual_tokens, actual_gpu_hours,"
        " wall_seconds, state_decision, recorded_at"
        " FROM outcome_records WHERE experiment_run_id=?"
        " ORDER BY id DESC LIMIT 1",
        (run_id,),
    )

    # The whole grant chain, not just the last one. Waiting for budget is a rung
    # of the ladder the page claims to show, and a run that reached a verdict
    # did so across a separate grant per stage -- pilot, full benchmark, audit --
    # each with its own cap. Showing only run.resource_grant_id showed the audit
    # grant alone and made the wait look like it never happened.
    #
    # Reached through the run's own compute requests rather than by timestamp:
    # run 274's pilot grant was issued 1.3 seconds BEFORE the run row existed,
    # so a "grants created at or after the run" filter silently dropped the
    # first rung of the very chain this row is here to show.
    # gpu_class is deliberately not selected. It is the hardware class a grant
    # requested, it is not evidence for anything, and an earlier draft rendered
    # it straight onto the public page.
    grants = _rows(
        "SELECT g.id, g.stage, g.token_cap, g.max_gpu_hours,"
        " g.status, g.grant_reason, g.created_at FROM resource_grants g"
        " WHERE g.id IN ("
        "   SELECT DISTINCT resource_grant_id FROM colab_work_requests_v1"
        "   WHERE experiment_run_id=? AND resource_grant_id IS NOT NULL"
        " ) ORDER BY g.id",
        (run_id,),
    )

    # Which compute account or GPU class ran the work is deliberately NOT
    # collected here. It is not evidence for any claim on this page -- the
    # holdout, the evaluator and the five hashes are -- and this page is
    # public. Naming the accounts would put infrastructure detail on a company
    # site in exchange for nothing a reader needs.

    reached = {
        row["to_state"]: row
        for row in _rows(
            "SELECT to_state, actor, created_at FROM evidence_state_transitions"
            " WHERE experiment_run_id=? ORDER BY id",
            (run_id,),
        )
    }
    # "planned" is the state a run is created in, so no transition writes it;
    # the run's own created_at is when the pre-registration was frozen.
    ladder = []
    for state, label_zh, label_en, why_zh, why_en in LADDER:
        hit = reached.get(state)
        if state == "planned":
            hit = hit or {"actor": "forge", "created_at": run.get("created_at")}
        raw_actor = (hit or {}).get("actor", "")
        actor_zh, actor_en = actor_labels(raw_actor) if raw_actor else ("", "")
        ladder.append({
            "state": state,
            "label_zh": label_zh,
            "label_en": label_en,
            "why_zh": why_zh,
            "why_en": why_en,
            "reached": bool(hit),
            # The abstracted label only. The raw actor is deliberately not
            # carried into the response: a field that exists is a field that
            # gets rendered by the next person who needs "just a bit more
            # detail", and the point of the mapping is that it cannot.
            "actor_zh": actor_zh,
            "actor_en": actor_en,
            "operator_entered": raw_actor.startswith("retrospective_review:"),
            "at": _ts((hit or {}).get("created_at")),
        })

    decision_input = _loads(decision.get("evidence_decision_json")).get("input", {})
    decision_body = _loads(decision.get("evidence_decision_json")).get("decision", {})

    return {
        "run": run,
        "idea": idea,
        "title": _scrub_text(str(idea.get("title") or "")),
        "operator_frozen": idea.get("model_version") == OPERATOR_FROZEN_MODEL_VERSION,
        "model_version": idea.get("model_version"),
        "grants": grants,
        "audit": audit,
        "decision": decision,
        "verdict": (decision.get("verdict") or outcome.get("verdict")
                    or run.get("hypothesis_verdict") or "pending"),
        "p_value": decision_input.get("p_value"),
        "alpha": decision_input.get("alpha"),
        "metric_value": decision_input.get("metric_value"),
        "baseline_value": decision_input.get("baseline_value"),
        "blockers": decision_body.get("blockers") or [],
        "significant": decision_body.get("significant"),
        "outcome": outcome,
        "ladder": ladder,
        "complete": bool(decision),
    }


def _headline(idea_row: dict) -> str:
    """The sentence a reader came for, not the name the system files it under.

    Same precedence web/app.py::_conclusion_headline uses for the hero
    conclusion -- proposed_method.one_line, then evidence_summary, then the
    internal identifier as a last resort -- so the drill-down and the headline
    above it never disagree about what a finding is called.
    """
    method = idea_row.get("proposed_method")
    if isinstance(method, str):
        try:
            method = json.loads(method)
        except (TypeError, ValueError):
            method = None
    if isinstance(method, dict):
        one_line = str(method.get("one_line") or "").strip()
        if one_line:
            return _scrub_text(one_line)
    summary = str(idea_row.get("evidence_summary") or "").strip()
    if summary:
        return _scrub_text(summary)
    return _scrub_text(str(idea_row.get("title") or ""))


@blueprint.get("/api/v1/judge/ladder/<int:run_id>")
def judge_ladder(run_id: int):
    """One run's evidence ladder, as JSON, for the drill-down on the homepage.

    Everything here is read live from the ledger. Two things are removed on the
    way out and neither is evidence for any claim: the compute vendor baked
    into some actor names, and the hardware class a grant requested. What the
    verdict actually rests on -- the holdout, the permutation test, the
    independent evaluator, and the five hashes -- is returned in full, because
    a drill-down a reader cannot check is decoration.
    """
    exhibit = _exhibit(run_id)
    if exhibit is None:
        return jsonify({"error": "no such run", "run_id": run_id}), 404

    idea = _one(
        "SELECT id, title, proposed_method, evidence_summary, model_version,"
        " predictions, falsification, problem_statement"
        " FROM deep_insights WHERE id=?",
        (exhibit["run"]["deep_insight_id"],),
    )
    full_run = _one(
        "SELECT program_md, success_criteria, baseline_metric_name"
        " FROM experiment_runs WHERE id=?",
        (run_id,),
    )
    verdict = str(exhibit["verdict"] or "")
    phrase_zh, phrase_en = VERDICT_PHRASE.get(verdict, (verdict, verdict))
    run = exhibit["run"]
    outcome = exhibit["outcome"] or {}
    audit = exhibit["audit"] or {}
    decision = exhibit["decision"] or {}

    return jsonify({
        "run_id": run["id"],
        "idea_id": run["deep_insight_id"],
        "agenda_id": run["agenda_id"],
        "verdict": verdict,
        "verdict_phrase": {"zh": phrase_zh, "en": phrase_en},
        "headline": _headline(idea),
        # Said plainly, next to the verdict: a candidate a human transcribed
        # from a paper is a reproduction under audit, not a discovery.
        "operator_frozen": exhibit["operator_frozen"],
        "model_version": exhibit["model_version"],
        "ladder": exhibit["ladder"],
        # What "the effect held" was measured against, written down before the
        # run existed. Without this the verdict is an assertion.
        "expectation": _declared_expectation(full_run or {}, idea),
        "problem": _scrub_text(str(idea.get("problem_statement") or "")),
        # The run's own research program, as the forge wrote it. Markdown, a
        # couple of thousand characters, and the answer to "what did it
        # actually do".
        "program_md": _scrub_text(str((full_run or {}).get("program_md") or "")),
        # Every stage's scored result, so the held-out number sits beside the
        # benchmark number instead of behind it.
        "measurements": _measurements(run_id),
        # The files behind the hashes, so a hash is checkable rather than
        # decorative.
        "artifacts": _artifacts(run_id),
        "statistics": {
            "metric_name": run.get("baseline_metric_name"),
            "metric_value": exhibit["metric_value"] if exhibit["metric_value"] is not None
                            else run.get("best_metric_value"),
            "baseline_value": exhibit["baseline_value"] if exhibit["baseline_value"] is not None
                              else run.get("baseline_metric_value"),
            "effect_pct": run.get("effect_pct"),
            "p_value": exhibit["p_value"],
            "alpha": exhibit["alpha"],
            "significant": exhibit["significant"],
            "blockers": exhibit["blockers"],
        },
        "audit": {
            "holdout_ref": audit.get("holdout_ref"),
            "holdout_hash": audit.get("holdout_hash"),
            "evaluator_ref": audit.get("evaluator_ref"),
            "evaluator_hash": audit.get("evaluator_hash"),
            "raw_artifacts_hash": audit.get("raw_artifacts_hash"),
            "claim_ledger_hash": audit.get("claim_ledger_hash"),
            "benchmark_contract_hash": audit.get("benchmark_contract_hash"),
        },
        "decision": {
            "id": decision.get("id"),
            "verdict_hash": decision.get("verdict_hash"),
            "created_at": _ts(decision.get("created_at")),
        },
        "grants": [
            {
                "id": g.get("id"),
                "stage": g.get("stage"),
                "token_cap": g.get("token_cap"),
                "max_gpu_hours": g.get("max_gpu_hours"),
                "status": g.get("status"),
            }
            for g in exhibit["grants"]
        ],
        "cost": {
            "tokens": outcome.get("actual_tokens"),
            "gpu_hours": outcome.get("actual_gpu_hours"),
            "wall_seconds": outcome.get("wall_seconds"),
        },
    })


# Artifact kinds a reader may download. Everything here is a measurement or a
# manifest describing one: the predictions the model actually produced, the
# dataset revision and slice they were produced on, the environment, the model,
# and the scored result. `source_data` is deliberately absent -- those rows
# point at internal plan JSON full of absolute paths, and they are not
# evidence.
DOWNLOADABLE_ARTIFACTS = (
    "final_results",
    "raw_predictions",
    "dataset_manifest",
    "environment_manifest",
    "model_manifest",
)

# Stages in the order they happen, so the measurement table reads down the page
# the way the run went.
STAGE_ORDER = ("pilot", "full_benchmark", "evidence_audit")

STAGE_LABELS = {
    "pilot": ("小样本试跑", "Pilot"),
    "full_benchmark": ("全量基准", "Full benchmark"),
    "evidence_audit": ("留出集复核", "Held-out re-check"),
}


def _json_or_none(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _declared_expectation(run: dict, idea: dict) -> dict:
    """What the run said, before it ran, would count as the effect working.

    This is the thing a reader is actually asking for when a verdict says the
    effect held: held against what? The answer was written down before the
    measurement existed -- the claim, the metric and its direction, the three
    thresholds, the confidence, the paper the predicted effect was taken from,
    and the conditions that would count as falsifying it.
    """
    criteria = _json_or_none(run.get("success_criteria")) or {}
    contract = criteria.get("publication_evidence_contract") or {}
    predictions = _json_or_none(idea.get("predictions")) or []
    first = predictions[0] if isinstance(predictions, list) and predictions else {}
    falsification = _json_or_none(idea.get("falsification")) or {}

    thresholds = {
        key: criteria.get(key)
        for key in ("exciting", "solid", "disappointing")
        if criteria.get(key) is not None
    }
    return {
        "claim": _scrub_text(str(
            contract.get("claim_to_validate")
            or first.get("outcome") or "").strip()),
        "metric_name": criteria.get("metric_name") or run.get("baseline_metric_name"),
        "metric_direction": criteria.get("metric_direction"),
        "thresholds": thresholds,
        "minimum_seeds": contract.get("minimum_seeds"),
        "confidence": first.get("confidence"),
        # The paper the predicted effect was taken from. Present on every
        # literature-grounded candidate and absent on every candidate the
        # proposer invented unaided -- which is the sharpest single predictor
        # in the ledger of whether a candidate ever reaches supported.
        "effect_reference": _scrub_text(str(first.get("effect_reference") or "")),
        "falsification": {
            key: _scrub_text(str(value))
            for key, value in falsification.items()
            if isinstance(value, str)
        } if isinstance(falsification, dict) else {},
    }


def _measurements(run_id: int) -> list[dict]:
    """The scored result at each stage, so the held-out number is not buried.

    The page used to show the full-benchmark figure alone. For run 274 that is
    0.625, while the held-out re-check the verdict actually rests on is 0.57 --
    both well clear of the 0.32 control, but showing only the larger of the two
    is the kind of selective reporting this system exists to catch.
    """
    rows = _rows(
        "SELECT artifact_stage, metric_key, metric_value, content_sha256"
        " FROM experiment_artifacts"
        " WHERE run_id=? AND artifact_type='final_results'"
        "   AND metric_value IS NOT NULL"
        " ORDER BY id",
        (run_id,),
    )
    by_stage: dict[str, dict] = {}
    for row in rows:
        stage = str(row.get("artifact_stage") or "")
        if stage:
            by_stage[stage] = row
    out = []
    for stage in STAGE_ORDER:
        row = by_stage.get(stage)
        if not row:
            continue
        zh, en = STAGE_LABELS.get(stage, (stage, stage))
        out.append({
            "stage": stage,
            "label_zh": zh,
            "label_en": en,
            "metric_key": row.get("metric_key"),
            "metric_value": row.get("metric_value"),
            "content_sha256": row.get("content_sha256"),
        })
    return out


def _artifacts(run_id: int) -> list[dict]:
    """Every downloadable measurement file, with the hash it is claimed to be.

    A hash on a page nobody can resolve to bytes is decoration. These rows are
    what turns "here is a sha256" into "here is the file, check it yourself".
    """
    placeholders = ", ".join("?" for _ in DOWNLOADABLE_ARTIFACTS)
    rows = _rows(
        "SELECT id, artifact_type, artifact_stage, content_sha256, path"
        f" FROM experiment_artifacts WHERE run_id=? AND artifact_type IN ({placeholders})"
        " ORDER BY id",
        (run_id, *DOWNLOADABLE_ARTIFACTS),
    )
    out = []
    for row in rows:
        if not row.get("content_sha256"):
            continue
        stage = str(row.get("artifact_stage") or "")
        zh, en = STAGE_LABELS.get(stage, (stage, stage))
        out.append({
            "id": row["id"],
            "kind": row.get("artifact_type"),
            "stage": stage,
            "stage_zh": zh,
            "stage_en": en,
            "sha256": row.get("content_sha256"),
            # The path itself never leaves the process; the id is the handle.
            "url": f"/api/v1/judge/artifact/{row['id']}",
        })
    return out


@blueprint.get("/api/v1/judge/artifact/<int:artifact_id>")
def judge_artifact(artifact_id: int):
    """Hand over one measurement file so a reader can check a hash themselves.

    The path comes from the ledger, never from the request, and is still
    confined to the idea workspace before it is opened: a row is a row, and a
    route that opens whatever a table says is one bad INSERT from serving
    anything on the disk.
    """
    row = _one(
        "SELECT id, run_id, artifact_type, path, content_sha256"
        " FROM experiment_artifacts WHERE id=?",
        (artifact_id,),
    )
    if not row or row.get("artifact_type") not in DOWNLOADABLE_ARTIFACTS:
        abort(404)
    raw_path = str(row.get("path") or "")
    if not raw_path:
        abort(404)
    try:
        resolved = pathlib.Path(raw_path).resolve()
        root = pathlib.Path(IDEA_WORKSPACE_DIR).resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        abort(404)
    if not resolved.is_file():
        abort(404)
    # Named for what it is, not where it lives: the filesystem layout is
    # nobody's business and the sha256 is how a reader identifies the file.
    suffix = ".jsonl" if row["artifact_type"] == "raw_predictions" else ".json"
    download_name = (
        f"run{row['run_id']}-{row['artifact_type']}-"
        f"{str(row['content_sha256'] or '')[:12]}{suffix}"
    )
    return send_file(
        resolved,
        mimetype="application/json",
        as_attachment=True,
        download_name=download_name,
        max_age=0,
    )
