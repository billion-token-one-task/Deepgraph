"""Benchmark harness management for unsupported experiment protocols.

This module owns the transition from "review blocked because the generated
runner is not expressive enough" to an explicit benchmark-engineering queue.
It does not pretend that an arbitrary benchmark can already run; it records the
required dataset, baseline, harness, and review work as first-class state so the
main experiment scheduler can keep moving.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from agents.benchmark_protocol import resolve_benchmark_protocol
from agents.loop_router import compact_loop_note, route_blockers
from agents.workspace_layout import write_plan_files
from contracts import DeepInsightSpec
from db import database as db


HARNESS_REQUIRED_STATUS = "harness_required"
HARNESS_REQUIRED_STAGE = "benchmark_harness_required"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _load_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def _named_rows(value: Any, *keys: str) -> list[str]:
    rows = value if isinstance(value, list) else [value] if value not in (None, "") else []
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if isinstance(row, Mapping):
            name = ""
            for key in keys:
                name = _text(row.get(key))
                if name:
                    break
        else:
            name = _text(row)
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def _dataset_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("benchmark_targets", "datasets"):
        value = plan.get(key)
        if not isinstance(value, list):
            continue
        for row in value:
            candidate = dict(row) if isinstance(row, Mapping) else {"name": row}
            name = _text(candidate.get("name") or candidate.get("hf_dataset") or candidate.get("dataset"))
            if name:
                rows.append(candidate)
    return rows


def _judgement_dict(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(payload or {})
    judgement = payload.get("judgement") if isinstance(payload.get("judgement"), Mapping) else payload
    return dict(judgement or {})


def judgement_requires_benchmark_harness(
    judgement_payload: Mapping[str, Any] | None,
    *,
    plan: Mapping[str, Any] | None = None,
) -> bool:
    """Return True when review found an unsupported benchmark runner contract."""

    judgement = _judgement_dict(judgement_payload)
    plan = dict(plan or {})
    env = judgement.get("environment_review") if isinstance(judgement.get("environment_review"), Mapping) else {}
    if env.get("benchmark_harness_required"):
        return True
    if plan.get("generated_runner_supported") is False:
        return True
    text = " ".join(
        [
            _text(judgement.get("summary")),
            " ".join(_named_rows(judgement.get("blockers"))),
            " ".join(_named_rows(judgement.get("warnings"))),
        ]
    ).lower()
    harness_tokens = (
        "dedicated benchmark harness",
        "dedicated benchmark harness/recipe",
        "custom harness",
        "generated real-benchmark runner does not support",
        "generated runner cannot execute",
        "task_type=",
        "benchmark harness required",
    )
    return any(token in text for token in harness_tokens)


def _required_capabilities(plan: Mapping[str, Any], judgement: Mapping[str, Any]) -> list[str]:
    text = json.dumps({"plan": plan, "judgement": judgement}, ensure_ascii=False, default=str).lower()
    capabilities: list[str] = [
        "load official/materialized benchmark split",
        "record split metadata and example counts",
        "run candidate and baseline methods under one evaluator",
        "write benchmark_summary/raw_predictions/per_seed/per_dataset artifacts",
    ]
    if any(token in text for token in ("attention", "q/k/v", "qkv", "dense reference", "certificate", "ccar")):
        capabilities.extend(
            [
                "capture model q/k/v or attention tensors at evaluation time",
                "compute dense-reference attention for audit comparisons",
                "log retained and omitted key/page ids",
                "record certificate slack, repair actions, and dense fallback events",
            ]
        )
    if any(token in text for token in ("longbench", "long context", "long-context")):
        capabilities.extend(
            [
                "support long-context multiple-choice QA prompts",
                "preserve official answer normalization and option scoring",
            ]
        )
    if any(token in text for token in ("baseline", "repo", "checkpoint")):
        capabilities.append("fetch or adapt official baselines/checkpoints with pinned revisions")
    out: list[str] = []
    seen: set[str] = set()
    for item in capabilities:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _agent_workflow(capabilities: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "agent": "Benchmark Manager",
            "owns": "official benchmark protocol, split policy, metric contract, and readiness state",
            "exit_criteria": [
                "benchmark protocol source is recorded",
                "official/materialized split decision is locked",
                "required artifacts are enumerated",
            ],
        },
        {
            "agent": "Dataset Fetch Agent",
            "owns": "dataset download/materialization, split inspection, schema detection, and count manifest",
            "exit_criteria": [
                "dataset refs and revisions are pinned",
                "split names and materialized example counts are written",
                "sample rows are schema-checked",
            ],
        },
        {
            "agent": "Baseline Fetch Agent",
            "owns": "baseline repository/checkpoint discovery, dependency install notes, and evaluator adapters",
            "exit_criteria": [
                "each required baseline is runnable or explicitly unavailable with evidence",
                "baseline command/config is pinned",
                "candidate and baseline budgets are comparable",
            ],
        },
        {
            "agent": "Benchmark Harness Code Agent",
            "owns": "custom runner implementation for capabilities unsupported by the generated QA/code runner",
            "required_capabilities": capabilities,
            "exit_criteria": [
                "runner emits all required benchmark artifacts",
                "runner fails closed when audit fields are missing",
                "dependency and hardware requirements are recorded",
            ],
        },
        {
            "agent": "Harness Review Agent",
            "owns": "pre-GPU audit that the harness matches the benchmark protocol and paper claims",
            "exit_criteria": [
                "official metric/split agreement passes",
                "no synthetic or smoke-only evidence can enter manuscript generation",
                "GPU execution can create a formal experiment_run",
            ],
        },
    ]


def build_harness_task(
    insight: Mapping[str, Any],
    *,
    judgement_payload: Mapping[str, Any] | None = None,
    source: str = "experiment_review",
) -> dict[str, Any]:
    spec = DeepInsightSpec.from_raw(insight)
    plan = dict(spec.experimental_plan or {})
    judgement = _judgement_dict(judgement_payload)
    publication_contract = plan.get("publication_evidence_contract") if isinstance(plan.get("publication_evidence_contract"), Mapping) else {}
    benchmark_protocol = (
        publication_contract.get("benchmark_protocol")
        if isinstance(publication_contract.get("benchmark_protocol"), Mapping)
        else plan.get("benchmark_protocol") if isinstance(plan.get("benchmark_protocol"), Mapping) else None
    )
    if not isinstance(benchmark_protocol, Mapping):
        benchmark_protocol = resolve_benchmark_protocol(
            plan,
            method=spec.proposed_method,
            claim=spec.title or spec.problem_statement,
        )
    datasets = _dataset_rows(plan)
    recipe_blockers = plan.get("benchmark_recipe_blockers") if isinstance(plan.get("benchmark_recipe_blockers"), list) else []
    capabilities = _required_capabilities(plan, judgement)
    benchmark_names = _named_rows(datasets, "name", "hf_dataset", "dataset")
    if not benchmark_names:
        benchmark_names = _named_rows(
            (benchmark_protocol or {}).get("full_benchmark_requirements", {}).get("required_dataset_names")
            if isinstance((benchmark_protocol or {}).get("full_benchmark_requirements"), Mapping)
            else [],
            "name",
        )
    routing_inputs = _named_rows(judgement.get("blockers")) + _named_rows(judgement.get("warnings"))
    routing_inputs.extend(_named_rows(recipe_blockers))
    summary = _text(judgement.get("summary"))
    if summary:
        routing_inputs.append(summary)
    if not routing_inputs:
        routing_inputs.append("Generated runner cannot execute the benchmark contract.")
    loop_route = route_blockers(
        routing_inputs,
        context={"source": source, "stage": HARNESS_REQUIRED_STAGE, "insight_id": spec.insight_id},
    )
    return {
        "schema_version": "benchmark_harness_task_v1",
        "status": HARNESS_REQUIRED_STATUS,
        "source": source,
        "deep_insight_id": spec.insight_id,
        "insight_title": spec.title,
        "harness_kind": "custom_benchmark_harness",
        "benchmark_names": benchmark_names,
        "dataset_refs": datasets,
        "model_refs": _named_rows(plan.get("model_targets") or plan.get("models"), "name", "hf_model", "model"),
        "baseline_refs": _named_rows(plan.get("baselines"), "name", "method", "model"),
        "recipe_blockers": recipe_blockers,
        "required_capabilities": capabilities,
        "benchmark_protocol": benchmark_protocol,
        "review_judgement": judgement,
        "loop_router": loop_route,
        "agent_workflow": _agent_workflow(capabilities),
        "next_actions": [
            "lock the official benchmark protocol or materialize the dataset when no official protocol exists",
            "download/cache datasets and baselines with pinned revisions",
            "generate a custom runner for unsupported task/audit capabilities",
            "run harness review before creating a GPU experiment_run",
            "return to formal experiment forge only after the harness emits required artifacts",
        ],
        "runnable_by_generated_runner": False,
    }


def _primary_benchmark_name(task: Mapping[str, Any]) -> str:
    names = task.get("benchmark_names")
    if isinstance(names, list) and names:
        return _text(names[0])
    refs = task.get("dataset_refs")
    if isinstance(refs, list) and refs:
        row = refs[0]
        if isinstance(row, Mapping):
            return _text(row.get("name") or row.get("hf_dataset") or row.get("dataset"))
    return "custom benchmark"


def upsert_harness_job(insight_id: int, task: Mapping[str, Any]) -> dict[str, Any]:
    scope = db.fetchone(
        """
        SELECT di.agenda_id, arj.resource_grant_id
        FROM deep_insights di
        JOIN auto_research_jobs arj
          ON arj.deep_insight_id=di.id AND arj.agenda_id=di.agenda_id
        JOIN resource_grants rg
          ON rg.id=arj.resource_grant_id
         AND rg.agenda_id=di.agenda_id
         AND rg.idea_id=di.id
         AND rg.status='active'
         AND rg.expires_at > CURRENT_TIMESTAMP
        WHERE di.id=?
        """,
        (int(insight_id),),
    )
    if not scope or int(scope.get("agenda_id") or 0) <= 0:
        raise PermissionError(
            "benchmark harness job requires an agenda-scoped active ResourceGrant"
        )
    agenda_id = int(scope["agenda_id"])
    resource_grant_id = int(scope["resource_grant_id"])
    payload = json.dumps(task, ensure_ascii=False, default=str)
    dataset_refs = json.dumps(task.get("dataset_refs") or [], ensure_ascii=False, default=str)
    baseline_refs = json.dumps(task.get("baseline_refs") or [], ensure_ascii=False, default=str)
    capabilities = json.dumps(task.get("required_capabilities") or [], ensure_ascii=False, default=str)
    benchmark_name = _primary_benchmark_name(task)
    loop_note = compact_loop_note(task.get("loop_router") if isinstance(task.get("loop_router"), Mapping) else None)
    last_note = "Queued for Benchmark Manager, Dataset/Baseline Fetch, Harness Code, and Harness Review agents."
    if loop_note:
        last_note = f"{last_note} {loop_note}"
    existing = db.fetchone(
        """
        SELECT id FROM benchmark_harness_jobs
        WHERE agenda_id=? AND deep_insight_id=?
        """,
        (agenda_id, int(insight_id)),
    )
    if existing:
        db.execute(
            """
            UPDATE benchmark_harness_jobs
            SET status=?,
                harness_kind=?,
                benchmark_name=?,
                dataset_refs=?,
                baseline_refs=?,
                required_capabilities=?,
                task_plan=?,
                last_error=?,
                last_note=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE agenda_id=? AND deep_insight_id=?
            """,
            (
                HARNESS_REQUIRED_STATUS,
                _text(task.get("harness_kind") or "custom_benchmark_harness"),
                benchmark_name,
                dataset_refs,
                baseline_refs,
                capabilities,
                payload,
                "Generated runner cannot execute the benchmark contract.",
                last_note,
                agenda_id,
                int(insight_id),
            ),
        )
        job_id = int(existing["id"])
    else:
        job_id = db.insert_returning_id(
            """
            INSERT INTO benchmark_harness_jobs
                (agenda_id, resource_grant_id, deep_insight_id, status,
                 harness_kind, benchmark_name, dataset_refs,
                 baseline_refs, required_capabilities, task_plan, last_error, last_note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                agenda_id,
                resource_grant_id,
                int(insight_id),
                HARNESS_REQUIRED_STATUS,
                _text(task.get("harness_kind") or "custom_benchmark_harness"),
                benchmark_name,
                dataset_refs,
                baseline_refs,
                capabilities,
                payload,
                "Generated runner cannot execute the benchmark contract.",
                last_note,
            ),
        )
    db.commit()
    return {"harness_job_id": job_id, "benchmark_name": benchmark_name}


def record_harness_required(
    insight_id: int,
    *,
    judgement_payload: Mapping[str, Any] | None = None,
    source: str = "experiment_review",
) -> dict[str, Any]:
    insight = db.fetchone("SELECT * FROM deep_insights WHERE id=?", (int(insight_id),))
    if not insight:
        return {"error": f"Deep insight {insight_id} not found"}
    task = build_harness_task(insight, judgement_payload=judgement_payload, source=source)
    job = upsert_harness_job(int(insight_id), task)
    paths = write_plan_files(
        int(insight_id),
        files={
            "benchmark_harness_task.json": task,
            "benchmark_harness_status.json": {
                "status": HARNESS_REQUIRED_STATUS,
                "stage": HARNESS_REQUIRED_STAGE,
                "loop_router": task.get("loop_router") if isinstance(task.get("loop_router"), Mapping) else {},
                **job,
            },
        },
        insight=dict(insight),
        mirror_to_run_spec=False,
    )
    return {**job, "task": task, "paths": paths}
