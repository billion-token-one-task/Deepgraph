"""Structured loop state for benchmark harness jobs.

The auto-research scheduler owns the queue, but harness jobs need a more
specific loop than one generic ``harness_required`` label. This module keeps the
state machine deliberately conservative: it may normalize known official dataset
recipes, but it does not claim a benchmark is executable until source locking,
materialization/counts, harness code, and review are all represented.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Mapping

from agents.dataset_resolver import resolve_known_dataset_recipe


HF_DATASET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


_STATUS_META = {
    "source_resolution_required": {
        "owner": "Benchmark Design Agent",
        "stage": "benchmark_literature_design",
        "next_actions": [
            "rerun literature-grounded benchmark selection with official dataset sources",
            "pin the dataset homepage, repository, files, license, and split policy",
            "rewrite benchmark_targets with concrete source refs before materialization",
        ],
    },
    "dataset_materialization_required": {
        "owner": "Dataset Fetch Agent",
        "stage": "dataset_materialization",
        "next_actions": [
            "download or cache the pinned official dataset artifacts",
            "inspect splits/schema and write materialized example counts",
            "mark dataset_cache_verified only after local files or cache entries are present",
        ],
    },
    "benchmark_harness_code_required": {
        "owner": "Benchmark Harness Code Agent",
        "stage": "benchmark_harness_required",
        "next_actions": [
            "generate or adapt the benchmark runner/evaluator for the locked protocol",
            "emit benchmark_summary, raw_predictions, per_seed, and per_dataset artifacts",
            "fail closed if official metrics, splits, or materialized counts are missing",
        ],
    },
    "ready_for_formal_forge": {
        "owner": "Harness Review Agent",
        "stage": "harness_review",
        "next_actions": [
            "review the materialized harness against the benchmark protocol",
            "then requeue the insight for formal experiment forge",
        ],
    },
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _looks_like_hf_dataset_id(value: Any) -> bool:
    return bool(HF_DATASET_RE.match(_text(value)))


def _candidate_names(row: Mapping[str, Any], benchmark_name: str | None = None) -> list[str]:
    names: list[str] = []
    for value in (
        row.get("name"),
        row.get("requested_name"),
        benchmark_name,
        row.get("dataset"),
        row.get("hf_dataset"),
    ):
        text = _text(value)
        if text and text.lower() not in {item.lower() for item in names}:
            names.append(text)
    return names


def _known_recipe_for_row(row: Mapping[str, Any], benchmark_name: str | None = None) -> dict[str, Any] | None:
    for name in _candidate_names(row, benchmark_name):
        recipe = resolve_known_dataset_recipe(name)
        if recipe:
            return recipe
    return None


def normalize_harness_dataset_ref(raw: Any, *, benchmark_name: str | None = None) -> dict[str, Any]:
    """Normalize one harness dataset ref using only the local official registry."""

    source = dict(raw) if isinstance(raw, Mapping) else {"name": raw}
    explicit_requires_harness = source.get("requires_harness") is True
    explicit_runner_unsupported = source.get("generated_runner_supported") is False
    recipe = _known_recipe_for_row(source, benchmark_name)
    if recipe:
        merged = {**source, **recipe}
        if source.get("name") and source.get("name") != recipe.get("name"):
            merged.setdefault("requested_name", source.get("name"))
        for key in ("why", "benchmark_axis", "max_eval_examples", "sample_policy"):
            if source.get(key) not in (None, "", []):
                merged.setdefault(key, source.get(key))
        if explicit_requires_harness:
            merged["requires_harness"] = True
        if explicit_runner_unsupported or merged.get("requires_harness"):
            merged["generated_runner_supported"] = False
        merged["resolver_normalized"] = True
        merged.setdefault("resolver_source", recipe.get("source") or "local_registry")
    else:
        merged = source

    hf_dataset = _text(merged.get("hf_dataset"))
    if hf_dataset and not _looks_like_hf_dataset_id(hf_dataset):
        merged.setdefault("hf_dataset_hint", hf_dataset)
        merged["hf_dataset"] = ""
    if merged.get("requires_harness") is True:
        merged["generated_runner_supported"] = False
    return merged


def _materialized_count(row: Mapping[str, Any]) -> int | None:
    for key in (
        "num_materialized_examples",
        "materialized_count",
        "num_examples",
        "num_test",
        "expected_examples",
        "size",
    ):
        try:
            value = int(row.get(key) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return None


def _source_state(row: Mapping[str, Any]) -> str:
    if _looks_like_hf_dataset_id(row.get("hf_dataset")):
        return "hf_dataset_pinned"
    if _as_list(row.get("direct_files")):
        return "direct_files_pinned"
    if _text(row.get("official_url") or row.get("url") or row.get("source_url")).startswith("http"):
        return "official_source_pinned"
    return "unresolved"


def _materialization_state(row: Mapping[str, Any]) -> tuple[str, int | None]:
    count = _materialized_count(row)
    verified = bool(
        row.get("dataset_cache_verified")
        or row.get("harness_materialized")
        or row.get("materialized")
        or row.get("benchmark_harness_ready")
        or count
    )
    return ("verified" if verified else "required", count)


def _dataset_status(row: Mapping[str, Any]) -> dict[str, Any]:
    source_state = _source_state(row)
    materialization_state, count = _materialization_state(row)
    if source_state == "unresolved":
        loop_status = "source_resolution_required"
    elif materialization_state != "verified":
        loop_status = "dataset_materialization_required"
    elif row.get("requires_harness") or row.get("generated_runner_supported") is False:
        loop_status = "benchmark_harness_code_required"
    else:
        loop_status = "ready_for_formal_forge"
    return {
        "name": row.get("name") or row.get("requested_name") or row.get("hf_dataset") or "dataset",
        "requested_name": row.get("requested_name") or "",
        "hf_dataset": row.get("hf_dataset") or "",
        "official_url": row.get("official_url") or row.get("url") or row.get("source_url") or "",
        "direct_files": _as_list(row.get("direct_files")),
        "requires_harness": bool(row.get("requires_harness")),
        "generated_runner_supported": row.get("generated_runner_supported") is True,
        "source_state": source_state,
        "materialization_state": materialization_state,
        "materialized_count": count,
        "loop_status": loop_status,
    }


def _overall_status(rows: list[dict[str, Any]]) -> str:
    statuses = [row.get("loop_status") for row in rows]
    for status in (
        "source_resolution_required",
        "dataset_materialization_required",
        "benchmark_harness_code_required",
    ):
        if status in statuses:
            return status
    if statuses:
        return "ready_for_formal_forge"
    return "source_resolution_required"


def prepare_harness_loop_task(
    task: Mapping[str, Any],
    *,
    benchmark_name: str | None = None,
    loop_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a task with normalized refs and a structured harness-loop state."""

    updated = dict(task or {})
    refs = _as_list(updated.get("dataset_refs"))
    normalized_refs = [normalize_harness_dataset_ref(ref, benchmark_name=benchmark_name) for ref in refs]
    dataset_rows = [_dataset_status(ref) for ref in normalized_refs]
    status = _overall_status(dataset_rows)
    meta = _STATUS_META[status]
    materialization_plan = {
        "schema_version": "dataset_materialization_plan_v1",
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "benchmark_name": benchmark_name or updated.get("benchmark_names") or updated.get("benchmark_name") or "",
        "datasets": dataset_rows,
    }
    if not dataset_rows:
        materialization_plan["blockers"] = ["benchmark_harness_task.dataset_refs is empty"]

    updated["dataset_refs"] = normalized_refs
    updated["dataset_materialization_plan"] = materialization_plan
    updated["dataset_materialization_status"] = {
        "status": status,
        "datasets": dataset_rows,
    }
    if loop_report:
        updated["loop_router"] = dict(loop_report)
    updated["loop_state"] = {
        "status": status,
        "owner": meta["owner"],
        "stage": meta["stage"],
        "next_actions": list(meta["next_actions"]),
        "dataset_count": len(dataset_rows),
        "ready_for_formal_forge": status == "ready_for_formal_forge",
    }
    return updated
