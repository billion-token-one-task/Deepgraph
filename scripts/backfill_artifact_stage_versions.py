#!/usr/bin/env python3
"""Append immutable full/holdout artifact versions for one existing run.

Dry-run is the default. ``--apply`` only writes new experiment_artifacts rows
and content-addressed snapshots; it never updates or deletes a legacy row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import database as db
from orchestrator.colab_worker import (
    _artifact_snapshot_path,
    _register_artifact_version,
    _verified_request_artifacts,
)


BACKFILL_STAGES = ("full_benchmark", "evidence_audit")


def _mapping(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _file_sha256(path: Path) -> str | None:
    try:
        if not path.is_file() or path.is_symlink():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _legacy_unresolved(run_id: int) -> dict[str, list[dict]]:
    unresolved: dict[str, list[dict]] = {}
    rows = db.fetchall(
        """
        SELECT id, artifact_type, path, content_sha256, metadata
        FROM experiment_artifacts
        WHERE run_id=? AND artifact_stage IS NULL
        ORDER BY id
        """,
        (int(run_id),),
    )
    for raw in rows:
        row = dict(raw)
        metadata = _mapping(row.get("metadata"))
        expected = str(row.get("content_sha256") or metadata.get("sha256") or "")
        observed = _file_sha256(Path(str(row.get("path") or "")))
        if expected and observed == expected:
            continue
        reason = "legacy_path_missing" if observed is None else "legacy_sha_mismatch"
        unresolved.setdefault(str(row.get("artifact_type") or ""), []).append(
            {
                "artifact_id": int(row["id"]),
                "expected_sha256": expected or None,
                "observed_sha256": observed,
                "reason": reason,
                "status": "legacy_unresolved",
            }
        )
    return unresolved


def build_plan(run_id: int) -> tuple[dict, list[dict], dict[str, list[dict]]]:
    run = db.fetchone(
        "SELECT id, agenda_id, deep_insight_id, workdir FROM experiment_runs WHERE id=?",
        (int(run_id),),
    )
    if not run:
        raise RuntimeError(f"experiment run {run_id} does not exist")
    run = dict(run)
    requests = db.fetchall(
        """
        SELECT cwr.id, cwr.experiment_run_id, cwr.agenda_id, cwr.idea_id,
               cwr.resource_grant_id, cwr.stage, cwr.artifact_output_dir,
               cwr.completed_at, rg.stage AS grant_stage
        FROM colab_work_requests_v1 AS cwr
        JOIN resource_grants AS rg ON rg.id=cwr.resource_grant_id
        WHERE cwr.experiment_run_id=? AND cwr.status='succeeded'
          AND cwr.stage IN ('full_benchmark', 'evidence_audit')
        ORDER BY cwr.completed_at, cwr.id
        """,
        (int(run_id),),
    )
    if not requests:
        raise RuntimeError(f"run {run_id} has no succeeded full/holdout requests")
    unresolved = _legacy_unresolved(run_id)
    items: list[dict] = []
    for raw_request in requests:
        request = dict(raw_request)
        stage = str(request.get("stage") or "").strip().lower()
        if stage != str(request.get("grant_stage") or "").strip().lower():
            raise RuntimeError(f"request {request['id']} stage differs from its grant")
        _payload, verification, artifacts = _verified_request_artifacts(run, request)
        for artifact in artifacts:
            artifact_type = str(artifact["artifact_type"])
            digest = str(artifact["content_sha256"])
            existing = db.fetchone(
                """
                SELECT id, path FROM experiment_artifacts
                WHERE agenda_id=? AND run_id=? AND artifact_type=?
                  AND artifact_stage=? AND content_sha256=?
                ORDER BY id DESC LIMIT 1
                """,
                (int(run["agenda_id"]), int(run_id), artifact_type, stage, digest),
            )
            snapshot = _artifact_snapshot_path(
                workdir=Path(str(run["workdir"])),
                artifact_stage=stage,
                artifact_type=artifact_type,
                source_path=Path(artifact["source_path"]),
                content_sha256=digest,
            )
            items.append(
                {
                    "action": "existing" if existing else "append",
                    "artifact_type": artifact_type,
                    "content_sha256": digest,
                    "existing_artifact_id": int(existing["id"]) if existing else None,
                    "legacy_unresolved": unresolved.get(artifact_type, []),
                    "metric_key": verification.metric_name,
                    "metric_value": (
                        verification.candidate_value
                        if artifact_type == "final_results"
                        else None
                    ),
                    "request_id": int(request["id"]),
                    "resource_grant_id": int(request["resource_grant_id"]),
                    "snapshot_path": str(snapshot),
                    "source_path": str(artifact["source_path"]),
                    "stage": stage,
                    "_content": artifact["content"],
                }
            )
    return run, items, unresolved


def run(*, run_id: int, apply: bool = False) -> dict:
    run_row, items, unresolved = build_plan(run_id)
    if apply:
        try:
            for item in items:
                if item["action"] == "existing":
                    continue
                item["artifact_id"] = _register_artifact_version(
                    run=run_row,
                    resource_grant_id=int(item["resource_grant_id"]),
                    artifact_stage=str(item["stage"]),
                    artifact_type=str(item["artifact_type"]),
                    source_path=Path(str(item["source_path"])),
                    content=item["_content"],
                    content_sha256=str(item["content_sha256"]),
                    metric_key=str(item["metric_key"]),
                    metric_value=item["metric_value"],
                    legacy_unresolved=list(item["legacy_unresolved"]),
                )
            db.commit()
        except Exception:
            db.rollback()
            raise
    public_items = [{key: value for key, value in item.items() if key != "_content"} for item in items]
    return {
        "mode": "apply" if apply else "dry_run",
        "production_database_written": bool(apply),
        "run_id": int(run_id),
        "items": public_items,
        "legacy_unresolved": unresolved,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.run_id <= 0:
        parser.error("--run-id must be positive")
    print(json.dumps(run(run_id=args.run_id, apply=args.apply), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
