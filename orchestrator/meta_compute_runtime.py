"""Runtime bridge from the v1 compute control plane to legacy GPU workers.

The legacy scheduler remains a transport during the controlled port.  New
submissions enter through ``ComputeScheduler`` and its durable repository, so
the legacy queue no longer owns admission, idempotency, or grant authority.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config import GPU_MODE
from contracts.meta_harness import ResourceGrant
from db import database as db
from meta_harness.compute import (
    ArtifactCollection,
    BackendCapability,
    ComputeBackendError,
    ComputeJob,
    ComputeScheduler,
    ComputeSubmission,
    LocalGPUBackend,
    SSHGPUBackend,
    SSHGPUConfig,
    UsageAccounting,
)
from meta_harness.compute_repository import ComputeJobRepository


def _load_json_list(value) -> list:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _parse_backend_job_id(value: str) -> int:
    prefix = "legacy-gpu-job:"
    if not str(value).startswith(prefix):
        raise ComputeBackendError("legacy GPU backend job id is invalid")
    try:
        job_id = int(str(value)[len(prefix) :])
    except ValueError as exc:
        raise ComputeBackendError("legacy GPU backend job id is invalid") from exc
    if job_id <= 0:
        raise ComputeBackendError("legacy GPU backend job id is invalid")
    return job_id


def _parse_run_ref(value: str) -> int:
    prefix = "experiment-run:"
    if not str(value).startswith(prefix):
        raise ComputeBackendError(
            "legacy GPU command_ref must identify an experiment run"
        )
    try:
        run_id = int(str(value)[len(prefix) :])
    except ValueError as exc:
        raise ComputeBackendError("experiment run command_ref is invalid") from exc
    if run_id <= 0:
        raise ComputeBackendError("experiment run command_ref is invalid")
    return run_id


def _iso(value) -> str | None:
    return str(value) if value else None


class LegacyGPUQueueTransport:
    """Backend-neutral adapter over the existing local/SSH worker queue."""

    def __init__(self, backend_kind: str):
        if backend_kind not in {"local_gpu", "ssh_gpu"}:
            raise ComputeBackendError("unsupported legacy GPU transport kind")
        self.backend_kind = backend_kind

    def capability(self) -> BackendCapability:
        metadata_filter = (
            "metadata LIKE ?" if self.backend_kind == "ssh_gpu"
            else "(metadata IS NULL OR metadata NOT LIKE ?)"
        )
        row = db.fetchone(
            f"""
            SELECT COUNT(*) AS gpu_count,
                   COALESCE(MAX(total_mem_gb), 0) AS max_vram_gb
            FROM gpu_workers
            WHERE {metadata_filter}
            """,
            ('%"backend": "ssh"%',),
        ) or {}
        gpu_count = int(row.get("gpu_count") or 0)
        return BackendCapability(
            backend_kind=self.backend_kind,
            available=gpu_count > 0,
            gpu_count=gpu_count,
            vram_gb=float(row.get("max_vram_gb") or 0),
            detail={"transport": "legacy_gpu_queue"},
        )

    def submit(self, request: ComputeSubmission) -> ComputeJob:
        from orchestrator import gpu_scheduler

        run_id = _parse_run_ref(request.command_ref)
        run = db.fetchone(
            """
            SELECT id, agenda_id, deep_insight_id, resource_grant_id,
                   resource_class
            FROM experiment_runs
            WHERE id=?
            """,
            (run_id,),
        )
        if (
            not run
            or int(run.get("agenda_id") or 0) != request.agenda_id
            or int(run.get("deep_insight_id") or 0) != request.idea_id
            or int(run.get("resource_grant_id") or 0)
            != request.resource_grant_id
        ):
            raise ComputeBackendError(
                "legacy GPU transport experiment scope mismatch"
            )
        resource_class = str(run.get("resource_class") or "gpu_small")
        job_id = gpu_scheduler.queue_run(
            insight_id=request.idea_id,
            run_id=run_id,
            resource_grant_id=request.resource_grant_id,
            resource_class=resource_class,
            priority=3 if request.stage == "full_benchmark" else 1,
            vram_required_gb=40 if resource_class == "gpu_large" else 16,
            timeout_s=request.timeout_seconds,
        )
        return ComputeJob(
            backend_kind=self.backend_kind,
            backend_job_id=f"legacy-gpu-job:{job_id}",
            idempotency_key=request.idempotency_key,
            status="submitted",
            heartbeat_at=datetime.now(timezone.utc).isoformat(),
        )

    def _row(self, backend_job_id: str) -> dict:
        row = db.fetchone(
            """
            SELECT gj.*, gw.heartbeat_at AS worker_heartbeat
            FROM gpu_jobs AS gj
            LEFT JOIN gpu_workers AS gw ON gw.id=gj.assigned_worker
            WHERE gj.id=?
            """,
            (_parse_backend_job_id(backend_job_id),),
        )
        if not row:
            raise ComputeBackendError("legacy GPU job was not found")
        return row

    def status(self, backend_job_id: str) -> ComputeJob:
        row = self._row(backend_job_id)
        legacy = str(row.get("status") or "")
        status_map = {
            "queued": "submitted",
            "running": "running",
            "completed": "succeeded",
            "failed": "failed",
            "cancelled": "cancelled",
        }
        status = status_map.get(legacy)
        if status is None:
            raise ComputeBackendError(f"unknown legacy GPU status:{legacy}")
        failure_reason = str(row.get("error_message") or "").strip() or None
        # The legacy worker may record post-run errors while setting completed.
        # Such a row is not a successful v1 backend result.
        if status == "succeeded" and failure_reason:
            status = "failed"
        return ComputeJob(
            backend_kind=self.backend_kind,
            backend_job_id=backend_job_id,
            idempotency_key=str(row.get("meta_harness_idempotency_key") or "persisted"),
            status=status,
            heartbeat_at=_iso(
                row.get("worker_heartbeat")
                or row.get("completed_at")
                or row.get("started_at")
            ),
            failure_reason=failure_reason if status == "failed" else None,
        )

    def heartbeat(self, backend_job_id: str) -> ComputeJob:
        return self.status(backend_job_id)

    def cancel(self, backend_job_id: str) -> ComputeJob:
        job_id = _parse_backend_job_id(backend_job_id)
        row = self._row(backend_job_id)
        if str(row.get("status") or "") != "queued":
            raise ComputeBackendError(
                "running legacy GPU cancellation requires manual worker control"
            )
        cursor = db.execute(
            """
            UPDATE gpu_jobs
            SET status='cancelled', completed_at=CURRENT_TIMESTAMP,
                error_message='cancelled_before_dispatch'
            WHERE id=? AND agenda_id=? AND status='queued'
            """,
            (job_id, int(row["agenda_id"])),
        )
        if int(getattr(cursor, "rowcount", 0) or 0) != 1:
            db.rollback()
            raise ComputeBackendError("legacy GPU cancellation race")
        db.commit()
        return ComputeJob(
            self.backend_kind,
            backend_job_id,
            "persisted",
            "cancelled",
            heartbeat_at=datetime.now(timezone.utc).isoformat(),
        )

    def collect_artifacts(
        self, backend_job_id: str, requirements: tuple[str, ...]
    ) -> ArtifactCollection:
        row = self._row(backend_job_id)
        run_id = int(row.get("experiment_run_id") or 0)
        artifacts = db.fetchall(
            """
            SELECT artifact_type, path
            FROM experiment_artifacts
            WHERE agenda_id=? AND run_id=?
            ORDER BY id
            """,
            (int(row.get("agenda_id") or 0), run_id),
        )
        evidence_counts = db.fetchone(
            """
            SELECT
                (SELECT COUNT(*) FROM experiment_iterations
                 WHERE agenda_id=? AND run_id=?) AS iteration_count,
                (SELECT COUNT(*) FROM experimental_claims
                 WHERE agenda_id=? AND run_id=?) AS claim_count
            """,
            (
                int(row.get("agenda_id") or 0),
                run_id,
                int(row.get("agenda_id") or 0),
                run_id,
            ),
        ) or {}
        manifest: dict[str, object] = {
            "artifacts": [
                {
                    "name": str(item.get("artifact_type") or ""),
                    "path": str(item.get("path") or ""),
                }
                for item in artifacts
            ],
            "legacy_gpu_job_id": int(row["id"]),
            "experiment_run_id": run_id,
            "run_manifest": {
                "uri": f"db:experiment_runs:{run_id}",
            },
        }
        names = {
            str(item.get("artifact_type") or "")
            for item in artifacts
            if item.get("artifact_type")
        }
        for item in artifacts:
            filename = Path(str(item.get("path") or "")).name
            if filename:
                names.add(filename)
            if filename in {
                "environment_manifest.json",
                "environment.json",
                "runtime_environment.json",
            }:
                names.add("environment_manifest")
                manifest["environment_manifest"] = {
                    "uri": str(item.get("path") or "")
                }
        names.add("run_manifest")
        if int(evidence_counts.get("iteration_count") or 0) > 0:
            names.add("raw_metrics")
            manifest["raw_metrics"] = {
                "uri": f"db:experiment_iterations:run:{run_id}",
                "row_count": int(evidence_counts["iteration_count"]),
            }
        if int(evidence_counts.get("claim_count") or 0) > 0:
            names.add("claim_ledger")
            manifest["claim_ledger"] = {
                "uri": f"db:experimental_claims:run:{run_id}",
                "row_count": int(evidence_counts["claim_count"]),
            }
        missing = tuple(sorted(set(requirements) - names))
        return ArtifactCollection(
            manifest=manifest,
            complete=bool(artifacts) and not missing,
            missing_requirements=missing,
        )

    def usage(self, backend_job_id: str) -> UsageAccounting:
        row = self._row(backend_job_id)
        run_id = int(row.get("experiment_run_id") or 0)
        usage = db.fetchone(
            """
            SELECT COALESCE(SUM(duration_seconds), 0) AS measured_seconds,
                   COALESCE(MAX(peak_memory_mb), 0) AS peak_memory_mb
            FROM experiment_iterations
            WHERE agenda_id=? AND run_id=?
            """,
            (int(row.get("agenda_id") or 0), run_id),
        ) or {}
        measured_seconds = float(usage.get("measured_seconds") or 0)
        gpu_count = max(1, int(row.get("gpu_count") or 1))
        return UsageAccounting(
            wall_seconds=measured_seconds,
            gpu_hours=measured_seconds * gpu_count / 3600.0,
            cpu_core_hours=0.0,
            backend_report={
                "source": "experiment_iterations.duration_seconds",
                "peak_memory_mb": float(usage.get("peak_memory_mb") or 0),
                "legacy_gpu_job_id": int(row["id"]),
            },
        )


def _backend_kind() -> str:
    return "ssh_gpu" if str(GPU_MODE).strip().lower() == "ssh" else "local_gpu"


def build_scheduler() -> ComputeScheduler:
    kind = _backend_kind()
    transport = LegacyGPUQueueTransport(kind)
    if kind == "ssh_gpu":
        backend = SSHGPUBackend(
            transport,
            SSHGPUConfig(
                target_ref="env:DEEPGRAPH_SSH_TARGET",
                credential_ref="env:DEEPGRAPH_SSH_CREDENTIAL",
                artifact_root="workspace/meta_harness/artifacts",
            ),
        )
    else:
        backend = LocalGPUBackend(transport)
    return ComputeScheduler(
        [backend],
        job_store=ComputeJobRepository(),
    )


def _grant_from_row(row: dict) -> ResourceGrant:
    return ResourceGrant(
        agenda_id=int(row["agenda_id"]),
        idea_id=int(row["idea_id"]),
        decision_packet_id=int(row["decision_packet_id"]),
        stage=str(row["stage"]),
        token_cap=int(row.get("token_cap") or 0),
        gpu_class=str(row.get("gpu_class") or "none"),
        max_gpu_hours=float(row.get("max_gpu_hours") or 0),
        backend_allowlist=_load_json_list(row.get("backend_allowlist_json")),
        artifact_requirements=_load_json_list(
            row.get("artifact_requirements_json")
        ),
        expires_at=str(row["expires_at"]),
        grant_reason=str(row["grant_reason"]),
        idempotency_key=str(row["idempotency_key"]),
        status=str(row["status"]),
        grant_id=int(row["id"]),
        reservation_id=int(row["reservation_id"]),
    )


def submit_experiment_run(
    *,
    agenda_id: int,
    idea_id: int,
    experiment_run_id: int,
    resource_grant_id: int,
    timeout_seconds: int,
) -> ComputeJob:
    grant_row = db.fetchone(
        """
        SELECT * FROM resource_grants
        WHERE id=? AND agenda_id=? AND idea_id=?
          AND status='active' AND expires_at > CURRENT_TIMESTAMP
        """,
        (resource_grant_id, agenda_id, idea_id),
    )
    if not grant_row:
        raise ComputeBackendError(
            "active scoped ResourceGrant is required for compute"
        )
    grant = _grant_from_row(grant_row)
    backend_kind = _backend_kind()
    request = ComputeSubmission(
        agenda_id=int(agenda_id),
        idea_id=int(idea_id),
        stage=grant.stage,
        resource_grant_id=int(resource_grant_id),
        idempotency_key=(
            f"experiment-run:{agenda_id}:{idea_id}:{experiment_run_id}:"
            f"{grant.stage}"
        ),
        command_ref=f"experiment-run:{int(experiment_run_id)}",
        artifact_namespace=(
            f"agenda-{agenda_id}/idea-{idea_id}/run-{experiment_run_id}"
        ),
        timeout_seconds=int(timeout_seconds),
        requested_gpu_hours=float(grant.max_gpu_hours),
    )
    return build_scheduler().submit(
        request,
        grant=grant,
        preferred_backends=[backend_kind],
    )


def settle_legacy_job(gpu_job_id: int) -> str:
    """Mirror a legacy worker observation into durable v1 compute state."""
    backend_job_id = f"legacy-gpu-job:{int(gpu_job_id)}"
    row = db.fetchone(
        """
        SELECT cj.backend_kind, cj.backend_job_id, cj.idempotency_key,
               cj.status, cj.heartbeat_at, cj.failure_reason,
               rg.artifact_requirements_json
        FROM compute_jobs_v1 AS cj
        JOIN resource_grants AS rg ON rg.id=cj.resource_grant_id
        WHERE cj.backend_job_id=?
        """,
        (backend_job_id,),
    )
    if not row:
        return "not_managed"
    if str(row.get("status") or "") in {
        "succeeded",
        "failed",
        "cancelled",
        "timed_out",
        "submission_unknown",
        "usage_unknown",
    }:
        return str(row.get("status"))
    scheduler = build_scheduler()
    observed = scheduler.refresh_and_settle(
        ComputeJob(
            backend_kind=str(row["backend_kind"]),
            backend_job_id=str(row["backend_job_id"]),
            idempotency_key=str(row["idempotency_key"]),
            status=(
                str(row["status"])
                if str(row["status"]) in {"submitted", "running", "cancel_requested"}
                else "running"
            ),
            heartbeat_at=_iso(row.get("heartbeat_at")),
            failure_reason=row.get("failure_reason"),
        ),
        requirements=tuple(
            str(value)
            for value in _load_json_list(
                row.get("artifact_requirements_json")
            )
        ),
    )
    return observed.status


def reconcile_on_startup() -> dict[str, int]:
    if not db._use_pg():  # noqa: SLF001
        raise ComputeBackendError(
            "meta-harness compute recovery requires PostgreSQL"
        )
    repository = ComputeJobRepository()
    totals = {
        "agendas": 0,
        "submission_unknown": 0,
        "usage_unknown": 0,
        "terminal_settled": 0,
        "settlement_errors": 0,
    }
    agendas = db.fetchall(
        """
        SELECT DISTINCT agenda_id
        FROM compute_jobs_v1
        WHERE status IN ('submitting', 'submitted', 'running',
                         'cancel_requested', 'collecting')
        ORDER BY agenda_id
        """
    )
    for agenda in agendas:
        agenda_id = int(agenda["agenda_id"])
        recovered = repository.reconcile_expired(agenda_id=agenda_id)
        totals["agendas"] += 1
        totals["submission_unknown"] += recovered["submission_unknown"]
        totals["usage_unknown"] += recovered["usage_unknown"]
        live = db.fetchall(
            """
            SELECT backend_job_id
            FROM compute_jobs_v1
            WHERE agenda_id=?
              AND backend_kind IN ('local_gpu', 'ssh_gpu')
              AND backend_job_id LIKE 'legacy-gpu-job:%'
              AND status IN ('submitted', 'running', 'cancel_requested',
                             'collecting')
            ORDER BY id
            """,
            (agenda_id,),
        )
        for row in live:
            try:
                result = settle_legacy_job(
                    _parse_backend_job_id(str(row["backend_job_id"]))
                )
                if result in {"succeeded", "failed", "cancelled", "timed_out"}:
                    totals["terminal_settled"] += 1
            except Exception:
                db.rollback()
                totals["settlement_errors"] += 1
    return totals
