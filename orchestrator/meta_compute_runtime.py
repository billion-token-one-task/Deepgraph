"""Runtime bridge from the v1 compute control plane to legacy GPU workers.

The legacy scheduler remains a transport during the controlled port.  New
submissions enter through ``ComputeScheduler`` and its durable repository, so
the legacy queue no longer owns admission, idempotency, or grant authority.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config import (
    COMPUTE_ARTIFACT_ROOT,
    COMPUTE_BACKENDS_ENABLED,
    COMPUTE_COLAB_ACCOUNTS_MANIFEST_REF,
    COMPUTE_COLAB_ALLOWED_ARTIFACT_ROOT,
    COMPUTE_COLAB_ALLOWED_CODE_ROOT,
    COMPUTE_COLAB_CLI_BINARY,
    COMPUTE_SSH_CREDENTIAL_REF,
    COMPUTE_SSH_TARGET_REF,
    GPU_MODE,
)
from contracts.meta_harness import ResourceGrant
from db import database as db
from meta_harness.compute import (
    ACTIVE_JOB_STATES,
    ArtifactCollection,
    BackendCapability,
    ColabGPUBackend,
    CPUBackend,
    ComputeBackendError,
    ComputeJob,
    ComputeScheduler,
    ComputeSubmission,
    LocalGPUBackend,
    SSHGPUBackend,
    SSHGPUConfig,
    UsageAccounting,
)
from meta_harness.backend_capability import (
    BackendCapabilityError,
    STATE_ENABLED,
    reports_from_config,
    require_schedulable,
)
from meta_harness.compute_repository import ComputeJobRepository
from meta_harness.attempt_gpu_usage import (
    AttemptGPUUsageError,
    GrantGPUUsageControl,
)
from meta_harness.backends.colab_durable import (
    ColabWorkRepository,
    ColabWorkSpec,
    build_transport as build_colab_transport,
)


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


def _parse_cpu_job_id(value: str) -> int:
    prefix = "cpu-experiment-run:"
    if not str(value).startswith(prefix):
        raise ComputeBackendError("CPU backend job id is invalid")
    try:
        run_id = int(str(value)[len(prefix) :])
    except ValueError as exc:
        raise ComputeBackendError("CPU backend job id is invalid") from exc
    if run_id <= 0:
        raise ComputeBackendError("CPU backend job id is invalid")
    return run_id


def _iso(value) -> str | None:
    return str(value) if value else None


class LegacyCPUValidationTransport:
    """Durable admission adapter for the synchronous validation worker.

    ``submit`` only creates the backend identity. The auto-research worker
    explicitly marks it running before invoking the legacy validation loop,
    then settles measured usage and artifacts through ``ComputeScheduler``.
    """

    backend_kind = "cpu"

    def capability(self) -> BackendCapability:
        return BackendCapability(
            backend_kind=self.backend_kind,
            available=True,
            detail={
                "transport": "legacy_cpu_validation",
                "admission": "durable_compute_jobs_v1",
            },
        )

    def _run(self, backend_job_id: str) -> dict:
        run_id = _parse_cpu_job_id(backend_job_id)
        row = db.fetchone(
            """
            SELECT id, agenda_id, deep_insight_id, resource_grant_id, status,
                   started_at, completed_at, error_message
            FROM experiment_runs
            WHERE id=?
            """,
            (run_id,),
        )
        if not row:
            raise ComputeBackendError("CPU experiment run was not found")
        return row

    def submit(self, request: ComputeSubmission) -> ComputeJob:
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
            or str(run.get("resource_class") or "cpu") != "cpu"
        ):
            raise ComputeBackendError("CPU experiment scope mismatch")
        return ComputeJob(
            backend_kind=self.backend_kind,
            backend_job_id=f"cpu-experiment-run:{run_id}",
            idempotency_key=request.idempotency_key,
            status="submitted",
            heartbeat_at=datetime.now(timezone.utc).isoformat(),
        )

    def status(self, backend_job_id: str) -> ComputeJob:
        row = self._run(backend_job_id)
        legacy = str(row.get("status") or "").strip().lower()
        if legacy == "completed":
            status = "succeeded"
        elif legacy in {"failed", "superseded", "reset", "archived"}:
            status = "failed"
        elif legacy in {"cancelled", "canceled"}:
            status = "cancelled"
        elif legacy in {"reproducing", "testing", "running", "running_cpu"}:
            status = "running"
        else:
            status = "submitted"
        reason = str(row.get("error_message") or "").strip() or None
        if status == "failed" and reason is None:
            reason = f"experiment_run_{legacy or 'failed'}"
        return ComputeJob(
            backend_kind=self.backend_kind,
            backend_job_id=backend_job_id,
            idempotency_key="persisted",
            status=status,
            heartbeat_at=_iso(
                row.get("completed_at")
                or row.get("started_at")
            ),
            failure_reason=reason if status == "failed" else None,
        )

    def heartbeat(self, backend_job_id: str) -> ComputeJob:
        return self.status(backend_job_id)

    def cancel(self, backend_job_id: str) -> ComputeJob:
        row = self._run(backend_job_id)
        legacy = str(row.get("status") or "").strip().lower()
        if legacy not in {"planned", "ready", "queued", "pending"}:
            raise ComputeBackendError(
                "running CPU cancellation requires cooperative worker control"
            )
        cursor = db.execute(
            """
            UPDATE experiment_runs
            SET status='cancelled', completed_at=CURRENT_TIMESTAMP,
                error_message='cancelled_before_dispatch'
            WHERE id=? AND agenda_id=? AND status=?
            """,
            (int(row["id"]), int(row["agenda_id"]), legacy),
        )
        if int(getattr(cursor, "rowcount", 0) or 0) != 1:
            db.rollback()
            raise ComputeBackendError("CPU cancellation race")
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
        row = self._run(backend_job_id)
        run_id = int(row["id"])
        artifacts = db.fetchall(
            """
            SELECT artifact_type, path
            FROM experiment_artifacts
            WHERE agenda_id=? AND run_id=?
            ORDER BY id
            """,
            (int(row["agenda_id"]), run_id),
        )
        counts = db.fetchone(
            """
            SELECT
                (SELECT COUNT(*) FROM experiment_iterations
                 WHERE agenda_id=? AND run_id=?) AS iteration_count,
                (SELECT COUNT(*) FROM experimental_claims
                 WHERE agenda_id=? AND run_id=?) AS claim_count
            """,
            (
                int(row["agenda_id"]),
                run_id,
                int(row["agenda_id"]),
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
            "experiment_run_id": run_id,
            "run_manifest": {"uri": f"db:experiment_runs:{run_id}"},
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
                "environment_report.json",
                "runtime_environment.json",
            }:
                names.add("environment_manifest")
                manifest["environment_manifest"] = {
                    "uri": str(item.get("path") or "")
                }
        names.add("run_manifest")
        if int(counts.get("iteration_count") or 0) > 0:
            names.add("raw_metrics")
            manifest["raw_metrics"] = {
                "uri": f"db:experiment_iterations:run:{run_id}",
                "row_count": int(counts["iteration_count"]),
            }
        if int(counts.get("claim_count") or 0) > 0:
            names.add("claim_ledger")
            manifest["claim_ledger"] = {
                "uri": f"db:experimental_claims:run:{run_id}",
                "row_count": int(counts["claim_count"]),
            }
        missing = tuple(sorted(set(requirements) - names))
        return ArtifactCollection(
            manifest=manifest,
            complete=bool(artifacts) and not missing,
            missing_requirements=missing,
        )

    def usage(self, backend_job_id: str) -> UsageAccounting:
        row = self._run(backend_job_id)
        usage = db.fetchone(
            """
            SELECT COALESCE(SUM(duration_seconds), 0) AS measured_seconds,
                   COALESCE(MAX(peak_memory_mb), 0) AS peak_memory_mb
            FROM experiment_iterations
            WHERE agenda_id=? AND run_id=?
            """,
            (int(row["agenda_id"]), int(row["id"])),
        ) or {}
        measured_seconds = float(usage.get("measured_seconds") or 0)
        return UsageAccounting(
            wall_seconds=measured_seconds,
            gpu_hours=0.0,
            cpu_core_hours=measured_seconds / 3600.0,
            backend_report={
                "source": "experiment_iterations.duration_seconds",
                "cpu_core_assumption": 1,
                "peak_memory_mb": float(usage.get("peak_memory_mb") or 0),
            },
        )


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
            meta_harness_idempotency_key=request.idempotency_key,
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
        compute_job = db.fetchone(
            "SELECT id FROM compute_jobs_v1 WHERE backend_kind=? AND backend_job_id=?",
            (self.backend_kind, str(backend_job_id)),
        )
        if not compute_job:
            raise ComputeBackendError(
                "canonical durable compute job is missing for GPU usage"
            )
        try:
            measured = GrantGPUUsageControl().usage_for_compute_job(
                int(compute_job["id"])
            )
        except AttemptGPUUsageError as exc:
            raise ComputeBackendError(str(exc)) from exc
        return UsageAccounting(
            wall_seconds=float(measured["wall_seconds"]),
            gpu_hours=float(measured["gpu_hours"]),
            cpu_core_hours=0.0,
            backend_report={
                "source": "experiment_attempt_gpu_reservations_v1",
                "attempt_reservation_id": int(
                    measured["attempt_reservation_id"]
                ),
                "gpu_count": int(measured["gpu_count"]),
                "reason_code": measured.get("reason_code"),
                "legacy_gpu_job_id": int(row["id"]),
            },
        )


def _backend_kind() -> str:
    return "ssh_gpu" if str(GPU_MODE).strip().lower() == "ssh" else "local_gpu"


def _enabled_backend_kinds() -> set[str]:
    enabled = {
        str(value).strip().lower()
        for value in (COMPUTE_BACKENDS_ENABLED or [])
        if str(value).strip()
    }
    unknown = enabled - {"cpu", "local_gpu", "ssh_gpu", "colab_gpu"}
    if unknown:
        raise ComputeBackendError(
            "unknown configured compute backend(s):" + ",".join(sorted(unknown))
        )
    return enabled


def build_scheduler() -> ComputeScheduler:
    """Build only backends whose capability is proven. No silent fallback.

    A backend that is configured but unverified is ``unknown``; it is left out
    of the scheduler entirely so that ordinary work can never land on it. A
    canary reaches it through a separately authorized path, not through here.
    """
    enabled = {
        kind
        for kind in _enabled_backend_kinds()
        if reports_from_config()[kind].state == STATE_ENABLED
    }
    backends = []
    if "cpu" in enabled:
        backends.append(CPUBackend(LegacyCPUValidationTransport()))
    kind = _backend_kind()
    if kind in enabled:
        transport = LegacyGPUQueueTransport(kind)
        if kind == "ssh_gpu":
            backends.append(
                SSHGPUBackend(
                    transport,
                    SSHGPUConfig(
                        target_ref=COMPUTE_SSH_TARGET_REF,
                        credential_ref=COMPUTE_SSH_CREDENTIAL_REF,
                        artifact_root=str(COMPUTE_ARTIFACT_ROOT),
                    ),
                )
            )
        else:
            backends.append(LocalGPUBackend(transport))
    if "colab_gpu" in enabled:
        transport = build_colab_transport(
            binary=COMPUTE_COLAB_CLI_BINARY,
            accounts_manifest_ref=COMPUTE_COLAB_ACCOUNTS_MANIFEST_REF,
            allowed_code_root=str(COMPUTE_COLAB_ALLOWED_CODE_ROOT),
            allowed_artifact_root=str(COMPUTE_COLAB_ALLOWED_ARTIFACT_ROOT),
        )
        backends.append(ColabGPUBackend(transport, list(transport.accounts)))
    return ComputeScheduler(
        backends,
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
        preflight_result_id=(
            int(row["preflight_result_id"])
            if row.get("preflight_result_id")
            else None
        ),
    )


def _require_grant_preflight(grant_row: dict, *, backend_kind: str) -> None:
    if not db._use_pg():  # noqa: SLF001
        return
    from meta_harness.preflight_repository import (
        CandidatePreflightRepository,
        PreflightPersistenceError,
    )

    try:
        CandidatePreflightRepository().require_passed(
            preflight_result_id=int(grant_row.get("preflight_result_id") or 0),
            agenda_id=int(grant_row["agenda_id"]),
            idea_id=int(grant_row["idea_id"]),
            allowed_backends=(str(backend_kind),),
            required_artifacts=tuple(
                _load_json_list(grant_row.get("artifact_requirements_json"))
            ),
        )
    except PreflightPersistenceError as exc:
        raise ComputeBackendError(str(exc)) from exc


def submit_experiment_run(
    *,
    agenda_id: int,
    idea_id: int,
    experiment_run_id: int,
    resource_grant_id: int,
    timeout_seconds: int,
    backend_kind: str | None = None,
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
    backend_kind = str(backend_kind or _backend_kind())
    if backend_kind not in {"cpu", "local_gpu", "ssh_gpu"}:
        raise ComputeBackendError(
            "runtime backend must be cpu or the configured legacy GPU transport"
        )
    if backend_kind in {"local_gpu", "ssh_gpu"} and backend_kind != _backend_kind():
        raise ComputeBackendError("requested GPU backend is not configured")
    if backend_kind not in _enabled_backend_kinds():
        raise ComputeBackendError(
            f"requested compute backend is disabled:{backend_kind}"
        )
    try:
        require_schedulable(backend_kind, reports_from_config())
    except BackendCapabilityError as exc:
        raise ComputeBackendError(str(exc)) from exc
    _require_grant_preflight(grant_row, backend_kind=backend_kind)
    attempt_key = (
        f"experiment-run:{agenda_id}:{idea_id}:{experiment_run_id}:"
        f"{grant.stage}"
    )
    effective_timeout_seconds = int(timeout_seconds)
    requested_gpu_hours = 0.0
    attempt_reservation = None
    if backend_kind != "cpu":
        try:
            attempt_reservation = GrantGPUUsageControl().reserve_attempt(
                agenda_id=int(agenda_id),
                idea_id=int(idea_id),
                resource_grant_id=int(resource_grant_id),
                attempt_key=attempt_key,
                backend_kind=backend_kind,
                requested_timeout_seconds=int(timeout_seconds),
                gpu_count=1,
                experiment_run_id=int(experiment_run_id),
            )
        except AttemptGPUUsageError as exc:
            raise ComputeBackendError(str(exc)) from exc
        effective_timeout_seconds = attempt_reservation.timeout_seconds
        requested_gpu_hours = attempt_reservation.reserved_gpu_seconds / 3600.0
    request = ComputeSubmission(
        agenda_id=int(agenda_id),
        idea_id=int(idea_id),
        stage=grant.stage,
        resource_grant_id=int(resource_grant_id),
        idempotency_key=attempt_key,
        command_ref=f"experiment-run:{int(experiment_run_id)}",
        artifact_namespace=(
            f"agenda-{agenda_id}/idea-{idea_id}/run-{experiment_run_id}"
        ),
        timeout_seconds=effective_timeout_seconds,
        requested_gpu_hours=requested_gpu_hours,
    )
    try:
        return build_scheduler().submit(
            request,
            grant=grant,
            preferred_backends=[backend_kind],
        )
    except Exception as exc:
        # A transport exception after submission is deliberately quarantined
        # as submission_unknown; its reservation may correspond to a live GPU
        # process and must survive recovery.  Failures known to precede a
        # backend submission release the unstarted reservation immediately.
        if (
            attempt_reservation is not None
            and "backend_submission_outcome_unknown" not in str(exc)
        ):
            GrantGPUUsageControl().release_unstarted(
                attempt_reservation.reservation_id,
                reason_code="compute_submission_failed_before_backend_start",
            )
        raise


def submit_colab_work(spec: ColabWorkSpec) -> ComputeJob:
    """Persist an operator request, then admit it through durable compute."""
    spec.validate()
    if "colab_gpu" not in _enabled_backend_kinds():
        raise ComputeBackendError("requested compute backend is disabled:colab_gpu")
    repository = ColabWorkRepository()
    request_id = repository.create(spec)
    grant_row = db.fetchone(
        """
        SELECT * FROM resource_grants
        WHERE id=? AND agenda_id=? AND idea_id=?
          AND status='active' AND expires_at > CURRENT_TIMESTAMP
        """,
        (spec.resource_grant_id, spec.agenda_id, spec.idea_id),
    )
    if not grant_row:
        raise ComputeBackendError(
            "active scoped ResourceGrant is required for Colab work"
        )
    grant = _grant_from_row(grant_row)
    _require_grant_preflight(grant_row, backend_kind="colab_gpu")
    try:
        attempt_reservation = GrantGPUUsageControl().reserve_attempt(
            agenda_id=spec.agenda_id,
            idea_id=spec.idea_id,
            resource_grant_id=spec.resource_grant_id,
            attempt_key=spec.idempotency_key,
            backend_kind="colab_gpu",
            requested_timeout_seconds=spec.timeout_seconds,
            gpu_count=1,
            experiment_run_id=spec.experiment_run_id,
        )
    except AttemptGPUUsageError as exc:
        raise ComputeBackendError(str(exc)) from exc
    request = ComputeSubmission(
        agenda_id=spec.agenda_id,
        idea_id=spec.idea_id,
        stage=spec.stage,
        resource_grant_id=spec.resource_grant_id,
        idempotency_key=spec.idempotency_key,
        command_ref=f"colab-work-request:{request_id}",
        artifact_namespace=(
            f"agenda-{spec.agenda_id}/idea-{spec.idea_id}/"
            f"colab-{request_id}"
        ),
        timeout_seconds=attempt_reservation.timeout_seconds,
        requested_gpu_hours=(
            attempt_reservation.reserved_gpu_seconds / 3600.0
        ),
    )
    try:
        job = build_scheduler().submit(
            request,
            grant=grant,
            preferred_backends=["colab_gpu"],
        )
    except Exception as exc:
        if "backend_submission_outcome_unknown" not in str(exc):
            GrantGPUUsageControl().release_unstarted(
                attempt_reservation.reservation_id,
                reason_code="colab_submission_failed_before_backend_start",
            )
        raise
    repository.bind_compute_job(request_id)
    return job


def mark_cpu_running(job: ComputeJob) -> None:
    if job.backend_kind != "cpu":
        raise ComputeBackendError("only a CPU compute job can enter the CPU lane")
    run_id = _parse_cpu_job_id(job.backend_job_id)
    run = db.fetchone(
        "SELECT agenda_id FROM experiment_runs WHERE id=?",
        (run_id,),
    )
    if not run:
        raise ComputeBackendError("CPU experiment run was not found")
    cursor = db.execute(
        """
        UPDATE experiment_runs
        SET status='running_cpu',
            started_at=COALESCE(started_at, CURRENT_TIMESTAMP)
        WHERE id=? AND agenda_id=?
          AND status NOT IN (
              'completed', 'failed', 'cancelled', 'superseded', 'reset',
              'archived', 'manuscript_blocked', 'bundle_ready'
          )
        """,
        (run_id, int(run["agenda_id"])),
    )
    if int(getattr(cursor, "rowcount", 0) or 0) != 1:
        db.rollback()
        raise ComputeBackendError("CPU experiment run is not startable")
    db.commit()
    ComputeJobRepository().record_backend_state(
        ComputeJob(
            backend_kind=job.backend_kind,
            backend_job_id=job.backend_job_id,
            idempotency_key=job.idempotency_key,
            status="running",
            heartbeat_at=datetime.now(timezone.utc).isoformat(),
        )
    )


def settle_cpu_run(experiment_run_id: int) -> str:
    backend_job_id = f"cpu-experiment-run:{int(experiment_run_id)}"
    row = db.fetchone(
        """
        SELECT cj.backend_kind, cj.backend_job_id, cj.idempotency_key,
               cj.status, cj.heartbeat_at, cj.failure_reason,
               rg.artifact_requirements_json
        FROM compute_jobs_v1 AS cj
        JOIN resource_grants AS rg ON rg.id=cj.resource_grant_id
        WHERE cj.backend_job_id=? AND cj.backend_kind='cpu'
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
    observed = build_scheduler().refresh_and_settle(
        ComputeJob(
            backend_kind="cpu",
            backend_job_id=backend_job_id,
            idempotency_key=str(row["idempotency_key"]),
            status=(
                str(row["status"])
                if str(row["status"]) in {
                    "submitted",
                    "running",
                    "cancel_requested",
                }
                else "running"
            ),
            heartbeat_at=_iso(row.get("heartbeat_at")),
            failure_reason=row.get("failure_reason"),
        ),
        requirements=tuple(
            str(value)
            for value in _load_json_list(row.get("artifact_requirements_json"))
        ),
    )
    return observed.status


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


def settle_colab_request(request_id: int) -> str:
    backend_job_id = f"colab-work-request:{int(request_id)}"
    row = db.fetchone(
        """
        SELECT cj.backend_kind, cj.backend_job_id, cj.idempotency_key,
               cj.status, cj.heartbeat_at, cj.failure_reason,
               rg.artifact_requirements_json
        FROM compute_jobs_v1 AS cj
        JOIN resource_grants AS rg ON rg.id=cj.resource_grant_id
        WHERE cj.backend_job_id=? AND cj.backend_kind='colab_gpu'
        """,
        (backend_job_id,),
    )
    if not row:
        return "not_managed"
    state = str(row.get("status") or "")
    if state in {
        "succeeded",
        "failed",
        "cancelled",
        "timed_out",
        "submission_unknown",
        "usage_unknown",
    }:
        return state
    observed = build_scheduler().refresh_and_settle(
        ComputeJob(
            backend_kind="colab_gpu",
            backend_job_id=backend_job_id,
            idempotency_key=str(row["idempotency_key"]),
            status=state if state in ACTIVE_JOB_STATES else "running",
            heartbeat_at=_iso(row.get("heartbeat_at")),
            failure_reason=row.get("failure_reason"),
        ),
        requirements=tuple(
            str(value)
            for value in _load_json_list(row.get("artifact_requirements_json"))
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
        "colab_restart_quarantined": 0,
        "colab_admission_rebound": 0,
        "colab_uncertain_quarantined": 0,
        "orphan_gpu_reservations_released": 0,
        "legacy_terminal_attempts_imported": 0,
        "terminal_attempts_settled": 0,
        "terminal_colab_attempts_settled": 0,
    }
    terminal_job_ids = GrantGPUUsageControl().reconcile_terminal_attempts()
    totals["terminal_attempts_settled"] = len(terminal_job_ids)
    totals["legacy_terminal_attempts_imported"] = (
        GrantGPUUsageControl().import_legacy_terminal_attempts()
    )
    totals["orphan_gpu_reservations_released"] = (
        GrantGPUUsageControl().release_orphaned_reservations()
    )
    colab_recovery = ColabWorkRepository().reconcile_on_startup()
    totals["colab_restart_quarantined"] = colab_recovery[
        "running_quarantined"
    ]
    totals["colab_admission_rebound"] = colab_recovery["admission_rebound"]
    totals["colab_uncertain_quarantined"] = colab_recovery[
        "uncertain_quarantined"
    ]
    terminal_colab_request_ids = (
        GrantGPUUsageControl().reconcile_terminal_colab_attempts()
    )
    totals["terminal_colab_attempts_settled"] = len(
        terminal_colab_request_ids
    )
    for request_id in terminal_colab_request_ids:
        settle_colab_request(request_id)
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
              AND backend_kind IN ('cpu', 'local_gpu', 'ssh_gpu', 'colab_gpu')
              AND status IN ('submitted', 'running', 'cancel_requested',
                             'collecting')
            ORDER BY id
            """,
            (agenda_id,),
        )
        for row in live:
            try:
                if str(row["backend_job_id"]).startswith("cpu-experiment-run:"):
                    result = settle_cpu_run(
                        _parse_cpu_job_id(str(row["backend_job_id"]))
                    )
                elif str(row["backend_job_id"]).startswith("legacy-gpu-job:"):
                    result = settle_legacy_job(
                        _parse_backend_job_id(str(row["backend_job_id"]))
                    )
                elif str(row["backend_job_id"]).startswith("colab-work-request:"):
                    result = settle_colab_request(
                        int(str(row["backend_job_id"]).split(":", 1)[1])
                    )
                else:
                    raise ComputeBackendError(
                        "managed compute backend job id is invalid"
                    )
                if result in {"succeeded", "failed", "cancelled", "timed_out"}:
                    totals["terminal_settled"] += 1
            except Exception:
                db.rollback()
                totals["settlement_errors"] += 1
    return totals
