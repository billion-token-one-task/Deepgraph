"""Backend-neutral compute admission and lifecycle contracts.

Backends receive an injected transport. This module never starts a local
process, SSH session, GPU task, or Colab session by itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol

from contracts.meta_harness import ResourceGrant
from meta_harness.grants import ResourceRequest, authorize


ACTIVE_JOB_STATES = {"submitted", "running", "cancel_requested"}
TERMINAL_JOB_STATES = {"succeeded", "failed", "cancelled", "timed_out"}
ALL_JOB_STATES = ACTIVE_JOB_STATES | TERMINAL_JOB_STATES
PERSISTENCE_ONLY_JOB_STATES = {
    "submitting",
    "submission_unknown",
    "collecting",
    "usage_unknown",
}


class ComputeBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendCapability:
    backend_kind: str
    available: bool
    cpu_cores: int = 0
    gpu_count: int = 0
    vram_gb: float = 0.0
    accelerator_names: tuple[str, ...] = field(default_factory=tuple)
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComputeSubmission:
    agenda_id: int
    idea_id: int
    stage: str
    resource_grant_id: int
    idempotency_key: str
    command_ref: str
    artifact_namespace: str
    timeout_seconds: int
    requested_gpu_hours: float = 0.0

    def validate(self) -> None:
        if min(self.agenda_id, self.idea_id, self.resource_grant_id) <= 0:
            raise ComputeBackendError("submission scope ids must be positive")
        if not self.stage or not self.idempotency_key or not self.command_ref:
            raise ComputeBackendError("submission metadata is incomplete")
        if not self.artifact_namespace or self.artifact_namespace.startswith("/"):
            raise ComputeBackendError("artifact namespace must be relative and explicit")
        if ".." in self.artifact_namespace.split("/"):
            raise ComputeBackendError("artifact namespace cannot escape its root")
        if self.timeout_seconds <= 0 or self.requested_gpu_hours < 0:
            raise ComputeBackendError("submission limits are invalid")


@dataclass(frozen=True)
class ComputeJob:
    backend_kind: str
    backend_job_id: str
    idempotency_key: str
    status: str
    heartbeat_at: str | None = None
    failure_reason: str | None = None

    def validate(self) -> None:
        if not self.backend_kind or not self.backend_job_id or not self.idempotency_key:
            raise ComputeBackendError("compute job identity is incomplete")
        if self.status not in ALL_JOB_STATES:
            raise ComputeBackendError(f"unknown compute job status: {self.status}")
        if self.status == "failed" and not self.failure_reason:
            raise ComputeBackendError("failed compute jobs require a failure reason")


@dataclass(frozen=True)
class ComputeClaim:
    record_id: int
    is_new: bool
    backend_kind: str
    idempotency_key: str
    status: str
    backend_job_id: str | None = None
    heartbeat_at: str | None = None
    failure_reason: str | None = None

    def existing_job(self) -> ComputeJob:
        if self.is_new:
            raise ComputeBackendError("new compute claim has no existing job")
        if self.status in PERSISTENCE_ONLY_JOB_STATES or not self.backend_job_id:
            raise ComputeBackendError(
                f"submission_reconciliation_required:{self.status}"
            )
        job = ComputeJob(
            backend_kind=self.backend_kind,
            backend_job_id=self.backend_job_id,
            idempotency_key=self.idempotency_key,
            status=self.status,
            heartbeat_at=self.heartbeat_at,
            failure_reason=self.failure_reason,
        )
        job.validate()
        return job


@dataclass(frozen=True)
class ArtifactCollection:
    manifest: Mapping[str, Any]
    complete: bool
    missing_requirements: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class UsageAccounting:
    wall_seconds: float
    gpu_hours: float
    cpu_core_hours: float
    backend_report: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if min(self.wall_seconds, self.gpu_hours, self.cpu_core_hours) < 0:
            raise ComputeBackendError("compute usage cannot be negative")


class BackendTransport(Protocol):
    def capability(self) -> BackendCapability: ...

    def submit(self, request: ComputeSubmission) -> ComputeJob: ...

    def status(self, backend_job_id: str) -> ComputeJob: ...

    def heartbeat(self, backend_job_id: str) -> ComputeJob: ...

    def cancel(self, backend_job_id: str) -> ComputeJob: ...

    def collect_artifacts(
        self, backend_job_id: str, requirements: tuple[str, ...]
    ) -> ArtifactCollection: ...

    def usage(self, backend_job_id: str) -> UsageAccounting: ...


class ComputeJobStore(Protocol):
    def claim(
        self,
        request: ComputeSubmission,
        *,
        backend_kind: str,
    ) -> ComputeClaim: ...

    def bind_submitted_job(self, record_id: int, job: ComputeJob) -> None: ...

    def mark_submission_unknown(self, record_id: int, *, reason: str) -> None: ...

    def record_id_for_job(self, job: ComputeJob) -> int: ...

    def record_backend_state(self, job: ComputeJob) -> str: ...

    def finalize_terminal(
        self, job: ComputeJob, *, usage: UsageAccounting
    ) -> None: ...

    def finalize_success(
        self,
        record_id: int,
        *,
        artifacts: ArtifactCollection,
        usage: UsageAccounting,
    ) -> None: ...


class ComputeBackend(ABC):
    kind: str

    @abstractmethod
    def capability(self) -> BackendCapability:
        raise NotImplementedError

    @abstractmethod
    def submit(self, request: ComputeSubmission) -> ComputeJob:
        raise NotImplementedError

    @abstractmethod
    def status(self, backend_job_id: str) -> ComputeJob:
        raise NotImplementedError

    @abstractmethod
    def heartbeat(self, backend_job_id: str) -> ComputeJob:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, backend_job_id: str) -> ComputeJob:
        raise NotImplementedError

    @abstractmethod
    def collect_artifacts(
        self, backend_job_id: str, requirements: tuple[str, ...]
    ) -> ArtifactCollection:
        raise NotImplementedError

    @abstractmethod
    def usage(self, backend_job_id: str) -> UsageAccounting:
        raise NotImplementedError


class _TransportBackend(ComputeBackend):
    kind = ""

    def __init__(self, transport: BackendTransport):
        self._transport = transport

    def _checked_job(self, job: ComputeJob) -> ComputeJob:
        job.validate()
        if job.backend_kind != self.kind:
            raise ComputeBackendError("transport returned the wrong backend kind")
        return job

    def capability(self) -> BackendCapability:
        value = self._transport.capability()
        if value.backend_kind != self.kind:
            raise ComputeBackendError("capability kind mismatch")
        return value

    def submit(self, request: ComputeSubmission) -> ComputeJob:
        request.validate()
        job = self._checked_job(self._transport.submit(request))
        if job.status not in {"submitted", "running"}:
            raise ComputeBackendError(
                f"backend submission did not create a live job: {job.status}"
            )
        return job

    def status(self, backend_job_id: str) -> ComputeJob:
        return self._checked_job(self._transport.status(backend_job_id))

    def heartbeat(self, backend_job_id: str) -> ComputeJob:
        return self._checked_job(self._transport.heartbeat(backend_job_id))

    def cancel(self, backend_job_id: str) -> ComputeJob:
        return self._checked_job(self._transport.cancel(backend_job_id))

    def collect_artifacts(
        self, backend_job_id: str, requirements: tuple[str, ...]
    ) -> ArtifactCollection:
        return self._transport.collect_artifacts(backend_job_id, requirements)

    def usage(self, backend_job_id: str) -> UsageAccounting:
        usage = self._transport.usage(backend_job_id)
        usage.validate()
        return usage


class CPUBackend(_TransportBackend):
    kind = "cpu"


class LocalGPUBackend(_TransportBackend):
    kind = "local_gpu"


@dataclass(frozen=True)
class SSHGPUConfig:
    target_ref: str
    credential_ref: str
    artifact_root: str

    def validate(self) -> None:
        if not self.target_ref or not self.credential_ref or not self.artifact_root:
            raise ComputeBackendError("SSH backend requires reference-only configuration")
        if any(marker in self.credential_ref.lower() for marker in ("password=", "token=", "key=")):
            raise ComputeBackendError("SSH credential_ref must not contain credential material")


class SSHGPUBackend(_TransportBackend):
    kind = "ssh_gpu"

    def __init__(self, transport: BackendTransport, config: SSHGPUConfig):
        config.validate()
        super().__init__(transport)
        self.config = config


@dataclass(frozen=True)
class ColabAccount:
    account_ref: str
    credential_ref: str
    isolated_home: str
    oauth_store: str
    session_namespace: str
    quota_gpu_hours: float

    def validate(self) -> None:
        if not all(
            (
                self.account_ref,
                self.credential_ref,
                self.isolated_home,
                self.oauth_store,
                self.session_namespace,
            )
        ):
            raise ComputeBackendError("Colab account isolation fields are required")
        if self.quota_gpu_hours <= 0:
            raise ComputeBackendError("Colab quota must be positive")
        if any(marker in self.credential_ref.lower() for marker in ("token=", "secret=", "cookie=")):
            raise ComputeBackendError("Colab credential_ref must be a secret reference")


class ColabGPUBackend(_TransportBackend):
    kind = "colab_gpu"

    def __init__(self, transport: BackendTransport, accounts: list[ColabAccount]):
        if not accounts:
            raise ComputeBackendError("Colab backend requires configured accounts")
        for account in accounts:
            account.validate()
        for attribute in ("account_ref", "isolated_home", "oauth_store", "session_namespace"):
            values = [getattr(account, attribute) for account in accounts]
            if len(values) != len(set(values)):
                raise ComputeBackendError(f"Colab accounts must have unique {attribute}")
        super().__init__(transport)
        self.accounts = tuple(accounts)


class ComputeScheduler:
    """Selects from a registry; backend-specific conditions stay in adapters."""

    def __init__(
        self,
        backends: list[ComputeBackend],
        *,
        job_store: ComputeJobStore | None = None,
        allow_ephemeral_idempotency: bool = False,
    ):
        self._backends = {backend.kind: backend for backend in backends}
        if len(self._backends) != len(backends):
            raise ComputeBackendError("duplicate ComputeBackend kind")
        self._job_store = job_store
        self._allow_ephemeral_idempotency = bool(allow_ephemeral_idempotency)
        self._idempotent_jobs: dict[tuple[str, str], ComputeJob] = {}

    def capabilities(self) -> dict[str, BackendCapability]:
        return {
            kind: backend.capability()
            for kind, backend in sorted(self._backends.items())
        }

    def submit(
        self,
        request: ComputeSubmission,
        *,
        grant: ResourceGrant | None,
        preferred_backends: list[str],
    ) -> ComputeJob:
        request.validate()
        if not preferred_backends:
            raise ComputeBackendError("preferred_backends cannot be empty")
        if self._job_store is None and not self._allow_ephemeral_idempotency:
            raise ComputeBackendError(
                "durable_compute_job_store_required; "
                "ephemeral idempotency is test-only"
            )
        failures: list[str] = []
        for kind in preferred_backends:
            backend = self._backends.get(kind)
            if backend is None:
                failures.append(f"{kind}:not_configured")
                continue
            authorize(
                grant,
                ResourceRequest(
                    agenda_id=request.agenda_id,
                    idea_id=request.idea_id,
                    stage=request.stage,
                    backend=kind,
                    resource_grant_id=request.resource_grant_id,
                    gpu_hours=request.requested_gpu_hours,
                ),
            )
            capability = backend.capability()
            if not capability.available:
                failures.append(f"{kind}:unavailable")
                continue
            key = (kind, request.idempotency_key)
            claim: ComputeClaim | None = None
            if self._job_store is not None:
                claim = self._job_store.claim(request, backend_kind=kind)
                if not claim.is_new:
                    existing = claim.existing_job()
                    if existing.status in ACTIVE_JOB_STATES:
                        return existing
                    raise ComputeBackendError(
                        f"idempotency_key_already_terminal:{existing.status}"
                    )
            elif key in self._idempotent_jobs:
                return self._idempotent_jobs[key]
            try:
                job = backend.submit(request)
            except Exception as exc:
                if claim is not None:
                    self._job_store.mark_submission_unknown(
                        claim.record_id,
                        reason=f"{kind}:{type(exc).__name__}",
                    )
                    raise ComputeBackendError(
                        "backend_submission_outcome_unknown;"
                        "manual_reconciliation_required"
                    ) from exc
                failures.append(f"{kind}:{type(exc).__name__}")
                continue
            if claim is not None:
                self._job_store.bind_submitted_job(claim.record_id, job)
            else:
                self._idempotent_jobs[key] = job
            return job
        raise ComputeBackendError(
            "no_allowed_backend_available:" + ",".join(failures)
        )

    def collect_if_successful(
        self,
        job: ComputeJob,
        *,
        requirements: tuple[str, ...],
    ) -> ArtifactCollection:
        job.validate()
        if job.status != "succeeded":
            raise ComputeBackendError(
                f"artifacts cannot certify non-successful job: {job.status}"
            )
        backend = self._backends.get(job.backend_kind)
        if backend is None:
            raise ComputeBackendError("backend no longer configured")
        artifacts = backend.collect_artifacts(job.backend_job_id, requirements)
        if not artifacts.complete or artifacts.missing_requirements:
            raise ComputeBackendError("required artifacts are incomplete")
        return artifacts

    def refresh_and_settle(
        self,
        job: ComputeJob,
        *,
        requirements: tuple[str, ...],
    ) -> ComputeJob:
        """Poll once and persist metered terminal truth.

        This method never maps a failed backend to success.  A backend usage
        failure leaves the durable job non-terminal for reconciliation.
        """
        if self._job_store is None:
            raise ComputeBackendError("durable_compute_job_store_required")
        backend = self._backends.get(job.backend_kind)
        if backend is None:
            raise ComputeBackendError("backend no longer configured")
        observed = backend.status(job.backend_job_id)
        observed.validate()
        if observed.status == "succeeded":
            state = self._job_store.record_backend_state(observed)
            if state != "collecting":
                raise ComputeBackendError(
                    f"successful backend did not enter artifact collection:{state}"
                )
            artifacts = backend.collect_artifacts(
                observed.backend_job_id, requirements
            )
            usage = backend.usage(observed.backend_job_id)
            usage.validate()
            record_id = self._job_store.record_id_for_job(observed)
            if not artifacts.complete or artifacts.missing_requirements:
                missing = ",".join(artifacts.missing_requirements)
                self._job_store.finalize_terminal(
                    ComputeJob(
                        backend_kind=observed.backend_kind,
                        backend_job_id=observed.backend_job_id,
                        idempotency_key=observed.idempotency_key,
                        status="failed",
                        heartbeat_at=observed.heartbeat_at,
                        failure_reason=(
                            "required_artifacts_incomplete:"
                            + (missing or "unknown")
                        ),
                    ),
                    usage=usage,
                )
                raise ComputeBackendError("required artifacts are incomplete")
            if int(record_id or 0) <= 0:
                raise ComputeBackendError("durable compute record was not found")
            self._job_store.finalize_success(
                int(record_id),
                artifacts=artifacts,
                usage=usage,
            )
            return observed
        if observed.status in {"failed", "cancelled", "timed_out"}:
            usage = backend.usage(observed.backend_job_id)
            usage.validate()
            self._job_store.finalize_terminal(observed, usage=usage)
            return observed
        self._job_store.record_backend_state(observed)
        return observed


def heartbeat_is_stale(value: str | None, *, timeout_seconds: int) -> bool:
    if not value or timeout_seconds <= 0:
        return True
    heartbeat = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if heartbeat.tzinfo is None:
        return True
    return (datetime.now(timezone.utc) - heartbeat.astimezone(timezone.utc)).total_seconds() > timeout_seconds
