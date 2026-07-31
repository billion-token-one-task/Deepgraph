"""ResourceGrant admission checks shared by LLM and compute routes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from contracts.meta_harness import ResourceGrant


class GrantDeniedError(PermissionError):
    pass


@dataclass(frozen=True)
class ResourceRequest:
    agenda_id: int
    idea_id: int
    stage: str
    backend: str
    resource_grant_id: int | None = None
    token_cap: int = 0
    gpu_hours: float = 0.0


def authorize(
    grant: ResourceGrant | None,
    request: ResourceRequest,
    *,
    now: datetime | None = None,
) -> ResourceGrant:
    if grant is None:
        raise GrantDeniedError("resource_grant_required")
    grant.validate()
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expires = datetime.fromisoformat(grant.expires_at.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    blockers: list[str] = []
    if grant.status != "active":
        blockers.append(f"grant_{grant.status}")
    if expires <= current:
        blockers.append("grant_expired")
    if grant.agenda_id != int(request.agenda_id):
        blockers.append("agenda_scope_mismatch")
    if grant.idea_id != int(request.idea_id):
        blockers.append("idea_scope_mismatch")
    if request.resource_grant_id is not None and int(grant.grant_id or 0) != int(
        request.resource_grant_id
    ):
        blockers.append("resource_grant_id_mismatch")
    if grant.stage != request.stage:
        blockers.append("stage_mismatch")
    if request.backend not in grant.backend_allowlist:
        blockers.append("backend_not_allowed")
    if int(request.token_cap) > grant.token_cap:
        blockers.append("token_cap_exceeded")
    if float(request.gpu_hours) > grant.max_gpu_hours:
        blockers.append("gpu_hour_cap_exceeded")
    if blockers:
        raise GrantDeniedError(",".join(blockers))
    return grant
