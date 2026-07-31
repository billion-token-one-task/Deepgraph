"""Authenticated, purpose-bound reviewer approvals.

The service verifies detached HMAC signatures using secret references supplied
through the environment.  It never stores the signing secret or raw signature.
HMAC is the v1 mechanism; a secret-manager-backed asymmetric verifier can
implement the same envelope later without changing repository contracts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


class ReviewerApprovalError(PermissionError):
    pass


def _utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ReviewerApprovalError("approval issued_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ReviewerApprovalError("approval issued_at must include a timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class ReviewerApproval:
    reviewer_id: str
    key_id: str
    purpose: str
    subject: str
    issued_at: str
    signature: str

    @classmethod
    def from_value(
        cls, value: "ReviewerApproval | Mapping[str, Any] | None"
    ) -> "ReviewerApproval":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ReviewerApprovalError("signed reviewer approval is required")
        return cls(
            reviewer_id=str(value.get("reviewer_id") or ""),
            key_id=str(value.get("key_id") or ""),
            purpose=str(value.get("purpose") or ""),
            subject=str(value.get("subject") or ""),
            issued_at=str(value.get("issued_at") or ""),
            signature=str(value.get("signature") or ""),
        )

    def signing_payload(self) -> bytes:
        values = {
            "issued_at": self.issued_at,
            "key_id": self.key_id,
            "purpose": self.purpose,
            "reviewer_id": self.reviewer_id,
            "subject": self.subject,
        }
        if any(not value.strip() for value in values.values()):
            raise ReviewerApprovalError("approval envelope is incomplete")
        _utc(self.issued_at)
        return json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def signature_hash(self) -> str:
        if not self.signature.strip():
            raise ReviewerApprovalError("approval signature is missing")
        return hashlib.sha256(self.signature.encode("utf-8")).hexdigest()

    def public_record(self) -> dict[str, str]:
        return {
            "reviewer_id": self.reviewer_id,
            "key_id": self.key_id,
            "purpose": self.purpose,
            "subject": self.subject,
            "issued_at": self.issued_at,
            "signature_hash": self.signature_hash(),
        }


class ReviewerApprovalVerifier:
    """Verify against a key-id -> ``env:NAME`` manifest."""

    def __init__(
        self,
        key_references: Mapping[str, str],
        *,
        max_age_seconds: int = 604800,
        future_skew_seconds: int = 300,
    ):
        self._key_references = {
            str(key): str(value) for key, value in key_references.items()
        }
        self._max_age = int(max_age_seconds)
        self._future_skew = int(future_skew_seconds)
        if self._max_age <= 0 or self._future_skew < 0:
            raise ReviewerApprovalError("approval time policy is invalid")

    @classmethod
    def from_environment(cls) -> "ReviewerApprovalVerifier":
        raw = os.getenv("DEEPGRAPH_REVIEWER_APPROVAL_KEYS_JSON", "").strip()
        if not raw:
            raise ReviewerApprovalError(
                "reviewer approval key manifest is not configured"
            )
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ReviewerApprovalError(
                "reviewer approval key manifest is invalid JSON"
            ) from exc
        if not isinstance(manifest, dict) or not manifest:
            raise ReviewerApprovalError(
                "reviewer approval key manifest must be a non-empty object"
            )
        max_age = int(
            os.getenv("DEEPGRAPH_REVIEWER_APPROVAL_MAX_AGE_SECONDS", "604800")
        )
        return cls(manifest, max_age_seconds=max_age)

    def _secret(self, key_id: str) -> bytes:
        reference = self._key_references.get(key_id, "")
        if not reference.startswith("env:"):
            raise ReviewerApprovalError(
                "reviewer key must be an environment secret reference"
            )
        env_name = reference[4:].strip()
        if not env_name or not env_name.replace("_", "").isalnum():
            raise ReviewerApprovalError("reviewer secret environment name is invalid")
        secret = os.getenv(env_name, "")
        if not secret:
            raise ReviewerApprovalError("reviewer approval secret is unavailable")
        return secret.encode("utf-8")

    def verify(
        self,
        approval: ReviewerApproval | Mapping[str, Any] | None,
        *,
        purpose: str,
        subject: str,
        now: datetime | None = None,
    ) -> ReviewerApproval:
        envelope = ReviewerApproval.from_value(approval)
        if envelope.purpose != purpose or envelope.subject != subject:
            raise ReviewerApprovalError("approval purpose or subject mismatch")
        issued_at = _utc(envelope.issued_at)
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if issued_at > current + timedelta(seconds=self._future_skew):
            raise ReviewerApprovalError("approval timestamp is in the future")
        if current - issued_at > timedelta(seconds=self._max_age):
            raise ReviewerApprovalError("reviewer approval has expired")
        expected = hmac.new(
            self._secret(envelope.key_id),
            envelope.signing_payload(),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, envelope.signature):
            raise ReviewerApprovalError("reviewer approval signature is invalid")
        return envelope


def scientific_manuscript_subject(
    *, agenda_id: int, experiment_run_id: int, verdict_hash: str
) -> str:
    return (
        f"scientific-manuscript:{int(agenda_id)}:{int(experiment_run_id)}:"
        f"{str(verdict_hash).strip().lower()}"
    )


def harness_candidate_subject(
    *, agenda_id: int, candidate_id: int, patch_hash: str
) -> str:
    return (
        f"harness-candidate:{int(agenda_id)}:{int(candidate_id)}:"
        f"{str(patch_hash).strip().lower()}"
    )
