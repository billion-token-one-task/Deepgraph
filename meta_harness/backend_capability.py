"""What each compute backend can actually do, stated explicitly.

The host has no GPU, and configuration currently carries both a legacy
``DEEPGRAPH_GPU_BACKEND=colab`` field and the current meta-harness SSH routing
fields (``cpu,ssh_gpu``, ``DEEPGRAPH_GPU_MODE=ssh``, credential references),
while the Colab account manifest is absent. Neither SSH GPU nor Colab is
therefore proven usable.

This module refuses to guess. A backend is:

``enabled``
    listed as enabled, fully configured, *and* recorded as verified by an
    operator after a real canary;
``unknown``
    listed and configured, but never verified -- usable only for a canary that
    is separately authorized, never for ordinary scheduling;
``disabled``
    not listed, missing configuration, or contradicted by the host.

There is no fallback path: an unavailable backend never silently becomes
another backend, and a legacy field never enables anything on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


STATE_ENABLED = "enabled"
STATE_UNKNOWN = "unknown"
STATE_DISABLED = "disabled"

KNOWN_BACKENDS = ("cpu", "local_gpu", "ssh_gpu", "colab_gpu")
GPU_BACKENDS = ("local_gpu", "ssh_gpu", "colab_gpu")


class BackendCapabilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendCapabilityReport:
    kind: str
    state: str
    reasons: tuple[str, ...] = ()
    # Names of secret/environment references only. Never a credential value.
    secret_refs: tuple[str, ...] = ()
    verification_required: tuple[str, ...] = field(default_factory=tuple)

    @property
    def usable_for_scheduling(self) -> bool:
        return self.state == STATE_ENABLED

    @property
    def usable_for_canary(self) -> bool:
        return self.state in {STATE_ENABLED, STATE_UNKNOWN}

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "state": self.state,
            "reasons": list(self.reasons),
            "secret_refs": list(self.secret_refs),
            "verification_required": list(self.verification_required),
            "usable_for_scheduling": self.usable_for_scheduling,
            "usable_for_canary": self.usable_for_canary,
        }


def _normalize(values: Iterable[str] | None) -> set[str]:
    return {
        str(value).strip().lower()
        for value in (values or [])
        if str(value).strip()
    }


def evaluate_backends(
    *,
    enabled: Iterable[str] | None,
    verified: Iterable[str] | None = None,
    gpu_mode: str = "single_host",
    ssh_target_ref: str = "",
    ssh_credential_ref: str = "",
    colab_manifest_ref: str = "",
    colab_binary: str = "",
    local_gpu_present: bool = False,
    legacy_gpu_backend: str = "",
) -> dict[str, BackendCapabilityReport]:
    """Classify every known backend. Unknown configuration is never enabled."""
    enabled_set = _normalize(enabled)
    verified_set = _normalize(verified)
    unknown_kinds = enabled_set - set(KNOWN_BACKENDS)
    if unknown_kinds:
        raise BackendCapabilityError(
            "unknown configured compute backend(s):" + ",".join(sorted(unknown_kinds))
        )
    legacy = str(legacy_gpu_backend or "").strip().lower()
    legacy_map = {"colab": "colab_gpu", "ssh": "ssh_gpu", "local": "local_gpu"}
    legacy_kind = legacy_map.get(legacy, "")

    reports: dict[str, BackendCapabilityReport] = {}
    for kind in KNOWN_BACKENDS:
        reasons: list[str] = []
        secret_refs: list[str] = []
        verification: list[str] = []
        configured = True

        if kind == "cpu":
            pass
        elif kind == "local_gpu":
            if not local_gpu_present:
                configured = False
                reasons.append("no_local_gpu_detected_on_host")
        elif kind == "ssh_gpu":
            if str(gpu_mode).strip().lower() != "ssh":
                configured = False
                reasons.append(f"gpu_mode_is_not_ssh:{gpu_mode}")
            if not ssh_target_ref:
                configured = False
                reasons.append("missing_ssh_target_ref")
            else:
                secret_refs.append("compute_backends.ssh_gpu.target_ref")
            if not ssh_credential_ref:
                configured = False
                reasons.append("missing_ssh_credential_ref")
            else:
                secret_refs.append("compute_backends.ssh_gpu.credential_ref")
            verification.append("ssh_reachability_and_gpu_presence_canary")
        elif kind == "colab_gpu":
            if not colab_manifest_ref:
                configured = False
                reasons.append("colab_accounts_manifest_absent")
            else:
                secret_refs.append("compute_backends.colab_gpu.accounts_manifest_ref")
            if not colab_binary:
                configured = False
                reasons.append("colab_cli_binary_not_configured")
            verification.append("colab_account_and_runtime_canary")

        if legacy_kind == kind and kind not in enabled_set:
            # A legacy field is evidence of intent, never an authorization.
            reasons.append("legacy_gpu_backend_field_conflicts_with_enabled_list")

        if kind not in enabled_set:
            state = STATE_DISABLED
            reasons.insert(0, "not_in_enabled_backend_list")
        elif not configured:
            state = STATE_DISABLED
        elif kind == "cpu" or kind in verified_set:
            state = STATE_ENABLED
        else:
            state = STATE_UNKNOWN
            reasons.append("configured_but_never_verified_by_a_canary")

        reports[kind] = BackendCapabilityReport(
            kind=kind,
            state=state,
            reasons=tuple(reasons),
            secret_refs=tuple(secret_refs),
            verification_required=tuple(verification),
        )
    return reports


def reports_from_config() -> dict[str, BackendCapabilityReport]:
    """Capability view of the live configuration. Read-only, no host calls."""
    import os

    from config import (
        COMPUTE_BACKENDS_ENABLED,
        COMPUTE_COLAB_ACCOUNTS_MANIFEST_REF,
        COMPUTE_COLAB_CLI_BINARY,
        COMPUTE_SSH_CREDENTIAL_REF,
        COMPUTE_SSH_TARGET_REF,
        COMPUTE_VERIFIED_BACKENDS,
        GPU_MODE,
    )

    return evaluate_backends(
        enabled=COMPUTE_BACKENDS_ENABLED,
        verified=COMPUTE_VERIFIED_BACKENDS,
        gpu_mode=GPU_MODE,
        ssh_target_ref=COMPUTE_SSH_TARGET_REF,
        ssh_credential_ref=COMPUTE_SSH_CREDENTIAL_REF,
        colab_manifest_ref=COMPUTE_COLAB_ACCOUNTS_MANIFEST_REF,
        colab_binary=COMPUTE_COLAB_CLI_BINARY,
        local_gpu_present=local_gpu_present(),
        legacy_gpu_backend=os.environ.get("DEEPGRAPH_GPU_BACKEND", ""),
    )


def local_gpu_present() -> bool:
    """Detect a local accelerator without importing a GPU library."""
    import shutil
    import subprocess

    binary = shutil.which("nvidia-smi")
    if not binary:
        return False
    try:
        completed = subprocess.run(
            [binary, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


def require_schedulable(
    kind: str,
    reports: Mapping[str, BackendCapabilityReport],
) -> BackendCapabilityReport:
    """Fail closed unless this exact backend is enabled and verified."""
    report = reports.get(str(kind).strip().lower())
    if report is None:
        raise BackendCapabilityError(f"unknown compute backend:{kind}")
    if not report.usable_for_scheduling:
        raise BackendCapabilityError(
            f"compute backend is not schedulable:{report.kind}:{report.state}:"
            + ",".join(report.reasons)
        )
    return report


def selected_canary_backend(
    reports: Mapping[str, BackendCapabilityReport],
    *,
    requested: str,
) -> BackendCapabilityReport:
    """Exactly one GPU backend may be selected for a canary, and it must be it.

    Ambiguity is refused rather than resolved: if more than one GPU backend is
    canary-eligible, the operator must narrow the enabled list first.
    """
    requested_kind = str(requested).strip().lower()
    if requested_kind not in GPU_BACKENDS:
        raise BackendCapabilityError(
            "a canary target must be a GPU backend:" + ",".join(GPU_BACKENDS)
        )
    eligible = [
        report
        for kind, report in reports.items()
        if kind in GPU_BACKENDS and report.usable_for_canary
    ]
    if not eligible:
        raise BackendCapabilityError("no GPU backend is eligible for a canary")
    if len(eligible) > 1:
        raise BackendCapabilityError(
            "more than one GPU backend is canary-eligible; enable exactly one:"
            + ",".join(sorted(report.kind for report in eligible))
        )
    report = eligible[0]
    if report.kind != requested_kind:
        raise BackendCapabilityError(
            f"requested canary backend {requested_kind} is not the eligible one:"
            f"{report.kind}"
        )
    return report
