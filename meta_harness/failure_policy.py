"""Stable failure reason classification and generic recovery policy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


TRANSPORT_CLASS_MARKERS = (
    "transport:",
    "controller_lost",
    "admission_abandoned",
    "colab provision failed",
)


# A lane that lacks a declared dependency cannot run this work at all. That is
# a property of the LANE, not of the experiment: request 97 was refused on
# colab-pro-2 for a package that had been pre-provisioned on aws-g5-1, where
# the six requests either side of it all succeeded (2026-08-20). Charging it
# to the science budget punishes a candidate for where it happened to land.
LANE_MISMATCH_MARKERS = (
    "dependency_install_blocked",
    "experiment_exit_78",
)


def is_lane_mismatch_failure(reason: object) -> bool:
    """True when the lane could not host the work, whatever the work was."""
    text = str(reason or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in LANE_MISMATCH_MARKERS)


def measured_nothing(reason: object) -> bool:
    """True when a failure evaluated no examples, so it judges no science.

    Transport deaths and lane mismatches are different causes with the same
    evidential content: nothing was measured, so nothing was learned about the
    hypothesis. Both must be kept out of the science retry budget, and both
    still terminate through their own bounded infrastructure budgets.
    """
    return is_transport_class_failure(reason) or is_lane_mismatch_failure(reason)


def is_transport_class_failure(reason: object) -> bool:
    """True when a flight died before it could measure anything.

    A refused provision, a recycled VM, an abandoned admission -- none of
    these evaluated an example, so they say nothing about the science and must
    not spend a science retry budget. Callers pair this with a separate,
    larger infrastructure budget that still terminates.
    """
    text = str(reason or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in TRANSPORT_CLASS_MARKERS)


REASON_CODES = {
    "dataset_unavailable",
    "repository_id_malformed",
    "model_download_timeout",
    "authentication_required",
    "runner_contract_violation",
    "cuda_oom",
    "metric_missing",
    "grant_gpu_hours_exhausted",
    "controller_lost",
    "scientific_negative_result",
    "dataset_schema_mismatch",
    "dependency_missing",
    "network_transient",
    "artifact_hash_mismatch",
    "artifact_contract_violation",
    "artifact_hashes_missing",
    "gpu_environment_missing",
    "label_fallback_forbidden",
    "metric_contract_unsupported",
    "metric_direction_invalid",
    "metric_non_finite",
    "metric_recomputation_mismatch",
    "sample_count_missing",
    "seed_contract_violation",
    # A run that reports no significance test cannot reach a supported
    # verdict, so the runner contract refuses it up front rather than
    # letting it spend a full budget first.
    "p_value_missing",
    "p_value_invalid",
    "permutation_contract_violation",
    # A result that measured nothing is a broken instrument, not a
    # scientific negative. Run 153 reported exact_match 0.0 against 0.0
    # from 24 truncated generations and was filed as a refutation.
    "metric_degenerate",
    "generation_truncated",
    "unknown_execution_failure",
}

RUNNER_CONTRACT_CODES = {
    "runner_contract_violation",
    "artifact_contract_violation",
    "artifact_hashes_missing",
    "artifact_hash_mismatch",
    "gpu_environment_missing",
    "label_fallback_forbidden",
    "metric_contract_unsupported",
    "metric_direction_invalid",
    "metric_non_finite",
    "metric_recomputation_mismatch",
    "sample_count_missing",
    "seed_contract_violation",
    # These are code defects in the runner -- it produced predictions but no
    # statistic -- so they belong on the repair path. Left unregistered they
    # would degrade to unknown_execution_failure and get a bare defer, the
    # exact trap the classify_failure comment above documents.
    "p_value_missing",
    "p_value_invalid",
    "permutation_contract_violation",
    "metric_degenerate",
    "generation_truncated",
}


@dataclass(frozen=True)
class FailureContext:
    reason_code: str
    detail: str
    code_hash: str
    environment_hash: str
    remaining_gpu_seconds: float
    retry_count: int = 0
    requirements: Mapping[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        # Diagnostic prose often contains timestamps, process IDs, transient
        # URLs, or retry counters.  The no-repeat identity is deliberately the
        # stable failure class plus identical code and environment.
        payload = {
            "reason_code": self.reason_code,
            "code_hash": self.code_hash,
            "environment_hash": self.environment_hash,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    retryable: bool
    invoke_llm_repair: bool
    reason_code: str
    backoff_seconds: int = 0
    adjustments: Mapping[str, Any] = field(default_factory=dict)


def classify_failure(
    *,
    message: str,
    returncode: int | None = None,
    final_results_present: bool = False,
) -> str:
    text = str(message or "").lower()
    # Ahead of the embedded-reason-code scan below, because the remote's own
    # self-report is what is wrong here: the HF client prints "You are sending
    # unauthenticated requests to the HF Hub" directly above the real error,
    # the runner stamped reason_code=authentication_required into its output,
    # and that code routes to 'defer' -- exactly wrong for a plan defect no
    # credential can fix. Idea 131 lost three grants to "gsm8k" instead of
    # "openai/gsm8k" (2026-08-19).
    if (
        "repository id must be" in text
        or "repo id must be" in text
        or "invalid hf uri" in text
    ):
        return "repository_id_malformed"
    for code in sorted(REASON_CODES):
        if code in text:
            return code
    if "cuda out of memory" in text or "cublas_status_alloc_failed" in text:
        return "cuda_oom"
    if "401" in text or "403" in text or "authentication" in text or "gated repo" in text:
        return "authentication_required"
    if "dataset" in text and any(token in text for token in ("not found", "unavailable", "404")):
        return "dataset_unavailable"
    if "schema" in text and any(token in text for token in ("field", "column", "feature")):
        return "dataset_schema_mismatch"
    if "model" in text and "timeout" in text:
        return "model_download_timeout"
    if any(token in text for token in ("connection reset", "temporary failure", "timed out", "timeout")):
        return "network_transient"
    if "no module named" in text or "modulenotfounderror" in text:
        return "dependency_missing"
    if "grant_gpu_hours_exhausted" in text:
        return "grant_gpu_hours_exhausted"
    if "controller_lost" in text or "connection_lost_after_submit" in text:
        return "controller_lost"
    if "artifact_hash_mismatch" in text:
        return "artifact_hash_mismatch"
    # Contract checks must precede the "no final_results" fallback. A run that
    # violates the runner or candidate-adapter contract dies before it can
    # write final_results, so testing for their absence first classified every
    # such failure as metric_missing -- which decide_recovery routes to a bare
    # defer, while runner_contract_violation routes to repair_code. That made
    # the repair path unreachable for exactly the failures a code repair is
    # meant to fix, and left granted candidates parked forever.
    if (
        "runner_contract" in text
        or "final_results" in text
        or "contracterror" in text
        or "candidate_adapter" in text
        or "candidate_hook" in text
        or "capability_scaffold" in text
    ):
        return "runner_contract_violation"
    if returncode in (0, None) and not final_results_present:
        return "metric_missing"
    return "unknown_execution_failure"


def decide_recovery(
    context: FailureContext,
    *,
    fingerprint_seen: bool,
) -> RecoveryDecision:
    reason = context.reason_code
    if reason not in REASON_CODES:
        reason = "unknown_execution_failure"
    if reason == "scientific_negative_result":
        return RecoveryDecision("record_outcome", False, False, reason)
    if reason in {"grant_gpu_hours_exhausted", "authentication_required"}:
        return RecoveryDecision("defer", False, False, reason)
    if reason == "repository_id_malformed":
        # No retry can fix a plan that names a dataset that cannot exist; the
        # candidate has to be repaired or abandoned, never re-flown.
        return RecoveryDecision("repair_plan", False, False, reason)
    if reason in {"network_transient", "model_download_timeout"}:
        retryable = context.retry_count < 3 and context.remaining_gpu_seconds > 30
        return RecoveryDecision(
            "retry_with_backoff" if retryable else "defer",
            retryable,
            False,
            reason,
            backoff_seconds=min(300, 15 * (2**context.retry_count)) if retryable else 0,
        )
    if fingerprint_seen:
        return RecoveryDecision("defer_duplicate_fingerprint", False, False, reason)
    if reason == "cuda_oom":
        retryable = context.retry_count < 2 and context.remaining_gpu_seconds > 30
        return RecoveryDecision(
            "retry_adjusted" if retryable else "defer",
            retryable,
            False,
            reason,
            adjustments={
                "batch_size": 1,
                "prefer_quantized": True,
                "allow_smaller_compatible_model": True,
            }
            if retryable
            else {},
        )
    if reason == "dataset_schema_mismatch":
        return RecoveryDecision(
            "select_compatible_adapter",
            True,
            False,
            reason,
            adjustments={"rerun_preflight": True},
        )
    if reason in {
        "dataset_unavailable",
        "dependency_missing",
        "metric_missing",
        "controller_lost",
    }:
        return RecoveryDecision("defer", False, False, reason)
    if reason in RUNNER_CONTRACT_CODES:
        return RecoveryDecision("repair_code", True, True, reason)
    return RecoveryDecision("defer", False, False, reason)
