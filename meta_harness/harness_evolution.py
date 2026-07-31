"""Isolation and approval policy for self-improving harness candidates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from meta_harness.reviewer_approval import (
    ReviewerApproval,
    ReviewerApprovalVerifier,
    harness_candidate_subject,
)


class HarnessPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class HarnessPolicy:
    version: str = "harness_policy_v1"
    max_modules: int = 2
    max_changed_lines: int = 200
    candidate_root: str = ""
    namespace_prefix: str = "meta_harness_candidate_"
    protected_paths: tuple[str, ...] = (
        "meta_harness/held_out/",
        "meta_harness/evaluators/",
        "meta_harness/safety/",
        "tests/held_out/",
        "tests/canary/",
        "db/migrations/",
        "config/production",
        ".env",
    )
    immutable_policy_paths: tuple[str, ...] = (
        "meta_harness/harness_evolution.py",
        "meta_harness/grants.py",
        "meta_harness/evidence_state.py",
        "contracts/meta_harness.py",
        "contracts/scientific_evidence.py",
        "agents/agenda_repository.py",
    )

    def validate(self) -> None:
        if self.max_modules <= 0 or self.max_changed_lines <= 0:
            raise HarnessPolicyError("diff limits must be positive")
        if not self.candidate_root:
            raise HarnessPolicyError("candidate_root is required")
        if not self.namespace_prefix:
            raise HarnessPolicyError("namespace_prefix is required")


@dataclass(frozen=True)
class HarnessCandidate:
    agenda_id: int
    candidate_ref: str
    base_commit: str
    worktree_path: str
    database_namespace: str
    artifact_namespace: str


@dataclass(frozen=True)
class HarnessPatch:
    agenda_id: int
    candidate_ref: str
    base_commit: str
    changed_paths: tuple[str, ...]
    added_lines: int
    deleted_lines: int
    patch_hash: str

    @property
    def changed_lines(self) -> int:
        return self.added_lines + self.deleted_lines


@dataclass(frozen=True)
class FailureCluster:
    agenda_id: int
    cluster_key: str
    signatures: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    occurrence_count: int

    def validate(self) -> None:
        if self.agenda_id <= 0 or not self.cluster_key:
            raise HarnessPolicyError("FailureCluster requires agenda scope and key")
        if not self.signatures or not self.evidence_refs or self.occurrence_count <= 0:
            raise HarnessPolicyError("FailureCluster requires observed evidence")


@dataclass(frozen=True)
class EvaluationRun:
    agenda_id: int
    suite: str
    status: str
    evaluator_ref: str
    evaluator_hash: str
    artifact_manifest: dict
    failure_reason: str | None = None


@dataclass(frozen=True)
class RegressionReport:
    agenda_id: int
    decision: str
    blockers: tuple[str, ...] = field(default_factory=tuple)
    reviewer: str | None = None
    reviewer_approved: bool = False
    reviewer_approval: ReviewerApproval | None = None


@dataclass(frozen=True)
class HarnessArchive:
    agenda_id: int
    source_commit: str
    source_tree_hash: str
    policy_hash: str
    evaluator_hash: str
    holdout_hash: str


def validate_candidate(
    candidate: HarnessCandidate,
    *,
    policy: HarnessPolicy,
    production_path: str,
    production_database_namespace: str,
) -> None:
    policy.validate()
    if candidate.agenda_id <= 0:
        raise HarnessPolicyError("candidate requires agenda_id")
    if not candidate.candidate_ref or not candidate.base_commit:
        raise HarnessPolicyError("candidate ref and base commit are required")
    worktree = Path(candidate.worktree_path).resolve()
    candidate_root = Path(policy.candidate_root).resolve()
    production = Path(production_path).resolve()
    try:
        worktree.relative_to(candidate_root)
    except ValueError as exc:
        raise HarnessPolicyError("candidate worktree is outside candidate_root") from exc
    if worktree == candidate_root:
        raise HarnessPolicyError("candidate requires a dedicated child worktree")
    if worktree == production or production in worktree.parents or worktree in production.parents:
        raise HarnessPolicyError("candidate worktree overlaps production")
    if not candidate.database_namespace.startswith(policy.namespace_prefix):
        raise HarnessPolicyError("candidate database namespace has the wrong prefix")
    if candidate.database_namespace == production_database_namespace:
        raise HarnessPolicyError("candidate database namespace overlaps production")
    if not candidate.artifact_namespace.startswith(policy.namespace_prefix):
        raise HarnessPolicyError("candidate artifact namespace is not isolated")


def validate_patch(patch: HarnessPatch, *, policy: HarnessPolicy) -> None:
    policy.validate()
    if patch.agenda_id <= 0:
        raise HarnessPolicyError("HarnessPatch requires agenda_id")
    if patch.added_lines < 0 or patch.deleted_lines < 0:
        raise HarnessPolicyError("patch line counts cannot be negative")
    # v1 treats each changed source file as a module. The limit is policy data,
    # not a permanent constant, and can evolve after reviewed calibration.
    if len(set(patch.changed_paths)) > policy.max_modules:
        raise HarnessPolicyError("patch exceeds configured module limit")
    if patch.changed_lines > policy.max_changed_lines:
        raise HarnessPolicyError("patch exceeds configured changed-line limit")
    for path in patch.changed_paths:
        normalized = path.lstrip("./")
        if any(normalized == prefix or normalized.startswith(prefix) for prefix in policy.protected_paths):
            raise HarnessPolicyError(f"candidate modified protected path: {path}")
        if normalized in policy.immutable_policy_paths:
            raise HarnessPolicyError(f"candidate modified immutable policy: {path}")
    digest_material = "\n".join(
        [patch.base_commit, *sorted(patch.changed_paths), str(patch.added_lines), str(patch.deleted_lines)]
    ).encode("utf-8")
    expected = hashlib.sha256(digest_material).hexdigest()
    if patch.patch_hash != expected:
        raise HarnessPolicyError("patch_hash does not match patch metadata")


def evaluate_candidate(
    runs: Iterable[EvaluationRun],
    *,
    candidate_id: int | None = None,
    patch_hash: str = "",
    reviewer_approval: ReviewerApproval | Mapping | None = None,
    approval_verifier: ReviewerApprovalVerifier | None = None,
) -> RegressionReport:
    runs = list(runs)
    agenda_ids = {run.agenda_id for run in runs}
    agenda_id = next(iter(agenda_ids), 0)
    by_suite = {run.suite: run for run in runs}
    required = {"held_in", "held_out", "canary"}
    blockers: list[str] = []
    if agenda_id <= 0:
        blockers.append("agenda_scope_missing")
    if len(agenda_ids) > 1:
        blockers.append("cross_agenda_evaluation")
    missing = required - set(by_suite)
    blockers.extend(f"missing_{suite}" for suite in sorted(missing))
    for suite in sorted(required.intersection(by_suite)):
        run = by_suite[suite]
        if run.status != "passed":
            blockers.append(f"{suite}_{run.status}")
        if not run.evaluator_ref:
            blockers.append(f"{suite}_evaluator_ref_missing")
        if not run.evaluator_hash:
            blockers.append(f"{suite}_evaluator_hash_missing")
        if not run.artifact_manifest:
            blockers.append(f"{suite}_artifacts_missing")
    if blockers:
        return RegressionReport(
            agenda_id=agenda_id,
            decision="reject",
            blockers=tuple(blockers),
        )
    if reviewer_approval is None:
        return RegressionReport(
            agenda_id=agenda_id,
            decision="awaiting_approval",
            blockers=("reviewer_approval_required",),
            reviewer_approved=False,
        )
    if int(candidate_id or 0) <= 0 or not patch_hash.strip():
        raise HarnessPolicyError(
            "signed approval requires candidate_id and patch_hash"
        )
    verifier = approval_verifier or ReviewerApprovalVerifier.from_environment()
    approval = verifier.verify(
        reviewer_approval,
        purpose="harness_upgrade",
        subject=harness_candidate_subject(
            agenda_id=agenda_id,
            candidate_id=int(candidate_id),
            patch_hash=patch_hash,
        ),
    )
    return RegressionReport(
        agenda_id=agenda_id,
        decision="approved",
        reviewer=approval.reviewer_id,
        reviewer_approved=True,
        reviewer_approval=approval,
    )
