"""Isolation and approval policy for self-improving harness candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path



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
class EvaluationRun:
    agenda_id: int
    suite: str
    status: str
    evaluator_ref: str
    evaluator_hash: str
    artifact_manifest: dict
    failure_reason: str | None = None


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
