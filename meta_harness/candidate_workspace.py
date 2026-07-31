"""Creation-time isolation for HarnessCandidate worktrees and processes."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from meta_harness.harness_evolution import (
    HarnessCandidate,
    HarnessPolicy,
    HarnessPolicyError,
    validate_candidate,
)


_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_-]+")
_SECRET_MARKERS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "API_KEY",
    "OAUTH",
    "COOKIE",
    "CREDENTIAL",
    "SSH_",
)


@dataclass(frozen=True)
class CandidateSandbox:
    candidate: HarnessCandidate
    database_url_ref: str
    artifact_root: str
    evaluator_root: str
    holdout_root: str
    policy_root: str

    def candidate_environment(
        self,
        base: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Return a scrubbed environment; protected inputs are path-only, read-only."""
        environment = {}
        for key, value in (base or os.environ).items():
            upper = key.upper()
            if key in {"DEEPGRAPH_DATABASE_URL", "DATABASE_URL", "HOME"}:
                continue
            if any(marker in upper for marker in _SECRET_MARKERS):
                continue
            environment[key] = str(value)
        environment.update(
            {
                "DEEPGRAPH_CANDIDATE_MODE": "1",
                "DEEPGRAPH_CANDIDATE_REF": self.candidate.candidate_ref,
                "DEEPGRAPH_CANDIDATE_WORKTREE": self.candidate.worktree_path,
                "DEEPGRAPH_CANDIDATE_DATABASE_NAMESPACE": self.candidate.database_namespace,
                "DEEPGRAPH_CANDIDATE_DATABASE_URL_REF": self.database_url_ref,
                "DEEPGRAPH_CANDIDATE_ARTIFACT_ROOT": self.artifact_root,
                "DEEPGRAPH_EVALUATOR_ROOT": self.evaluator_root,
                "DEEPGRAPH_HOLDOUT_ROOT": self.holdout_root,
                "DEEPGRAPH_POLICY_ROOT": self.policy_root,
                "DEEPGRAPH_PROTECTED_INPUTS_READ_ONLY": "1",
            }
        )
        return environment


class CandidateWorkspaceManager:
    def __init__(
        self,
        *,
        repository_path: str,
        production_path: str,
        production_database_namespace: str,
        policy: HarnessPolicy,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        self.repository_path = str(Path(repository_path).resolve())
        self.production_path = str(Path(production_path).resolve())
        self.production_database_namespace = production_database_namespace
        self.policy = policy
        self.runner = runner
        policy.validate()

    def create(
        self,
        *,
        agenda_id: int,
        base_commit: str,
        candidate_name: str,
    ) -> HarnessCandidate:
        if not re.fullmatch(r"[0-9a-f]{40}", base_commit):
            raise HarnessPolicyError("candidate base_commit must be a full object hash")
        safe_name = _SAFE_NAME.sub("-", candidate_name).strip("-")[:48]
        if not safe_name:
            raise HarnessPolicyError("candidate name is empty after normalization")
        candidate_ref = f"meta-harness/{safe_name}"
        worktree = Path(self.policy.candidate_root).resolve() / safe_name
        if worktree.exists():
            raise HarnessPolicyError("candidate worktree path already exists")
        candidate = HarnessCandidate(
            agenda_id=agenda_id,
            candidate_ref=candidate_ref,
            base_commit=base_commit,
            worktree_path=str(worktree),
            database_namespace=f"{self.policy.namespace_prefix}{safe_name}",
            artifact_namespace=f"{self.policy.namespace_prefix}{safe_name}/artifacts",
        )
        validate_candidate(
            candidate,
            policy=self.policy,
            production_path=self.production_path,
            production_database_namespace=self.production_database_namespace,
        )
        worktree.parent.mkdir(parents=True, exist_ok=True)
        completed = self.runner(
            [
                "git",
                "-C",
                self.repository_path,
                "worktree",
                "add",
                "--detach",
                str(worktree),
                base_commit,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            raise HarnessPolicyError(
                "candidate worktree creation failed: "
                + (completed.stderr or completed.stdout or "")[-500:]
            )
        return candidate
