"""Hash-pinned evaluator execution outside the candidate process boundary."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from meta_harness.harness_evolution import (
    EvaluationRun,
    HarnessCandidate,
    HarnessPolicy,
    HarnessPolicyError,
    validate_candidate,
)


_SUITES = {"held_in", "held_out", "canary"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    if not root.is_dir():
        raise HarnessPolicyError(f"hash-pinned root is not a directory:{root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise HarnessPolicyError(
                f"isolated evaluator input contains a symlink:{relative}"
            )
        if path.is_dir():
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _dedicated_child(value: str, root: str, *, label: str) -> Path:
    path = Path(value).resolve()
    allowed = Path(root).resolve()
    try:
        path.relative_to(allowed)
    except ValueError as exc:
        raise HarnessPolicyError(f"{label} is outside its isolated root") from exc
    if path == allowed:
        raise HarnessPolicyError(f"{label} requires a dedicated child path")
    return path


@dataclass(frozen=True)
class EvaluatorSuiteSpec:
    suite: str
    evaluator_root: str
    evaluator_entrypoint: str
    evaluator_hash: str
    suite_root: str
    suite_hash: str
    output_dir: str
    timeout_seconds: int = 1800
    arguments: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.suite not in _SUITES:
            raise HarnessPolicyError("invalid evaluator suite")
        entrypoint = Path(self.evaluator_entrypoint)
        if entrypoint.is_absolute() or ".." in entrypoint.parts:
            raise HarnessPolicyError(
                "evaluator entrypoint must be relative to evaluator_root"
            )
        if not self.evaluator_hash or not self.suite_hash:
            raise HarnessPolicyError("evaluator and suite hashes are required")
        if self.timeout_seconds <= 0:
            raise HarnessPolicyError("evaluator timeout must be positive")
        if any("\x00" in str(value) for value in self.arguments):
            raise HarnessPolicyError("evaluator arguments contain NUL")


class IsolatedEvaluatorRunner:
    """Run a trusted evaluator with bubblewrap-style mount isolation."""

    def __init__(
        self,
        *,
        policy: HarnessPolicy,
        production_path: str,
        production_database_namespace: str,
        evaluator_root: str,
        holdout_root: str,
        artifact_root: str,
        isolation_binary: str = "bwrap",
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        policy.validate()
        self.policy = policy
        self.production_path = production_path
        self.production_database_namespace = production_database_namespace
        self.evaluator_root = str(Path(evaluator_root).resolve())
        self.holdout_root = str(Path(holdout_root).resolve())
        self.artifact_root = str(Path(artifact_root).resolve())
        self.isolation_binary = isolation_binary
        self.runner = runner

    def _isolation_path(self) -> str:
        value = shutil.which(self.isolation_binary)
        if not value:
            raise HarnessPolicyError(
                "evaluator isolation binary is unavailable; refusing fallback"
            )
        path = Path(value).resolve()
        if not path.is_file():
            raise HarnessPolicyError("evaluator isolation binary is invalid")
        return str(path)

    def _command(
        self,
        *,
        spec: EvaluatorSuiteSpec,
        candidate_path: Path,
        evaluator_path: Path,
        suite_path: Path,
        output_path: Path,
    ) -> list[str]:
        entrypoint = evaluator_path / spec.evaluator_entrypoint
        if not entrypoint.is_file():
            raise HarnessPolicyError("evaluator entrypoint does not exist")
        executable = "/evaluator/" + Path(spec.evaluator_entrypoint).as_posix()
        command = [
            self._isolation_path(),
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--setenv",
            "HOME",
            "/tmp/empty-home",
            "--setenv",
            "DEEPGRAPH_EVALUATION_SUITE",
            spec.suite,
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/tmp/empty-home",
        ]
        for system_root in ("/usr", "/bin", "/lib", "/lib64"):
            if Path(system_root).exists():
                command.extend(("--ro-bind", system_root, system_root))
        command.extend(
            (
                "--ro-bind",
                str(candidate_path),
                "/candidate",
                "--ro-bind",
                str(evaluator_path),
                "/evaluator",
                "--ro-bind",
                str(suite_path),
                "/suite",
                "--bind",
                str(output_path),
                "/output",
                "--chdir",
                "/evaluator",
                executable,
                "--candidate",
                "/candidate",
                "--suite",
                "/suite",
                "--output",
                "/output",
                *tuple(str(value) for value in spec.arguments),
            )
        )
        return command

    def run(
        self,
        *,
        candidate: HarnessCandidate,
        spec: EvaluatorSuiteSpec,
    ) -> EvaluationRun:
        spec.validate()
        validate_candidate(
            candidate,
            policy=self.policy,
            production_path=self.production_path,
            production_database_namespace=self.production_database_namespace,
        )
        candidate_path = Path(candidate.worktree_path).resolve()
        evaluator_path = _dedicated_child(
            spec.evaluator_root,
            self.evaluator_root,
            label="evaluator_root",
        )
        suite_path = _dedicated_child(
            spec.suite_root,
            self.holdout_root,
            label="suite_root",
        )
        output_path = _dedicated_child(
            spec.output_dir,
            self.artifact_root,
            label="evaluator output_dir",
        )
        if output_path.exists() and any(output_path.iterdir()):
            raise HarnessPolicyError("evaluator output_dir must start empty")
        output_path.mkdir(parents=True, exist_ok=True)
        evaluator_hash = _tree_hash(evaluator_path)
        suite_hash = _tree_hash(suite_path)
        if evaluator_hash != spec.evaluator_hash:
            raise HarnessPolicyError("evaluator tree hash does not match pinned hash")
        if suite_hash != spec.suite_hash:
            raise HarnessPolicyError("evaluation suite hash does not match pinned hash")
        candidate_before = _tree_hash(candidate_path)
        failure_reason: str | None = None
        try:
            completed = self.runner(
                self._command(
                    spec=spec,
                    candidate_path=candidate_path,
                    evaluator_path=evaluator_path,
                    suite_path=suite_path,
                    output_path=output_path,
                ),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=spec.timeout_seconds,
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "LANG": "C.UTF-8",
                },
            )
            if completed.returncode != 0:
                failure_reason = f"evaluator_exit_{completed.returncode}"
        except subprocess.TimeoutExpired:
            completed = None
            failure_reason = "evaluator_timeout"
        candidate_after = _tree_hash(candidate_path)
        if candidate_after != candidate_before:
            raise HarnessPolicyError("candidate tree changed during isolated evaluation")
        result_path = output_path / "result.json"
        result: dict = {}
        if result_path.is_file() and not result_path.is_symlink():
            try:
                loaded = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = {}
            if isinstance(loaded, dict):
                result = loaded
        if not result:
            failure_reason = failure_reason or "evaluator_result_missing_or_invalid"
        reported = str(result.get("status") or "").strip().lower()
        if reported not in {"passed", "failed"}:
            failure_reason = failure_reason or "evaluator_status_invalid"
        elif reported == "failed":
            failure_reason = failure_reason or str(
                result.get("failure_reason") or "evaluator_reported_failure"
            )
        status = "passed" if reported == "passed" and failure_reason is None else "failed"
        files: list[dict] = []
        for path in sorted(output_path.rglob("*")):
            if path.is_symlink():
                raise HarnessPolicyError("evaluator output contains a symlink")
            if path.is_file():
                mode = stat.S_IMODE(path.stat().st_mode)
                files.append(
                    {
                        "path": path.relative_to(output_path).as_posix(),
                        "sha256": _sha256_file(path),
                        "size": path.stat().st_size,
                        "mode": f"{mode:04o}",
                    }
                )
        manifest: Mapping[str, object] = {
            "suite": spec.suite,
            "candidate_ref": candidate.candidate_ref,
            "candidate_tree_hash_before": candidate_before,
            "candidate_tree_hash_after": candidate_after,
            "evaluator_hash": evaluator_hash,
            "suite_hash": suite_hash,
            "network": "unshared",
            "candidate_mount": "read_only",
            "files": files,
        }
        return EvaluationRun(
            agenda_id=candidate.agenda_id,
            suite=spec.suite,
            status=status,
            evaluator_ref=(
                f"isolated:{Path(spec.evaluator_root).name}/"
                f"{spec.evaluator_entrypoint}"
            ),
            evaluator_hash=evaluator_hash,
            artifact_manifest=dict(manifest),
            failure_reason=(
                failure_reason
                or (
                    str(result.get("failure_reason"))
                    if result.get("failure_reason")
                    else None
                )
            ),
        )
