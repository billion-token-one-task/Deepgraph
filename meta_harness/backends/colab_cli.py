"""Hardened multi-account Colab CLI executor.

This is a semantic port of production snapshot 7d0b42a's Colab lifecycle:
new -> upload -> exec -> collect -> stop. It is intentionally not wired into
startup code; an operator must configure and validate it in isolated canary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from contracts.meta_harness import ResourceGrant
from meta_harness.compute import ColabAccount, ComputeBackendError
from meta_harness.grants import ResourceRequest, authorize


_SENTINEL = "__DEEPGRAPH_COLAB_RETURN_CODE__:"
_SESSION_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


class ColabCLIError(ComputeBackendError):
    pass


@dataclass(frozen=True)
class ColabCLIConfig:
    binary: str
    allowed_code_root: str
    allowed_artifact_root: str
    gpu_type: str = "T4"
    provision_timeout_seconds: int = 300
    upload_timeout_seconds: int = 300
    download_timeout_seconds: int = 300
    exec_buffer_seconds: int = 180
    stop_timeout_seconds: int = 120
    allow_dependency_install: bool = False

    def validate(self) -> None:
        if not self.binary or not self.allowed_code_root or not self.allowed_artifact_root:
            raise ColabCLIError(
                "Colab CLI binary and isolated code/artifact roots are required"
            )
        if min(
            self.provision_timeout_seconds,
            self.upload_timeout_seconds,
            self.download_timeout_seconds,
            self.exec_buffer_seconds,
            self.stop_timeout_seconds,
        ) <= 0:
            raise ColabCLIError("Colab CLI timeouts must be positive")
        if self.allow_dependency_install:
            raise ColabCLIError(
                "meta-harness-v1 forbids implicit dependency installation; "
                "use a reviewed pinned runtime"
            )


@dataclass(frozen=True)
class ColabExecutionRequest:
    agenda_id: int
    idea_id: int
    stage: str
    resource_grant_id: int
    idempotency_key: str
    code_dir: str
    command_tokens: tuple[str, ...]
    environment: Mapping[str, str]
    timeout_seconds: int
    artifact_paths: tuple[str, ...]
    artifact_output_dir: str


@dataclass(frozen=True)
class ColabExecutionResult:
    status: str
    returncode: int | None
    stdout: str
    session: str
    account_ref: str
    gpu_type: str
    wall_seconds: float
    artifact_manifest: Mapping[str, object]
    failure_reason: str | None = None


def _safe_remote_environment(environment: Mapping[str, str]) -> dict[str, str]:
    allowed: dict[str, str] = {}
    for key, value in environment.items():
        if key.startswith(("BENCHMARK_", "DG_PUBLIC_")) or key in {
            "OMP_NUM_THREADS",
            "TOKENIZERS_PARALLELISM",
        }:
            allowed[str(key)] = str(value)
    return allowed


def _safe_relative_paths(values: Sequence[str]) -> tuple[str, ...]:
    paths: list[str] = []
    for raw in values:
        path = Path(str(raw))
        if path.is_absolute() or ".." in path.parts:
            raise ColabCLIError("artifact paths must be relative to code_dir")
        normalized = path.as_posix().lstrip("./")
        if normalized:
            paths.append(normalized)
    if not paths:
        raise ColabCLIError("at least one artifact path is required")
    return tuple(dict.fromkeys(paths))


def _within_root(value: str, root: str, *, label: str) -> Path:
    path = Path(value).resolve()
    allowed = Path(root).resolve()
    try:
        path.relative_to(allowed)
    except ValueError as exc:
        raise ColabCLIError(f"{label} is outside its configured isolated root") from exc
    if path == allowed:
        raise ColabCLIError(f"{label} requires a dedicated child path")
    return path


def _validate_code_tree(code_dir: Path) -> None:
    if not code_dir.is_dir():
        raise ColabCLIError("Colab code_dir does not exist or is not a directory")
    blocked_names = {
        "authorized_keys",
        "credentials.json",
        "token.json",
        "cookies.json",
        "id_rsa",
        "id_ed25519",
    }
    blocked_parts = {"backups", "oauth_home", ".ssh"}
    for path in code_dir.rglob("*"):
        relative = path.relative_to(code_dir)
        lowered_parts = {part.lower() for part in relative.parts}
        name = path.name.lower()
        if path.is_symlink():
            raise ColabCLIError(f"Colab code tree contains a symlink: {relative}")
        if (
            lowered_parts.intersection(blocked_parts)
            or name in blocked_names
            or name.startswith(".env")
            or ".bak-" in name
            or name.endswith((".dump", ".backup"))
        ):
            raise ColabCLIError(
                f"Colab code tree contains a forbidden credential/backup path: {relative}"
            )


def _runner_source(request: ColabExecutionRequest) -> str:
    environment = _safe_remote_environment(request.environment)
    artifacts = _safe_relative_paths(request.artifact_paths)
    return f"""\
import json, os, pathlib, subprocess, sys, tarfile

root = pathlib.Path("/content/code")
root.mkdir(parents=True, exist_ok=True)
with tarfile.open("/content/code.tar.gz") as archive:
    archive.extractall(root, filter="data")

if (root / "requirements.txt").exists():
    print("dependency_install_blocked: reviewed runtime must be pre-provisioned")
    print("{_SENTINEL}78")
    raise SystemExit(0)

environment = dict(os.environ)
environment.update({json.dumps(environment, ensure_ascii=False)})
command = list({json.dumps(list(request.command_tokens), ensure_ascii=False)})
if command and (command[0].endswith(("python", "python3")) or "/python" in command[0]):
    command[0] = sys.executable
process = subprocess.run(
    command,
    cwd=root,
    env=environment,
    capture_output=True,
    text=True,
    timeout={int(request.timeout_seconds)},
)
sys.stdout.write(process.stdout or "")
if process.stderr:
    sys.stdout.write("\\n--- STDERR ---\\n" + process.stderr)

artifact_archive = pathlib.Path("/content/deepgraph-artifacts.tar.gz")
with tarfile.open(artifact_archive, "w:gz") as archive:
    for relative in {json.dumps(list(artifacts), ensure_ascii=False)}:
        path = root / relative
        if path.exists():
            archive.add(path, arcname=relative)
print("\\n{_SENTINEL}" + str(process.returncode))
"""


def _split_result(stdout: str, process_returncode: int) -> tuple[int | None, str]:
    index = stdout.rfind(_SENTINEL)
    if index < 0:
        return None, stdout
    body = stdout[:index].rstrip()
    tail = stdout[index + len(_SENTINEL) :].strip()
    try:
        return int(tail.split()[0]), body
    except (IndexError, ValueError):
        return None, body


class ColabAccountPool:
    """In-process concurrency/quota admission; durable usage comes from OutcomeRecord."""

    def __init__(self, accounts: Sequence[ColabAccount]):
        if not accounts:
            raise ColabCLIError("at least one Colab account is required")
        for account in accounts:
            account.validate()
        for attribute in (
            "account_ref",
            "isolated_home",
            "oauth_store",
            "session_namespace",
        ):
            values = [getattr(account, attribute) for account in accounts]
            if len(values) != len(set(values)):
                raise ColabCLIError(f"Colab accounts must have unique {attribute}")
        self._accounts = tuple(accounts)
        self._active = {account.account_ref: 0 for account in accounts}
        self._used_hours = {account.account_ref: 0.0 for account in accounts}
        self._lock = threading.Lock()

    def acquire(self, requested_hours: float) -> ColabAccount:
        with self._lock:
            eligible = [
                account
                for account in self._accounts
                if self._used_hours[account.account_ref] + requested_hours
                <= account.quota_gpu_hours
                and self._active[account.account_ref] == 0
            ]
            if not eligible:
                raise ColabCLIError("no Colab account has isolated quota capacity")
            account = min(
                eligible,
                key=lambda item: (
                    self._used_hours[item.account_ref],
                    item.account_ref,
                ),
            )
            self._active[account.account_ref] += 1
            return account

    def release(self, account: ColabAccount, used_hours: float) -> None:
        with self._lock:
            self._active[account.account_ref] = max(
                0, self._active[account.account_ref] - 1
            )
            self._used_hours[account.account_ref] += max(0.0, float(used_hours))


class ColabCLIExecutor:
    def __init__(
        self,
        config: ColabCLIConfig,
        accounts: Sequence[ColabAccount],
        *,
        secret_materializer: Callable[[ColabAccount], None],
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        config.validate()
        self.config = config
        self.accounts = ColabAccountPool(accounts)
        self.secret_materializer = secret_materializer
        self.runner = runner

    def _run(
        self,
        account: ColabAccount,
        args: Sequence[str],
        timeout: int,
    ) -> subprocess.CompletedProcess:
        environment = dict(os.environ)
        environment["HOME"] = account.isolated_home
        environment["DEEPGRAPH_COLAB_OAUTH_STORE"] = account.oauth_store
        return self.runner(
            [self.config.binary, *args],
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )

    def execute(
        self,
        request: ColabExecutionRequest,
        *,
        grant: ResourceGrant | None,
    ) -> ColabExecutionResult:
        if request.timeout_seconds <= 0:
            raise ColabCLIError("Colab execution requires a positive timeout")
        requested_hours = request.timeout_seconds / 3600.0
        authorize(
            grant,
            ResourceRequest(
                agenda_id=request.agenda_id,
                idea_id=request.idea_id,
                stage=request.stage,
                backend="colab_gpu",
                resource_grant_id=request.resource_grant_id,
                gpu_hours=requested_hours,
            ),
        )
        session_seed = (
            f"dg-a{request.agenda_id}-i{request.idea_id}-"
            f"{request.idempotency_key[:16]}"
        )
        session = _SESSION_SAFE.sub("-", session_seed).strip("-")[:48]
        code_dir = _within_root(
            request.code_dir,
            self.config.allowed_code_root,
            label="Colab code_dir",
        )
        _validate_code_tree(code_dir)
        output_dir = _within_root(
            request.artifact_output_dir,
            self.config.allowed_artifact_root,
            label="Colab artifact_output_dir",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        account = self.accounts.acquire(requested_hours)
        started = False
        start = time.monotonic()
        returncode: int | None = None
        stdout = ""
        manifest: dict[str, object] = {}
        failure_reason: str | None = None
        try:
            self.secret_materializer(account)
            with tempfile.TemporaryDirectory(prefix="deepgraph-colab-") as temp:
                temp_dir = Path(temp)
                code_archive = temp_dir / "code.tar.gz"
                runner_path = temp_dir / "runner.py"
                artifact_archive = temp_dir / "artifacts.tar.gz"
                with tarfile.open(code_archive, "w:gz") as archive:
                    archive.add(code_dir, arcname=".")
                runner_path.write_text(_runner_source(request), encoding="utf-8")
                created = self._run(
                    account,
                    ("new", "-s", session, "--gpu", self.config.gpu_type),
                    self.config.provision_timeout_seconds,
                )
                if created.returncode != 0:
                    raise ColabCLIError(
                        "colab provision failed: "
                        + (created.stderr or created.stdout or "")[-400:]
                    )
                started = True
                for local, remote in (
                    (code_archive, "/content/code.tar.gz"),
                    (runner_path, "/content/runner.py"),
                ):
                    uploaded = self._run(
                        account,
                        ("upload", "-s", session, str(local), remote),
                        self.config.upload_timeout_seconds,
                    )
                    if uploaded.returncode != 0:
                        raise ColabCLIError(
                            "colab upload failed: "
                            + (uploaded.stderr or uploaded.stdout or "")[-400:]
                        )
                exec_timeout = request.timeout_seconds + self.config.exec_buffer_seconds
                executed = self._run(
                    account,
                    (
                        "exec",
                        "-s",
                        session,
                        "--file",
                        str(runner_path),
                        "--timeout",
                        str(exec_timeout),
                    ),
                    exec_timeout + self.config.stop_timeout_seconds,
                )
                returncode, stdout = _split_result(
                    executed.stdout or "", executed.returncode
                )
                if returncode is None:
                    raise ColabCLIError("Colab output omitted the return-code sentinel")
                downloaded = self._run(
                    account,
                    (
                        "download",
                        "-s",
                        session,
                        "/content/deepgraph-artifacts.tar.gz",
                        str(artifact_archive),
                    ),
                    self.config.download_timeout_seconds,
                )
                if downloaded.returncode != 0 or not artifact_archive.exists():
                    raise ColabCLIError("Colab artifact collection failed")
                with tarfile.open(artifact_archive) as archive:
                    archive.extractall(output_dir, filter="data")
                files = []
                for path in sorted(output_dir.rglob("*")):
                    if path.is_file():
                        files.append(
                            {
                                "path": path.relative_to(output_dir).as_posix(),
                                "size": path.stat().st_size,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            }
                        )
                missing = [
                    relative
                    for relative in _safe_relative_paths(request.artifact_paths)
                    if not (output_dir / relative).exists()
                ]
                manifest = {
                    "account_ref": account.account_ref,
                    "session_namespace": account.session_namespace,
                    "files": files,
                    "missing_requirements": missing,
                    "complete": bool(files) and not missing,
                }
                if returncode != 0:
                    failure_reason = f"experiment_exit_{returncode}"
                elif manifest.get("complete") is not True:
                    failure_reason = "required_artifacts_missing"
        except subprocess.TimeoutExpired:
            failure_reason = "timeout"
        except Exception as exc:
            failure_reason = f"transport:{type(exc).__name__}:{exc}"
        finally:
            if started:
                try:
                    self._run(
                        account,
                        ("stop", "-s", session),
                        self.config.stop_timeout_seconds,
                    )
                except Exception:
                    failure_reason = failure_reason or "session_stop_failed"
            wall_seconds = time.monotonic() - start
            used_hours = wall_seconds / 3600.0
            self.accounts.release(account, used_hours)
        status = (
            "succeeded"
            if returncode == 0 and manifest.get("complete") is True and not failure_reason
            else "timed_out"
            if failure_reason == "timeout"
            else "failed"
        )
        return ColabExecutionResult(
            status=status,
            returncode=returncode,
            stdout=stdout,
            session=session,
            account_ref=account.account_ref,
            gpu_type=self.config.gpu_type,
            wall_seconds=wall_seconds,
            artifact_manifest=manifest,
            failure_reason=failure_reason,
        )
