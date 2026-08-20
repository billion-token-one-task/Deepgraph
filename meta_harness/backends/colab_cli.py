"""Hardened multi-account Colab CLI executor.

This is a semantic port of production snapshot 7d0b42a's Colab lifecycle:
new -> upload -> exec -> collect -> stop. It is intentionally not wired into
startup code; an operator must configure and validate it in isolated canary.
"""

from __future__ import annotations

import base64
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from contracts.meta_harness import ResourceGrant
from db import database as db
from meta_harness.compute import ColabAccount, ComputeBackendError
from meta_harness.grants import ResourceRequest, authorize


_SENTINEL = "__DEEPGRAPH_COLAB_RETURN_CODE__:"
# Long sessions lose their VM the moment compute ends (runs 163 and 166 both
# finished with returncode 0 and then found /content empty on download), so
# the archive also rides the exec stdout stream -- the one channel that
# demonstrably survives -- as base64, and collection falls back to it.
_EMBED_BEGIN = "__DEEPGRAPH_COLAB_ARTIFACT_B64_BEGIN__:"
_EMBED_END = "__DEEPGRAPH_COLAB_ARTIFACT_B64_END__"
_EMBED_MAX_BYTES = 25 * 1024 * 1024
_SESSION_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")
# Scaffold: a provisioning refusal is a quota-window verdict, not a blip.
_PROVISION_FAILURES_BEFORE_COOLDOWN = 2
_PROVISION_REFUSAL_MARKER = "colab provision failed"
# Stages whose single job is an hour or more of irreplaceable work.
# What every lane is assumed to carry. A hosted notebook session is recreated
# from scratch each time, so anything installed into one does not survive to
# the next -- only the dedicated host can hold a pre-provisioned runtime.
# Work declaring anything beyond this base therefore has exactly one lane that
# can host it, and sending it elsewhere is a lottery: idea 175 declared POT,
# drew colab-pro-2 and then colab-pro, and lost a run to exit 78 each time
# while aws-g5-1 sat ready (2026-08-20).
_BASE_RUNTIME = frozenset(
    {"torch", "transformers", "datasets", "accelerate", "numpy", "pip",
     "setuptools", "wheel", "huggingface-hub", "tokenizers", "safetensors"}
)


def _requires_provisioned_runtime(code_dir: Path) -> bool:
    """Does this work declare a dependency only a provisioned lane will have?"""
    try:
        text = (Path(code_dir) / "requirements.txt").read_text(encoding="utf-8")
    except Exception:
        return False
    for line in text.splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        for separator in ("==", ">=", "<=", "~=", ">", "<", "["):
            name = name.split(separator)[0]
        name = name.strip().lower().replace("_", "-")
        if name and name not in _BASE_RUNTIME:
            return True
    return False


_LONG_RUNNING_STAGES = frozenset({"full_benchmark", "evidence_audit", "validation"})
_PROVISION_COOLDOWN_SECONDS = 3600


class ColabCLIError(ComputeBackendError):
    pass


class ColabCapacityUnavailable(ColabCLIError):
    """No lane is free right now -- a condition of the pool, not of the work.

    Every lane busy, or every lane cooling off after provisioning refusals,
    used to surface as a bare ColabCLIError. The worker treats any exception
    as losing control of its claim, so it failed the request and requeued it
    immediately, which failed again against the same full pool: request 65
    burned ten attempts in three minutes on 2026-08-19 and run 189's audit
    could not proceed. Waiting for capacity must cost the request nothing.
    """


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
            # Decode throughput knob. This allowlist silently dropped it, so
            # runs 159/160 executed at the materialized default of batch 8
            # (~4 GPU-hours of work) and were both killed at their 2-hour
            # caps (2026-08-18).
            "DEEPGRAPH_RUNNER_BATCH_SIZE",
            "DEEPGRAPH_RUNNER_EXAMPLE_OFFSET",
            "DEEPGRAPH_RUNNER_MAX_SEEDS",
            "PYTHONUNBUFFERED",
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
import json, os, pathlib, shutil, subprocess, sys, tarfile

root = pathlib.Path("/content/code")
root.mkdir(parents=True, exist_ok=True)
with tarfile.open("/content/code.tar.gz") as archive:
    archive.extractall(root, filter="data")

requirements_file = root / "requirements.txt"
if requirements_file.exists():
    # The rule is that this runtime installs nothing; it is not that a bundle
    # may not declare what it needs. A materialized runner bundle always ships
    # a requirements.txt, so refusing on the file's presence rejected runtimes
    # that already satisfied every line of it. Verify instead, and keep the
    # refusal for anything genuinely missing -- the remote still never installs.
    import importlib.metadata
    import importlib.util

    # Ask the installer what is installed, rather than guessing an import
    # name from a distribution name. A hand-maintained map of the two is
    # always one package behind: it held five entries, and POT -- whose
    # module is "ot" -- was reported missing on 2026-08-20 immediately after
    # it had been installed successfully, costing idea 157 three runs
    # (193/194/196). Pillow/PIL, scikit-learn/sklearn and opencv-python/cv2
    # are the same shape. importlib.metadata normalises case and separators,
    # so "POT", "pot" and "p-o-t" all resolve to the same distribution.
    # find_spec stays as the fallback for a module present without
    # distribution metadata (a vendored or stdlib-adjacent import).
    missing = []
    for line in requirements_file.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        for separator in ("==", ">=", "<=", "~=", ">", "<", "["):
            name = name.split(separator)[0]
        name = name.strip()
        if not name:
            continue
        try:
            importlib.metadata.distribution(name)
            continue
        except Exception:
            pass
        if importlib.util.find_spec(name.replace("-", "_")) is None:
            missing.append(name)
    if missing:
        print(
            "dependency_install_blocked: reviewed runtime must be pre-provisioned; "
            "missing " + ", ".join(sorted(missing))
        )
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
# The portable runner executes under /content/code but deliberately writes its
# durable outputs to ../results.  Stage those outputs back under the isolated
# code root before archiving, rather than asking the request to authorize a
# traversal path.  A runner that writes in-place still works unchanged.
for relative in {json.dumps(list(artifacts), ensure_ascii=False)}:
    destination = root / relative
    for candidate in (destination, root.parent / "results" / relative):
        if candidate.is_file():
            if candidate != destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, destination)
            break
with tarfile.open(artifact_archive, "w:gz") as archive:
    for relative in {json.dumps(list(artifacts), ensure_ascii=False)}:
        path = root / relative
        if path.exists():
            archive.add(path, arcname=relative)
if artifact_archive.is_file() and artifact_archive.stat().st_size <= {_EMBED_MAX_BYTES}:
    import base64 as _b64
    _payload = _b64.b64encode(artifact_archive.read_bytes()).decode()
    print("\\n{_EMBED_BEGIN}" + str(len(_payload)))
    for _start in range(0, len(_payload), 65536):
        print(_payload[_start:_start + 65536])
    print("{_EMBED_END}")
print("\\n{_SENTINEL}" + str(process.returncode))
"""


def _strip_embedded_archive(stdout: str) -> tuple[str, bytes | None]:
    """Split the base64 artifact payload out of the exec stream.

    Returns the stdout with the payload removed (so tails and hashes stay
    readable) plus the decoded archive bytes, or None when absent/corrupt.
    """
    begin = stdout.rfind(_EMBED_BEGIN)
    if begin < 0:
        return stdout, None
    after = stdout[begin + len(_EMBED_BEGIN):]
    head, sep, rest = after.partition("\n")
    end = rest.find(_EMBED_END)
    if not sep or end < 0:
        return stdout, None
    payload = "".join(rest[:end].split())
    cleaned = (stdout[:begin] + rest[end + len(_EMBED_END):]).rstrip()
    try:
        expected = int(head.strip() or "0")
    except ValueError:
        return cleaned, None
    if expected and len(payload) != expected:
        return cleaned, None
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception:
        return cleaned, None
    return cleaned, data or None


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
        self._provision_failures: dict[str, int] = {}
        self._blocked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def _eligible_locked(
        self,
        requested_hours: float,
        stage: str | None,
        require_dedicated: bool = False,
    ) -> list[ColabAccount]:
        """Lanes that could take this work right now. Caller holds the lock."""
        now = time.monotonic()
        eligible = [
            account
            for account in self._accounts
            if self._used_hours[account.account_ref] + requested_hours
            <= account.quota_gpu_hours
            and self._active[account.account_ref] == 0
            and self._blocked_until.get(account.account_ref, 0.0) <= now
            and not durable_provision_cooldown(account.account_ref)
        ]
        # Measured 2026-08-19: hosted notebook sessions lose their VM on
        # roughly half of the hour-plus jobs (requests 28, 38, 46), while
        # the dedicated host finished every one. A full benchmark or an
        # audit holdout is an hour of work whose loss costs a whole run,
        # so those stages take a dedicated host whenever one is free and
        # fall back to a notebook lane only when none is.
        dedicated = [
            account
            for account in eligible
            if getattr(account, "transport", "colab") == "ssh"
        ]
        if require_dedicated:
            # A hard requirement, not a preference. Falling back to a lane
            # that provably cannot host the work trades a wait for a certain
            # failure: idea 175 drew colab-pro-2 and then colab-pro for a
            # package only the dedicated host carries, and lost a run each
            # time (2026-08-20). Waiting costs minutes; the fallback costs a
            # run and a grant.
            return dedicated
        if stage in _LONG_RUNNING_STAGES and dedicated:
            eligible = dedicated
        return eligible

    def has_capacity(
        self, requested_hours: float = 0.0, *, stage: str | None = None
    ) -> bool:
        """Could any lane take this work right now, without claiming one?

        Probed before a request is claimed. Claiming first and discovering the
        pool is full made a temporary capacity condition look like a failure
        of the work: request 65 burned ten attempts in three minutes against a
        full pool on 2026-08-19 and run 189's audit could not proceed.
        """
        with self._lock:
            return bool(self._eligible_locked(requested_hours, stage))

    def acquire(
        self,
        requested_hours: float,
        *,
        stage: str | None = None,
        require_dedicated: bool = False,
    ) -> ColabAccount:
        with self._lock:
            now = time.monotonic()
            eligible = self._eligible_locked(requested_hours, stage, require_dedicated)
            if not eligible:
                # A cooling account is capacity that exists but is unusable
                # right now; say so distinctly from a genuinely full pool.
                if any(
                    self._blocked_until.get(account.account_ref, 0.0) > now
                    or durable_provision_cooldown(account.account_ref)
                    for account in self._accounts
                ):
                    raise ColabCapacityUnavailable(
                        "every Colab account is cooling off after provisioning failures"
                    )
                raise ColabCapacityUnavailable(
                    "no Colab account has isolated quota capacity"
                )
            account = min(
                eligible,
                key=lambda item: (
                    # Faster, already-paid-for hardware goes first; the
                    # measured A10G lane runs a pilot in 8 minutes against
                    # the T4 lanes' 33-54 (2026-08-19).
                    int(getattr(item, "priority", 100)),
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

    def record_provision_failure(self, account: ColabAccount) -> None:
        """Cool an account off after repeated provisioning refusals.

        Colab hands out TooManyAssignmentsError for the rest of a quota
        window; colab-pro-2 burned four candidate launches in 1.3s each on
        2026-08-19 because nothing remembered that. Consecutive failures
        park the account; any success clears the count.
        """
        with self._lock:
            ref = account.account_ref
            self._provision_failures[ref] = self._provision_failures.get(ref, 0) + 1
            if self._provision_failures[ref] >= _PROVISION_FAILURES_BEFORE_COOLDOWN:
                self._blocked_until[ref] = (
                    time.monotonic() + _PROVISION_COOLDOWN_SECONDS
                )

    def record_provision_success(self, account: ColabAccount) -> None:
        with self._lock:
            self._provision_failures[account.account_ref] = 0
            self._blocked_until.pop(account.account_ref, None)


_PROVISION_COOLDOWN_CAP_SECONDS = 12 * 3600
_PROVISION_STREAK_LOOKBACK = 12


def durable_provision_cooldown(account_ref: str) -> bool:
    """Is this account cooling off, according to the record that survives?

    The in-memory counter is process state keyed on time.monotonic(), so a web
    restart forgets every refusal. Deploy restarts on 2026-08-19 handed
    colab-pro-2 fresh chances within two minutes of each other and it refused
    each one, and every refusal costs a whole experiment run. The durable
    record of what each lane did already exists in colab_work_requests_v1;
    read it rather than keeping a second, more forgetful copy.

    The wait doubles with each consecutive refusal. A flat hour assumed the
    vendor's quota window was about an hour; colab-pro-2's record says
    otherwise -- ten refusals against one success across a whole day, and
    three refusals in a row (22:25, 22:27, 23:42) each landing immediately
    after the previous cooldown expired, costing one experiment run every
    time. Doubling lets the record tell us how long the window really is
    instead of guessing again.

    Fails open: an unreadable history must never take a lane out of service.
    """
    try:
        rows = db.fetchall(
            """
            SELECT status, failure_reason, created_at
            FROM colab_work_requests_v1
            WHERE account_ref=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (str(account_ref), int(_PROVISION_STREAK_LOOKBACK)),
        )
    except Exception:
        return False

    streak = 0
    newest = None
    for row in rows:
        is_refusal = (
            str(row.get("status")) == "failed"
            and _PROVISION_REFUSAL_MARKER
            in str(row.get("failure_reason") or "").lower()
        )
        if not is_refusal:
            break  # a success, or any other failure, ends the streak
        if newest is None:
            newest = row.get("created_at")
        streak += 1

    if streak < _PROVISION_FAILURES_BEFORE_COOLDOWN:
        return False
    if not isinstance(newest, datetime):
        return False
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=timezone.utc)

    extra = streak - _PROVISION_FAILURES_BEFORE_COOLDOWN
    cooldown = min(
        _PROVISION_COOLDOWN_SECONDS * (2**extra),
        _PROVISION_COOLDOWN_CAP_SECONDS,
    )
    age = (datetime.now(timezone.utc) - newest).total_seconds()
    return age < cooldown


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
        if getattr(account, "transport", "colab") == "ssh":
            return self._run_ssh(account, args, timeout)
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

    def _run_ssh(
        self,
        account: ColabAccount,
        args: Sequence[str],
        timeout: int,
    ) -> subprocess.CompletedProcess:
        """Translate the five transport verbs onto a plain SSH GPU host.

        The remote wrapper is unchanged: /content exists as a real directory
        on the host, so new/upload/exec/download/stop are the only pieces
        that differ from the Colab tunnel.
        """
        opts = [
            "-i", account.ssh_key_path,
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=6",
        ]
        target = account.ssh_target
        verb = str(args[0])

        def _go(cmd: Sequence[str]) -> subprocess.CompletedProcess:
            return self.runner(
                list(cmd),
                timeout=timeout,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

        if verb == "new":
            return _go([
                "ssh", *opts, target,
                "sudo mkdir -p /content && sudo chown $(whoami) /content"
                " && rm -rf /content/* && nvidia-smi -L",
            ])
        if verb == "upload":
            local, remote = str(args[3]), str(args[4])
            return _go(["scp", *opts, local, f"{target}:{remote}"])
        if verb == "exec":
            runner_file = str(args[args.index("--file") + 1])
            exec_timeout = int(args[args.index("--timeout") + 1])
            pushed = _go(
                ["scp", *opts, runner_file, f"{target}:/content/dg-exec-runner.py"]
            )
            if pushed.returncode != 0:
                return pushed
            return _go([
                "ssh", *opts, target,
                f"cd /content && timeout {exec_timeout} "
                "$HOME/dgvenv/bin/python /content/dg-exec-runner.py",
            ])
        if verb == "download":
            remote, local = str(args[3]), str(args[4])
            return _go(["scp", *opts, f"{target}:{remote}", local])
        if verb == "stop":
            return _go([
                "ssh", *opts, target,
                "pkill -f dg-exec-runner.py 2>/dev/null; rm -rf /content/*; true",
            ])
        raise ColabCLIError(f"unsupported ssh transport verb: {verb}")

    def _reap_orphan_sessions(self, account: ColabAccount) -> int:
        """Stop every session on an account the pool says is idle.

        ColabAccountPool.acquire only hands out an account with zero active
        requests, so a session found here belongs to a controller that died
        without stopping it. Those sessions keep consuming the account's GPU
        assignment quota and make every later provision fail immediately.
        """
        if getattr(account, "transport", "colab") == "ssh":
            return 0
        listed = self._run(
            account, ("sessions",), self.config.provision_timeout_seconds
        )
        if listed.returncode != 0:
            return 0
        reaped = 0
        for line in (listed.stdout or "").splitlines():
            match = re.match(r"^\[([^\]]+)\]", line.strip())
            if not match:
                continue
            name = match.group(1).strip()
            if not name or name == "?":
                continue
            stopped = self._run(
                account, ("stop", "-s", name), self.config.stop_timeout_seconds
            )
            if stopped.returncode == 0:
                reaped += 1
        return reaped

    def run_request(
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
        needs_runtime = _requires_provisioned_runtime(code_dir)
        account = self.accounts.acquire(
            requested_hours,
            stage=request.stage,
            require_dedicated=needs_runtime,
        )
        # Routing has been hard to reason about after the fact -- a request
        # that should have taken the dedicated lane took a notebook one and
        # the reason was not recoverable from any record. Say the decision
        # out loud so the next surprise is one log line away.
        print(
            f"[COLAB] route request={request.idempotency_key} "
            f"stage={request.stage} needs_runtime={needs_runtime} "
            f"-> {account.account_ref}",
            flush=True,
        )
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
                    # The pool guarantees one live request per account, so
                    # any session already open on this account is an orphan
                    # from a killed controller -- and Colab counts it against
                    # the account's GPU assignments, which is why provisioning
                    # kept being refused within 1.3 seconds (2026-08-19).
                    # Reap them and try once more before giving up.
                    reaped = self._reap_orphan_sessions(account)
                    if reaped:
                        created = self._run(
                            account,
                            ("new", "-s", session, "--gpu", self.config.gpu_type),
                            self.config.provision_timeout_seconds,
                        )
                if created.returncode != 0:
                    self.accounts.record_provision_failure(account)
                    raise ColabCLIError(
                        "colab provision failed: "
                        + (created.stderr or created.stdout or "")[-400:]
                    )
                self.accounts.record_provision_success(account)
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
                stdout, embedded_archive = _strip_embedded_archive(stdout)
                # Run 163 finished 2.8 hours of decoding with returncode 0
                # and lost everything to one failed download (2026-08-18).
                # The archive already exists remotely; pulling it is the one
                # step retries cannot corrupt.
                download_error = ""
                # A runner that already failed has no artifacts to collect.
                # Trying anyway spent four download retries and then reported
                # "artifact collection failed", which masked a clean
                # dependency_install_blocked (exit 78, request 66 on
                # 2026-08-20) as a TRANSPORT failure. That misclassification
                # is not cosmetic: transport failures draw on the larger
                # infrastructure retry budget, so a deterministic environment
                # mismatch would be retried as though a different lane could
                # fix it. The exit code is the cause; missing artifacts are
                # the consequence.
                collected = returncode == 0 or bool(embedded_archive)
                if collected:
                    for attempt in range(1, 5):
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
                        if downloaded.returncode == 0 and artifact_archive.exists() \
                                and artifact_archive.stat().st_size > 0:
                            break
                        download_error = (
                            downloaded.stderr or downloaded.stdout or "no output"
                        )[-300:]
                        time.sleep(min(10 * attempt, 30))
                    else:
                        if embedded_archive:
                            # The VM (and its tar) is gone, but the archive
                            # also rode the exec stream; restore it from there.
                            artifact_archive.write_bytes(embedded_archive)
                        else:
                            raise ColabCLIError(
                                "Colab artifact collection failed after 4 "
                                "attempts: " + download_error
                            )
                if collected:
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
            gpu_type=getattr(account, "gpu_type", "") or self.config.gpu_type,
            wall_seconds=wall_seconds,
            artifact_manifest=manifest,
            failure_reason=failure_reason,
        )
