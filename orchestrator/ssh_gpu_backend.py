"""SSH-backed remote GPU execution for experiment runs."""

from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from config import (
    GPU_REMOTE_AUTO_PIP_INSTALL,
    GPU_REMOTE_SETUP_TIMEOUT_SECONDS,
    GPU_REMOTE_SSH_PASSWORD,
)


def _load_metadata(worker: Mapping[str, Any] | None) -> dict[str, Any]:
    if not worker:
        return {}
    raw = worker.get("metadata")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def is_ssh_worker(worker: Mapping[str, Any] | None) -> bool:
    return _load_metadata(worker).get("backend") == "ssh"


def _ssh_password(worker: Mapping[str, Any] | None) -> str:
    metadata = _load_metadata(worker)
    return str(metadata.get("ssh_password") or GPU_REMOTE_SSH_PASSWORD or "")


def _ssh_target(worker: Mapping[str, Any]) -> str:
    metadata = _load_metadata(worker)
    user = str(metadata.get("ssh_user") or "").strip()
    host = str(metadata.get("ssh_host") or "").strip()
    if not user or not host:
        raise RuntimeError("SSH GPU worker is missing ssh_user or ssh_host metadata.")
    return f"{user}@{host}"


def _ssh_base_command(worker: Mapping[str, Any]) -> list[str]:
    metadata = _load_metadata(worker)
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
    if _ssh_password(worker):
        cmd.extend(["-o", "PubkeyAuthentication=no"])
    port = int(metadata.get("ssh_port") or 22)
    cmd.extend(["-p", str(port), _ssh_target(worker)])
    return cmd


def _rsync_ssh_command(worker: Mapping[str, Any]) -> str:
    metadata = _load_metadata(worker)
    parts = ["ssh", "-o", "StrictHostKeyChecking=no"]
    if _ssh_password(worker):
        parts.extend(["-o", "PubkeyAuthentication=no"])
    parts.extend(["-p", str(int(metadata.get("ssh_port") or 22))])
    return " ".join(parts)


def _with_askpass_env(worker: Mapping[str, Any]) -> tuple[dict[str, str], str | None]:
    env = os.environ.copy()
    password = _ssh_password(worker)
    if not password:
        return env, None
    fd, askpass_path = tempfile.mkstemp(prefix="deepgraph-ssh-askpass-", suffix=".sh")
    os.close(fd)
    Path(askpass_path).write_text("#!/bin/sh\nprintf '%s\\n' \"$DEEPGRAPH_SSH_PASSWORD\"\n", encoding="utf-8")
    os.chmod(askpass_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    env["DEEPGRAPH_SSH_PASSWORD"] = password
    env["SSH_ASKPASS"] = askpass_path
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env["DISPLAY"] = env.get("DISPLAY") or ":0"
    return env, askpass_path


def _run_subprocess(cmd: list[str], *, worker: Mapping[str, Any], timeout: int | None = None, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    env, askpass_path = _with_askpass_env(worker)
    try:
        return subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            env=env,
        )
    finally:
        if askpass_path:
            try:
                os.remove(askpass_path)
            except OSError:
                pass


def _run_ssh(worker: Mapping[str, Any], remote_script: str, *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    cmd = _ssh_base_command(worker) + [f"bash -lc {shlex.quote(remote_script)}"]
    return _run_subprocess(cmd, worker=worker, timeout=timeout)


def _ensure_remote_directory(worker: Mapping[str, Any], remote_dir: str) -> None:
    result = _run_ssh(worker, f"mkdir -p {shlex.quote(remote_dir)}", timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"failed to create remote dir {remote_dir}")


def _rsync(worker: Mapping[str, Any], source: str, dest: str, *, delete: bool = False, timeout: int | None = None) -> None:
    cmd = [
        "rsync",
        "-az",
        "--partial",
        "-e",
        _rsync_ssh_command(worker),
    ]
    if delete:
        cmd.append("--delete")
    cmd.extend([source, dest])
    result = _run_subprocess(cmd, worker=worker, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "rsync failed")


def sync_workdir_to_remote(*, worker: Mapping[str, Any], local_workdir: Path, remote_workdir: str) -> None:
    _ensure_remote_directory(worker, remote_workdir)
    _rsync(worker, f"{local_workdir}/", f"{_ssh_target(worker)}:{remote_workdir}/", delete=True, timeout=600)


def sync_workdir_from_remote(*, worker: Mapping[str, Any], remote_workdir: str, local_workdir: Path) -> None:
    local_workdir.mkdir(parents=True, exist_ok=True)
    _rsync(worker, f"{_ssh_target(worker)}:{remote_workdir}/", f"{local_workdir}/", delete=False, timeout=600)


def _remote_python(worker: Mapping[str, Any], command_tokens: list[str], local_python: str) -> list[str]:
    metadata = _load_metadata(worker)
    remote_python = str(metadata.get("python_bin") or "python").strip() or "python"
    if not command_tokens:
        return []
    rewritten = list(command_tokens)
    first = rewritten[0]
    if first == local_python or first.endswith("/python") or first.endswith("/python3") or os.path.basename(first).startswith("python"):
        rewritten[0] = remote_python
    return rewritten


def _remote_paths(worker: Mapping[str, Any], run_id: int, local_workdir: Path) -> tuple[str, str]:
    metadata = _load_metadata(worker)
    base_dir = str(metadata.get("remote_base_dir") or "/root/deepgraph-remote-worker").rstrip("/")
    remote_workdir = f"{base_dir}/runs/run_{run_id}"
    remote_code_dir = f"{remote_workdir}/{local_workdir.joinpath('code').relative_to(local_workdir).as_posix()}"
    return remote_workdir, remote_code_dir


def _install_remote_repo_deps(*, worker: Mapping[str, Any], remote_code_dir: str, remote_python: str) -> None:
    """Best-effort install of cloned repo dependencies on the SSH worker before training."""
    if not GPU_REMOTE_AUTO_PIP_INSTALL:
        return
    script = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(remote_code_dir)}",
            "export PIP_DISABLE_PIP_VERSION_CHECK=1",
            "export PIP_DEFAULT_TIMEOUT=120",
            f"{shlex.quote(remote_python)} -m pip install -U pip setuptools wheel",
            (
                f"if [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ]; then "
                f"echo '[deepgraph-remote] pip install -e .'; "
                f"{shlex.quote(remote_python)} -m pip install -e . || "
                f"{shlex.quote(remote_python)} -m pip install .; "
                f"elif [ -f requirements.txt ]; then "
                f"echo '[deepgraph-remote] pip install -r requirements.txt'; "
                f"{shlex.quote(remote_python)} -m pip install -r requirements.txt; "
                f"else echo '[deepgraph-remote] skip auto pip (no pyproject/setup/requirements)'; fi"
            ),
        ]
    )
    result = _run_ssh(worker, script, timeout=max(120, int(GPU_REMOTE_SETUP_TIMEOUT_SECONDS)))
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip() or "remote dependency install failed"
        raise RuntimeError(msg)


def run_remote_experiment(
    *,
    worker: Mapping[str, Any],
    run_id: int,
    local_workdir: Path,
    local_code_dir: Path,
    time_budget: int,
    command_tokens: list[str],
    local_python: str,
) -> dict[str, Any]:
    metadata = _load_metadata(worker)
    remote_workdir, remote_code_dir = _remote_paths(worker, run_id, local_workdir)
    sync_workdir_to_remote(worker=worker, local_workdir=local_workdir, remote_workdir=remote_workdir)

    visible_device = str(metadata.get("visible_device") or "0")
    remote_tokens = _remote_python(worker, command_tokens, local_python)
    if not remote_tokens:
        raise RuntimeError("remote experiment received an empty command")

    _install_remote_repo_deps(worker=worker, remote_code_dir=remote_code_dir, remote_python=remote_tokens[0])

    remote_lines = [
        "set -euo pipefail",
        f"cd {shlex.quote(remote_code_dir)}",
        "export PYTHONUNBUFFERED=1",
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(visible_device)}",
        'echo "REMOTE_EXECUTOR=ssh_gpu_backend"',
        'echo "REMOTE_HOST=$(hostname)"',
        f'echo "CUDA_VISIBLE_DEVICES={visible_device}"',
        f'if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --id={shlex.quote(visible_device)} --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true; fi',
        f"timeout {max(1, int(time_budget) + 60)}s {shlex.join(remote_tokens)}",
    ]
    result = _run_ssh(worker, "\n".join(remote_lines), timeout=max(120, int(time_budget) + 180))
    sync_workdir_from_remote(worker=worker, remote_workdir=remote_workdir, local_workdir=local_workdir)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "remote_host": str(metadata.get("ssh_host") or ""),
        "remote_workdir": remote_workdir,
        "backend": "ssh",
        "worker_id": worker.get("id"),
        "visible_device": visible_device,
    }
