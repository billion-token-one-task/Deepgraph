"""SSH-backed remote GPU execution for experiment runs."""

from __future__ import annotations

import json
import hashlib
import os
import shlex
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from config import (
    EXPERIMENT_REAL_BENCHMARK_DATASET,
    EXPERIMENT_REAL_BENCHMARK_DATASET_CONFIG,
    EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES,
    EXPERIMENT_REAL_BENCHMARK_SEEDS,
    EXPERIMENT_REAL_LLM_MODEL,
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


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = _first_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in ("hf_model", "model", "name", "id", "dataset", "hf_dataset"):
            text = _first_text(value.get(key))
            if text:
                return text
        return ""
    text = str(value or "").strip()
    return text


def _int_text(value: Any) -> str:
    if isinstance(value, list):
        return str(len(value)) if value else ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ""


def _load_workdir_json(local_workdir: Path, filename: str) -> dict[str, Any]:
    for path in (local_workdir / "spec" / filename, local_workdir / filename):
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                return payload if isinstance(payload, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _default_benchmark_env() -> dict[str, str]:
    return {
        "DEEPGRAPH_BENCHMARK_MODEL": str(EXPERIMENT_REAL_LLM_MODEL),
        "DEEPGRAPH_BENCHMARK_DATASET": str(EXPERIMENT_REAL_BENCHMARK_DATASET),
        "DEEPGRAPH_BENCHMARK_DATASET_CONFIG": str(EXPERIMENT_REAL_BENCHMARK_DATASET_CONFIG),
        "DEEPGRAPH_BENCHMARK_MAX_EXAMPLES": str(EXPERIMENT_REAL_BENCHMARK_MAX_EXAMPLES),
        "DEEPGRAPH_BENCHMARK_SEEDS": str(EXPERIMENT_REAL_BENCHMARK_SEEDS),
    }


def benchmark_env_from_workdir(local_workdir: Path) -> dict[str, str]:
    """Return benchmark env vars aligned with the locked run contract.

    Generated benchmark runners embed a contract in ``train.py`` and the
    workdir spec. The remote launcher must not overwrite that contract with
    process-wide defaults, otherwise evidence may be produced under a different
    model or budget than the manuscript claims.
    """
    env = _default_benchmark_env()
    proxy = _load_workdir_json(local_workdir, "proxy_config.json")
    if not proxy:
        return env

    manifest = proxy.get("benchmark_manifest") if isinstance(proxy.get("benchmark_manifest"), dict) else {}
    full_stage = manifest.get("full_benchmark_stage") if isinstance(manifest.get("full_benchmark_stage"), dict) else {}
    sanity_stage = manifest.get("sanity_stage") if isinstance(manifest.get("sanity_stage"), dict) else {}
    contract = proxy.get("publication_evidence_contract")
    if not isinstance(contract, dict):
        contract = {}

    model = (
        _first_text(full_stage.get("models"))
        or _first_text(contract.get("required_models"))
        or _first_text(manifest.get("models"))
        or _first_text(sanity_stage.get("models"))
        or _first_text(proxy.get("benchmark_model"))
    )
    if model:
        env["DEEPGRAPH_BENCHMARK_MODEL"] = model

    dataset = _first_text(proxy.get("benchmark_dataset"))
    if dataset:
        env["DEEPGRAPH_BENCHMARK_DATASET"] = dataset
    dataset_config = _first_text(proxy.get("benchmark_dataset_config"))
    if dataset_config:
        env["DEEPGRAPH_BENCHMARK_DATASET_CONFIG"] = dataset_config

    max_examples = (
        _int_text(proxy.get("benchmark_max_examples_per_seed"))
        or _int_text(full_stage.get("max_examples_per_seed"))
        or _int_text(full_stage.get("max_eval_examples"))
        or _int_text(sanity_stage.get("max_examples_per_seed"))
    )
    if max_examples:
        env["DEEPGRAPH_BENCHMARK_MAX_EXAMPLES"] = max_examples

    seeds = (
        _int_text(proxy.get("benchmark_seeds"))
        or _int_text(full_stage.get("seeds"))
        or _int_text(contract.get("minimum_seeds"))
        or _int_text(sanity_stage.get("seeds"))
    )
    if seeds:
        env["DEEPGRAPH_BENCHMARK_SEEDS"] = seeds

    return env


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
    suffix = ".cmd" if os.name == "nt" else ".sh"
    fd, askpass_path = tempfile.mkstemp(prefix="deepgraph-ssh-askpass-", suffix=suffix)
    os.close(fd)
    if os.name == "nt":
        Path(askpass_path).write_text("@echo off\r\necho %DEEPGRAPH_SSH_PASSWORD%\r\n", encoding="utf-8")
    else:
        Path(askpass_path).write_text("#!/bin/sh\nprintf '%s\\n' \"$DEEPGRAPH_SSH_PASSWORD\"\n", encoding="utf-8")
        os.chmod(askpass_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    env["DEEPGRAPH_SSH_PASSWORD"] = password
    env["SSH_ASKPASS"] = askpass_path
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env["DISPLAY"] = env.get("DISPLAY") or ":0"
    return env, askpass_path


def _run_subprocess(
    cmd: list[str],
    *,
    worker: Mapping[str, Any],
    timeout: int | None = None,
    capture_output: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env, askpass_path = _with_askpass_env(worker)
    try:
        if input_text is not None:
            proc = subprocess.run(
                cmd,
                input=input_text.encode("utf-8"),
                capture_output=capture_output,
                text=False,
                timeout=timeout,
                env=env,
            )
            stdout = proc.stdout.decode("utf-8", errors="replace") if isinstance(proc.stdout, bytes) else proc.stdout
            stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else proc.stderr
            return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
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
    cmd = _ssh_base_command(worker) + ["bash -s"]
    script = remote_script.replace("\r\n", "\n").replace("\r", "\n")
    return _run_subprocess(cmd, worker=worker, timeout=timeout, input_text=script)


def _scp_target(worker: Mapping[str, Any], remote_path: str) -> str:
    return f"{_ssh_target(worker)}:{remote_path}"


def _scp(worker: Mapping[str, Any], source: str, dest: str, *, timeout: int | None = None) -> None:
    metadata = _load_metadata(worker)
    cmd = ["scp", "-o", "StrictHostKeyChecking=no"]
    if os.name == "nt":
        # Windows OpenSSH SFTP-mode scp can corrupt larger gzip streams with some
        # remote images. Legacy SCP mode preserves the exact bytes in our smoke tests.
        cmd.append("-O")
    if _ssh_password(worker):
        cmd.extend(["-o", "PubkeyAuthentication=no"])
    cmd.extend(["-P", str(int(metadata.get("ssh_port") or 22)), source, dest])
    result = _run_subprocess(cmd, worker=worker, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "scp failed")


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


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if any(part in {".git", "__pycache__", ".mypy_cache", ".pytest_cache"} for part in parts):
        return None
    return info


def _make_tarball(source_dir: Path) -> str:
    fd, tar_path = tempfile.mkstemp(prefix="deepgraph-workdir-", suffix=".tar.gz")
    os.close(fd)
    with tarfile.open(tar_path, "w:gz") as tar:
        for child in source_dir.iterdir():
            tar.add(child, arcname=child.name, filter=_tar_filter)
    return tar_path


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_sha256(worker: Mapping[str, Any], remote_path: str) -> str:
    quoted = shlex.quote(remote_path)
    python_digest = (
        "import hashlib,sys;"
        "h=hashlib.sha256();"
        "f=open(sys.argv[1],'rb');"
        "[h.update(b) for b in iter(lambda:f.read(1048576), b'')];"
        "print(h.hexdigest())"
    )
    script = "\n".join(
        [
            "set -o pipefail",
            f"if [ ! -s {quoted} ]; then ls -l {quoted} >&2; exit 66; fi",
            "if command -v sha256sum >/dev/null 2>&1; then",
            f"  sha256sum {quoted} | awk '{{print $1}}'",
            "elif command -v shasum >/dev/null 2>&1; then",
            f"  shasum -a 256 {quoted} | awk '{{print $1}}'",
            "elif command -v python3 >/dev/null 2>&1; then",
            f"  python3 -c {shlex.quote(python_digest)} {quoted}",
            "elif command -v python >/dev/null 2>&1; then",
            f"  python -c {shlex.quote(python_digest)} {quoted}",
            "else",
            "  echo 'no remote sha256 tool available' >&2",
            "  exit 127",
            "fi",
        ]
    )
    last_msg = ""
    for attempt in range(3):
        result = _run_ssh(worker, script, timeout=120)
        output = (result.stdout or "").strip().split()
        if result.returncode == 0 and output:
            return output[0].strip()
        last_msg = (result.stderr or result.stdout or "").strip()
        if attempt < 2:
            time.sleep(1.0 + attempt)
    raise RuntimeError(last_msg or f"empty checksum output for {remote_path}")


def _safe_extract_tarball(tar_path: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dest_dir / member.name).resolve()
            if target != dest_root and dest_root not in target.parents:
                raise RuntimeError(f"unsafe tar member path: {member.name}")
        tar.extractall(dest_dir)


def _sync_to_remote_without_rsync(*, worker: Mapping[str, Any], local_workdir: Path, remote_workdir: str) -> None:
    parent = str(Path(remote_workdir).parent).replace("\\", "/")
    upload_path = f"{remote_workdir.rstrip('/')}.upload.{os.getpid()}.tar.gz"
    tar_path = _make_tarball(local_workdir)
    try:
        result = _run_ssh(worker, f"mkdir -p {shlex.quote(parent)}", timeout=120)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"failed to create remote parent {parent}")
        expected_sha = _sha256_file(tar_path)
        last_error = ""
        for attempt in range(2):
            _scp(worker, tar_path, _scp_target(worker, upload_path), timeout=600)
            actual_sha = _remote_sha256(worker, upload_path)
            if actual_sha == expected_sha:
                break
            last_error = f"remote upload checksum mismatch on attempt {attempt + 1}: {actual_sha} != {expected_sha}"
        else:
            raise RuntimeError(last_error or "remote upload checksum mismatch")
        script = "\n".join(
            [
                "set -euo pipefail",
                f"rm -rf {shlex.quote(remote_workdir)}",
                f"mkdir -p {shlex.quote(remote_workdir)}",
                f"tar -xzf {shlex.quote(upload_path)} -C {shlex.quote(remote_workdir)}",
                f"rm -f {shlex.quote(upload_path)}",
            ]
        )
        result = _run_ssh(worker, script, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "remote tar extract failed")
    finally:
        try:
            os.remove(tar_path)
        except OSError:
            pass


def _sync_from_remote_without_rsync(*, worker: Mapping[str, Any], remote_workdir: str, local_workdir: Path) -> None:
    remote_tar = f"{remote_workdir.rstrip('/')}.download.{os.getpid()}.tar.gz"
    fd, local_tar = tempfile.mkstemp(prefix="deepgraph-remote-workdir-", suffix=".tar.gz")
    os.close(fd)
    try:
        script = "\n".join(
            [
                "set -euo pipefail",
                f"rm -f {shlex.quote(remote_tar)}",
                f"cd {shlex.quote(remote_workdir)}",
                f"tar -czf {shlex.quote(remote_tar)} .",
            ]
        )
        result = _run_ssh(worker, script, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "remote tar create failed")
        _scp(worker, _scp_target(worker, remote_tar), local_tar, timeout=600)
        _safe_extract_tarball(local_tar, local_workdir)
    finally:
        _run_ssh(worker, f"rm -f {shlex.quote(remote_tar)}", timeout=120)
        try:
            os.remove(local_tar)
        except OSError:
            pass


def sync_workdir_to_remote(*, worker: Mapping[str, Any], local_workdir: Path, remote_workdir: str) -> None:
    if shutil.which("rsync") is None:
        _sync_to_remote_without_rsync(worker=worker, local_workdir=local_workdir, remote_workdir=remote_workdir)
        return
    _ensure_remote_directory(worker, remote_workdir)
    _rsync(worker, f"{local_workdir}/", f"{_ssh_target(worker)}:{remote_workdir}/", delete=True, timeout=600)


def sync_workdir_from_remote(*, worker: Mapping[str, Any], remote_workdir: str, local_workdir: Path) -> None:
    if shutil.which("rsync") is None:
        _sync_from_remote_without_rsync(worker=worker, remote_workdir=remote_workdir, local_workdir=local_workdir)
        return
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


def _remote_run_paths(remote_workdir: str, run_id: int) -> dict[str, str]:
    run_name = f"run_{int(run_id)}"
    return {
        "launcher": f"{remote_workdir.rstrip('/')}/.deepgraph_exec_{run_name}.sh",
        "pid_file": f"{remote_workdir.rstrip('/')}/.deepgraph_{run_name}.pgid",
        "run_log": f"{remote_workdir.rstrip('/')}/run.log",
    }


def _remote_launcher_script(
    *,
    run_id: int,
    remote_code_dir: str,
    remote_workdir: str,
    visible_device: str,
    command_tokens: list[str],
    benchmark_env: Mapping[str, str] | None = None,
) -> str:
    paths = _remote_run_paths(remote_workdir, run_id)
    command_line = shlex.join(command_tokens)
    wrapped_command = f"exec {command_line}"
    set_id_command = f"bash -lc {shlex.quote(wrapped_command)}"
    env_exports = benchmark_env or _default_benchmark_env()
    benchmark_lines = [
        f"export {key}={shlex.quote(str(value))}"
        for key, value in sorted(env_exports.items())
        if key.startswith("DEEPGRAPH_BENCHMARK_") and str(value).strip()
    ]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"pid_file={shlex.quote(paths['pid_file'])}",
            f"remote_log={shlex.quote(paths['run_log'])}",
            f"cd {shlex.quote(remote_code_dir)}",
            "mkdir -p \"$(dirname \"$remote_log\")\"",
            ": > \"$remote_log\"",
            "exec > >(tee -a \"$remote_log\") 2>&1",
            "export PYTHONUNBUFFERED=1",
            f"export CUDA_VISIBLE_DEVICES={shlex.quote(visible_device)}",
            f"export DEEPGRAPH_RUN_ID={shlex.quote(str(int(run_id)))}",
            f"export DEEPGRAPH_REMOTE_WORKDIR={shlex.quote(remote_workdir)}",
            *benchmark_lines,
            'echo "REMOTE_EXECUTOR=ssh_gpu_backend"',
            'echo "REMOTE_HOST=$(hostname)"',
            f'echo "CUDA_VISIBLE_DEVICES={visible_device}"',
            (
                f"if command -v nvidia-smi >/dev/null 2>&1; then "
                f"nvidia-smi --id={shlex.quote(visible_device)} "
                f"--query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true; fi"
            ),
            "cleanup() {",
            "    status=$?",
            "    trap - TERM INT HUP",
            "    if [ -f \"$pid_file\" ]; then",
            "        pgid=$(tr -dc '0-9' < \"$pid_file\" || true)",
            "        if [ -n \"$pgid\" ]; then",
            "            kill -TERM -- \"-$pgid\" >/dev/null 2>&1 || kill -TERM \"$pgid\" >/dev/null 2>&1 || true",
            "            sleep 2",
            "            kill -KILL -- \"-$pgid\" >/dev/null 2>&1 || kill -KILL \"$pgid\" >/dev/null 2>&1 || true",
            "        fi",
            "        rm -f \"$pid_file\"",
            "    fi",
            "    exit \"$status\"",
            "}",
            "trap cleanup TERM INT HUP",
            f"if command -v setsid >/dev/null 2>&1; then setsid {set_id_command} & else {set_id_command} & fi",
            "child=$!",
            "echo \"$child\" > \"$pid_file\"",
            "set +e",
            "wait \"$child\"",
            "status=$?",
            "set -e",
            "rm -f \"$pid_file\"",
            "trap - TERM INT HUP",
            "exit \"$status\"",
        ]
    )


def cleanup_remote_run_processes(*, worker: Mapping[str, Any], run_id: int, remote_workdir: str) -> None:
    """Best-effort cleanup for one remote run without touching other runs."""
    paths = _remote_run_paths(remote_workdir, run_id)
    script = "\n".join(
        [
            "set +e",
            f"pid_file={shlex.quote(paths['pid_file'])}",
            f"launcher={shlex.quote(paths['launcher'])}",
            "if [ -f \"$pid_file\" ]; then",
            "    pgid=$(tr -dc '0-9' < \"$pid_file\" || true)",
            "    if [ -n \"$pgid\" ]; then",
            "        kill -TERM -- \"-$pgid\" >/dev/null 2>&1 || kill -TERM \"$pgid\" >/dev/null 2>&1 || true",
            "        sleep 2",
            "        kill -KILL -- \"-$pgid\" >/dev/null 2>&1 || kill -KILL \"$pgid\" >/dev/null 2>&1 || true",
            "    fi",
            "    rm -f \"$pid_file\"",
            "fi",
            "if command -v pgrep >/dev/null 2>&1; then",
            "    for pid in $(pgrep -f \"$launcher\" || true); do",
            "        kill -TERM \"$pid\" >/dev/null 2>&1 || true",
            "        sleep 1",
            "        kill -KILL \"$pid\" >/dev/null 2>&1 || true",
            "    done",
            "fi",
            f"remote_root=$(readlink -f {shlex.quote(remote_workdir)} 2>/dev/null || printf '%s\\n' {shlex.quote(remote_workdir)})",
            "run_pids=''",
            "for cwd_link in /proc/[0-9]*/cwd; do",
            "    pid=${cwd_link#/proc/}",
            "    pid=${pid%/cwd}",
            "    [ \"$pid\" = \"$$\" ] && continue",
            "    cwd=$(readlink -f \"$cwd_link\" 2>/dev/null || true)",
            "    case \"$cwd\" in",
            "        \"$remote_root\"|\"$remote_root\"/*) run_pids=\"$run_pids $pid\" ;;",
            "    esac",
            "done",
            "for pid in $run_pids; do",
            "    kill -TERM \"$pid\" >/dev/null 2>&1 || true",
            "done",
            "if [ -n \"$run_pids\" ]; then sleep 2; fi",
            "for pid in $run_pids; do",
            "    kill -KILL \"$pid\" >/dev/null 2>&1 || true",
            "done",
            "rm -f \"$launcher\"",
            "exit 0",
        ]
    )
    try:
        _run_ssh(worker, script, timeout=120)
    except Exception:
        pass


def _install_remote_repo_deps(
    *,
    worker: Mapping[str, Any],
    remote_code_dir: str,
    remote_python: str,
    remote_workdir: str | None = None,
) -> None:
    """Best-effort install of cloned repo dependencies on the SSH worker before training."""
    if not GPU_REMOTE_AUTO_PIP_INSTALL:
        return
    setup_log = f"{remote_workdir.rstrip('/')}/.deepgraph_remote_setup.log" if remote_workdir else ""
    script = "\n".join(
        [
            "set -euo pipefail",
            f"setup_log={shlex.quote(setup_log)}",
            (
                "if [ -n \"$setup_log\" ]; then "
                "mkdir -p \"$(dirname \"$setup_log\")\"; "
                ": > \"$setup_log\"; "
                "exec > >(tee -a \"$setup_log\") 2>&1; "
                "fi"
            ),
            'echo "[deepgraph-remote] setup started"',
            f"cd {shlex.quote(remote_code_dir)}",
            "export PIP_DISABLE_PIP_VERSION_CHECK=1",
            "export PIP_DEFAULT_TIMEOUT=120",
            f"{shlex.quote(remote_python)} -m pip --version || {shlex.quote(remote_python)} -m ensurepip --upgrade || true",
            (
                f"{shlex.quote(remote_python)} -m pip install --prefer-binary setuptools wheel "
                "|| echo '[deepgraph-remote] warning: optional pip bootstrap failed; continuing to requirements'"
            ),
            (
                f"if [ -f requirements-experiment.txt ]; then "
                f"echo '[deepgraph-remote] pip install -r requirements-experiment.txt'; "
                f"{shlex.quote(remote_python)} -m pip install --prefer-binary -r requirements-experiment.txt; "
                f"fi"
            ),
            (
                f"if [ -f requirements.txt ]; then "
                f"echo '[deepgraph-remote] pip install -r requirements.txt'; "
                f"{shlex.quote(remote_python)} -m pip install --prefer-binary -r requirements.txt; "
                f"fi"
            ),
            (
                f"if [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ]; then "
                f"echo '[deepgraph-remote] pip install -e .'; "
                f"{shlex.quote(remote_python)} -m pip install -e . || "
                f"{shlex.quote(remote_python)} -m pip install .; "
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
    benchmark_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    metadata = _load_metadata(worker)
    remote_workdir, remote_code_dir = _remote_paths(worker, run_id, local_workdir)
    cleanup_remote_run_processes(worker=worker, run_id=run_id, remote_workdir=remote_workdir)
    sync_workdir_to_remote(worker=worker, local_workdir=local_workdir, remote_workdir=remote_workdir)

    visible_device = str(metadata.get("visible_device") or "0")
    remote_tokens = _remote_python(worker, command_tokens, local_python)
    if not remote_tokens:
        raise RuntimeError("remote experiment received an empty command")

    try:
        _install_remote_repo_deps(
            worker=worker,
            remote_code_dir=remote_code_dir,
            remote_python=remote_tokens[0],
            remote_workdir=remote_workdir,
        )
    except Exception:
        cleanup_remote_run_processes(worker=worker, run_id=run_id, remote_workdir=remote_workdir)
        try:
            sync_workdir_from_remote(worker=worker, remote_workdir=remote_workdir, local_workdir=local_workdir)
        except Exception:
            pass
        raise

    paths = _remote_run_paths(remote_workdir, run_id)
    launcher = _remote_launcher_script(
        run_id=run_id,
        remote_code_dir=remote_code_dir,
        remote_workdir=remote_workdir,
        visible_device=visible_device,
        command_tokens=remote_tokens,
        benchmark_env=benchmark_env or benchmark_env_from_workdir(local_workdir),
    )
    delimiter = f"DEEPGRAPH_REMOTE_LAUNCHER_{int(run_id)}"
    while delimiter in launcher:
        delimiter += "_END"
    remote_lines = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(remote_workdir)}",
        f"cat > {shlex.quote(paths['launcher'])} <<'{delimiter}'",
        launcher,
        delimiter,
        f"chmod +x {shlex.quote(paths['launcher'])}",
        f"timeout --kill-after=30s {max(1, int(time_budget) + 60)}s {shlex.quote(paths['launcher'])}",
    ]
    result: subprocess.CompletedProcess[str] | None = None
    try:
        result = _run_ssh(worker, "\n".join(remote_lines), timeout=max(120, int(time_budget) + 180))
    except subprocess.TimeoutExpired as exc:
        cleanup_remote_run_processes(worker=worker, run_id=run_id, remote_workdir=remote_workdir)
        try:
            sync_workdir_from_remote(worker=worker, remote_workdir=remote_workdir, local_workdir=local_workdir)
        except Exception:
            pass
        raise RuntimeError(f"remote experiment timed out after {exc.timeout}s") from exc
    finally:
        cleanup_remote_run_processes(worker=worker, run_id=run_id, remote_workdir=remote_workdir)
    sync_workdir_from_remote(worker=worker, remote_workdir=remote_workdir, local_workdir=local_workdir)
    if result is None:
        raise RuntimeError("remote experiment failed before producing a process result")
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
