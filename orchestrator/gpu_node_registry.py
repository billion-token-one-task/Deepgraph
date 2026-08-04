"""Register SSH GPU nodes from their real hardware, keyed by host and port.

Two problems this fixes, both hit during the 2026-08-04 recovery:

* worker ids were ``ssh:{host}:gpu{n}`` with no port, so two rented nodes behind
  one public IP on different ports collided and only one could ever register;
* the GPU model and memory were typed into configuration, so the table said
  "L40S 46GB" while the box was actually an A100. Here they are read from the
  machine with ``nvidia-smi`` instead.

A node is described by non-secret fields plus a ``credential_ref`` naming the
environment variable that holds its SSH password or key. The registrar never
handles the credential itself; the existing ``ssh_gpu_backend`` connection code
resolves it at run time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SSHNodeSpec:
    host: str
    port: int
    user: str
    credential_ref: str
    remote_base_dir: str = "/root/deepgraph-remote-worker"
    python_bin: str = "python"

    def validate(self) -> None:
        if not self.host or not self.user:
            raise ValueError("SSH node requires host and user")
        if not (0 < int(self.port) < 65536):
            raise ValueError("SSH node port is out of range")
        if not str(self.credential_ref or "").strip():
            raise ValueError("SSH node requires a credential reference")
        if not str(self.credential_ref).startswith(("env:", "secret:")):
            raise ValueError(
                "credential_ref must be an env: or secret: reference, never a literal"
            )


def parse_gpu_query(text: str) -> list[dict]:
    """Parse ``nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader``."""
    gpus: list[dict] = []
    for line in str(text or "").strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        mib = None
        for token in parts[2].replace("MiB", " ").replace("MB", " ").split():
            if token.isdigit():
                mib = int(token)
                break
        gpus.append(
            {
                "device": parts[0],
                "gpu_model": parts[1] or "unknown",
                "total_mem_gb": round(mib / 1024, 2) if mib else None,
            }
        )
    return gpus


QUERY = "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader"


def canary_node(
    spec: SSHNodeSpec,
    *,
    run_remote: Callable[[dict, str], object],
    worker_row: Callable[[SSHNodeSpec], dict],
) -> list[dict]:
    """Read the node's GPUs over SSH without running any workload.

    ``run_remote`` is the existing ``ssh_gpu_backend._run_ssh`` and
    ``worker_row`` builds the metadata dict it expects. Raises if the node is
    unreachable or reports no GPU, so an unverified node is never registered.
    """
    spec.validate()
    completed = run_remote(worker_row(spec), QUERY)
    if getattr(completed, "returncode", 1) != 0:
        stderr = str(getattr(completed, "stderr", "") or "")[:200]
        raise RuntimeError(f"GPU canary failed for {spec.host}:{spec.port}: {stderr}")
    gpus = parse_gpu_query(getattr(completed, "stdout", "") or "")
    if not gpus:
        raise RuntimeError(
            f"GPU canary returned no devices for {spec.host}:{spec.port}"
        )
    return gpus


def node_spec_from_mapping(payload: dict) -> SSHNodeSpec:
    spec = SSHNodeSpec(
        host=str(payload.get("host") or "").strip(),
        port=int(payload.get("port") or 22),
        user=str(payload.get("user") or "").strip(),
        credential_ref=str(payload.get("credential_ref") or "").strip(),
        remote_base_dir=str(payload.get("remote_base_dir") or "/root/deepgraph-remote-worker"),
        python_bin=str(payload.get("python_bin") or "python"),
    )
    spec.validate()
    return spec


def configured_nodes(raw_json: str) -> list[SSHNodeSpec]:
    """Parse ``DEEPGRAPH_GPU_SSH_NODES`` (a JSON list). Empty on any problem."""
    raw = str(raw_json or "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = payload.get("nodes") or []
    if not isinstance(payload, list):
        return []
    nodes = []
    for item in payload:
        if isinstance(item, dict):
            try:
                nodes.append(node_spec_from_mapping(item))
            except (TypeError, ValueError):
                continue
    return nodes
