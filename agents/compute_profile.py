"""Runtime compute capability detection for idea and experiment routing."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

from config import (
    COMPUTE_LOCAL_GPU_POLICY,
    GPU_MODE,
    GPU_REMOTE_SSH_HOST,
    GPU_REMOTE_SSH_USER,
)


@dataclass(frozen=True)
class ComputeProfile:
    local_gpu_available: bool
    remote_gpu_configured: bool
    gpu_allowed: bool
    gpu_block_reason: str
    accelerator: str = "cpu"
    device_count: int = 0
    total_vram_gb: float = 0.0
    device_names: tuple[str, ...] = ()

    @property
    def has_gpu_lane(self) -> bool:
        return self.gpu_allowed and (self.local_gpu_available or self.remote_gpu_configured)


def _nvidia_smi_profile() -> tuple[bool, int, float, tuple[str, ...]]:
    if not shutil.which("nvidia-smi"):
        return False, 0, 0.0, ()
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False, 0, 0.0, ()
    if proc.returncode != 0:
        return False, 0, 0.0, ()
    names: list[str] = []
    total_mb = 0.0
    for raw in proc.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",", 1)]
        if len(parts) != 2:
            continue
        names.append(parts[0])
        try:
            total_mb += float(parts[1])
        except ValueError:
            pass
    return bool(names), len(names), round(total_mb / 1024, 2), tuple(names)


def detect_compute_profile() -> ComputeProfile:
    """Probe available compute and apply the configured GPU admission policy.

    Policy values:
    - ``auto``: allow GPU ideas only when local CUDA/NVIDIA or remote SSH GPU is configured.
    - ``require``: same as auto, but reports a stricter block reason.
    - ``allow_remote``: allow SSH GPU even without local GPU.
    - ``force_cpu``: block all GPU ideas.
    - ``ignore``: do not block GPU ideas even if no GPU is detected.
    """

    policy = (COMPUTE_LOCAL_GPU_POLICY or "auto").strip().lower()
    local_ok, count, vram, names = _nvidia_smi_profile()
    remote_ok = GPU_MODE == "ssh" and bool(GPU_REMOTE_SSH_HOST and GPU_REMOTE_SSH_USER)
    accelerator = "cuda" if local_ok else ("ssh_gpu" if remote_ok else "cpu")

    if policy == "force_cpu":
        return ComputeProfile(
            local_gpu_available=local_ok,
            remote_gpu_configured=remote_ok,
            gpu_allowed=False,
            gpu_block_reason="compute.local_gpu_policy=force_cpu",
            accelerator=accelerator,
            device_count=count,
            total_vram_gb=vram,
            device_names=names,
        )
    if policy == "ignore":
        return ComputeProfile(
            local_gpu_available=local_ok,
            remote_gpu_configured=remote_ok,
            gpu_allowed=True,
            gpu_block_reason="",
            accelerator=accelerator,
            device_count=count,
            total_vram_gb=vram,
            device_names=names,
        )

    gpu_lane = local_ok or remote_ok
    if gpu_lane:
        return ComputeProfile(
            local_gpu_available=local_ok,
            remote_gpu_configured=remote_ok,
            gpu_allowed=True,
            gpu_block_reason="",
            accelerator=accelerator,
            device_count=count,
            total_vram_gb=vram,
            device_names=names,
        )

    reason = (
        "GPU resource requested but no local NVIDIA GPU was detected and no SSH GPU worker is configured."
        if policy != "require"
        else "compute.local_gpu_policy=require but no usable local/remote GPU was detected."
    )
    return ComputeProfile(
        local_gpu_available=False,
        remote_gpu_configured=False,
        gpu_allowed=False,
        gpu_block_reason=reason,
        accelerator="cpu",
        device_count=0,
        total_vram_gb=0.0,
        device_names=(),
    )


def gpu_required_resource(resource_class: str | None) -> bool:
    return str(resource_class or "").strip().lower().startswith("gpu")


def gpu_resource_allowed(resource_class: str | None, profile: ComputeProfile | None = None) -> tuple[bool, str]:
    if not gpu_required_resource(resource_class):
        return True, ""
    profile = profile or detect_compute_profile()
    if profile.has_gpu_lane:
        return True, ""
    return False, profile.gpu_block_reason

