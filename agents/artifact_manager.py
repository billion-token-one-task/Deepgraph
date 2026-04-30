"""Filesystem artifact helpers for SciForge experiment workdirs."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ARTIFACT_DIRS = ("logs", "results", "figures", "tables", "manuscript", "reviews")
MANIFEST_NAME = "artifact_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_workdir(workdir: Path) -> Path:
    return Path(workdir).resolve()


def _ensure_inside_workdir(workdir: Path, path: Path) -> Path:
    root = _resolve_workdir(workdir)
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Artifact path escapes workdir: {path}")
    return resolved


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(workdir: Path) -> Path:
    return _resolve_workdir(workdir) / MANIFEST_NAME


def _load_manifest(workdir: Path, run_id: int | None = None) -> dict:
    path = _manifest_path(workdir)
    if not path.exists():
        return {"schema_version": 1, "run_id": run_id, "artifacts": []}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        manifest = {"schema_version": 1, "run_id": run_id, "artifacts": []}
    manifest.setdefault("schema_version", 1)
    manifest.setdefault("artifacts", [])
    if run_id is not None:
        manifest["run_id"] = run_id
    return manifest


def ensure_artifact_dirs(workdir: Path) -> dict[str, Path]:
    """Create the standard artifact directories for an experiment workdir."""
    root = _resolve_workdir(workdir)
    dirs = {}
    for name in ARTIFACT_DIRS:
        path = root / "artifacts" / name
        path.mkdir(parents=True, exist_ok=True)
        dirs[name] = path
    return dirs


def artifact_path(workdir: Path, relative_path: str) -> Path:
    """Return a safe path inside workdir for a relative artifact path."""
    rel = Path(relative_path)
    if rel.is_absolute():
        raise ValueError(f"Artifact path must be relative: {relative_path}")
    return _ensure_inside_workdir(workdir, _resolve_workdir(workdir) / rel)


def record_artifact(
    workdir: Path,
    run_id: int,
    artifact_type: str,
    path: Path,
    metadata: dict | None = None,
) -> dict:
    """Record an artifact in the workdir manifest and return the manifest entry."""
    root = _resolve_workdir(workdir)
    resolved = _ensure_inside_workdir(root, path)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(str(path))

    rel_path = resolved.relative_to(root).as_posix()
    entry = {
        "type": artifact_type,
        "path": rel_path,
        "sha256": _checksum(resolved),
        "created_at": _utc_now(),
        "metadata": metadata or {},
    }

    manifest = _load_manifest(root, run_id=run_id)
    manifest["artifacts"] = [
        item for item in manifest.get("artifacts", [])
        if not (item.get("type") == artifact_type and item.get("path") == rel_path)
    ]
    manifest["artifacts"].append(entry)
    _manifest_path(root).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return entry


def list_artifacts(workdir: Path) -> list[dict]:
    """List manifest artifacts, or an empty list if no manifest exists."""
    path = _manifest_path(workdir)
    if not path.exists():
        return []
    return list(_load_manifest(workdir).get("artifacts", []))


def read_text_artifact(workdir: Path, relative_path: str, max_chars: int = 20000) -> str:
    """Read a text artifact inside workdir, capped to max_chars."""
    path = artifact_path(workdir, relative_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(relative_path)
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
