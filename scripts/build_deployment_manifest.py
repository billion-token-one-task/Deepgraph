#!/usr/bin/env python3
"""Build (or verify) a file-level deployment manifest from a reviewed spec.

The production tree is a hybrid runtime snapshot, so nothing here converges a
whole tree. The manifest names exactly the files a batch deploys, each with its
source commit, SHA256, owner/mode expectation, backup path, and the health
check that must pass afterwards.

It is read-only: it hashes source files in this repository, asks Git for path
status in the runtime tree without taking a lock, and prints JSON.  A spec may
also request SHA256-only predeployment target snapshots; target bytes are
hashed locally and are never printed.  It never copies, writes, or restarts.

    build_deployment_manifest.py --spec deploy/manifest/recovery_2026-08-03.spec.json
    build_deployment_manifest.py --spec ... --out deploy/manifest/recovery_2026-08-03.json
    build_deployment_manifest.py --spec ... --verify deploy/manifest/recovery_2026-08-03.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit() -> str:
    completed = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "uncommitted"


def _worktree_clean() -> bool:
    """Tracked-file cleanliness: every manifested file is committed.

    Untracked files are ignored on purpose. They cannot be manifested (each
    entry records a source commit), and unrelated working files must not block
    a deployment record.
    """
    completed = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return not completed.stdout.strip()


def _runtime_local_changes(runtime_root: str) -> set[str]:
    """Paths the runtime tree has changed relative to its own snapshot."""
    completed = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "-C",
            runtime_root,
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    changed: set[str] = set()
    for line in completed.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        changed.add(path)
    return changed


def build(spec_path: Path) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    runtime_root = str(spec.get("runtime_root") or "")
    backup_root = str(spec.get("backup_root") or "/home/ec2-user/deepgraph-rollback")
    manifest_key = str(spec["manifest_key"])
    capture_predeploy = bool(spec.get("capture_target_predeploy_sha256"))
    local_changes = _runtime_local_changes(runtime_root) if runtime_root else set()

    batches = []
    for raw_batch in spec.get("batches", []):
        files = []
        for entry in raw_batch.get("files", []):
            source = ROOT / str(entry["source"])
            if not source.is_file():
                raise SystemExit(f"manifest source is missing:{entry['source']}")
            target = str(entry["target"]).replace("{runtime_root}", runtime_root)
            relative = str(entry["source"])
            target_path = Path(target)
            target_present = target_path.is_file() if target_path.is_absolute() else False
            target_predeploy_sha256 = (
                _sha256(target_path)
                if capture_predeploy and target_present
                else None
            )
            files.append(
                {
                    "source": relative,
                    "source_sha256": _sha256(source),
                    "source_bytes": source.stat().st_size,
                    "target": target,
                    "owner": entry.get("owner", ""),
                    "mode": entry.get("mode", ""),
                    "kind": entry.get("kind", "file"),
                    "replaces": entry.get("replaces"),
                    # A file the runtime already modified locally must be
                    # diffed by an operator before it is overwritten.
                    "runtime_locally_modified": relative in local_changes,
                    "backup_artifact": (
                        f"{backup_root}/{manifest_key}/{Path(target).name}.bak"
                        if entry.get("kind", "file") == "file"
                        else None
                    ),
                    "target_predeploy_state": (
                        "present"
                        if target_present
                        else "absent"
                        if target_path.is_absolute()
                        else "not_a_file_target"
                    ),
                    "target_predeploy_sha256": target_predeploy_sha256,
                }
            )
        batches.append(
            {
                "batch": raw_batch["batch"],
                "authorization": raw_batch.get("authorization", "required-before-deploy"),
                "purpose": raw_batch.get("purpose", ""),
                "restart_required": raw_batch.get("restart_required", []),
                "systemd_reload_required": bool(
                    raw_batch.get("systemd_reload_required")
                ),
                "health_check": raw_batch.get("health_check", ""),
                "acceptance": raw_batch.get("acceptance", ""),
                "files": files,
            }
        )

    needs_review = sorted(
        entry["source"]
        for batch in batches
        for entry in batch["files"]
        if entry["runtime_locally_modified"]
    )
    return {
        "manifest_key": manifest_key,
        "description": spec.get("description", ""),
        "source_commit": _source_commit(),
        "source_worktree_clean": _worktree_clean(),
        "runtime_root": runtime_root,
        "backup_root": backup_root,
        "environment_keys": spec.get("environment_keys", []),
        "preconditions": spec.get("preconditions", []),
        "deployment_steps": spec.get("deployment_steps", []),
        "tests": spec.get("tests", ""),
        "batches": batches,
        "operator_diff_required": needs_review,
        "rollback": spec.get("rollback")
        or {
            "order": "reverse batch order",
            "steps": [
                "restore each backup_artifact over its target and verify SHA256",
                "systemctl daemon-reload when a unit file changed",
                "restart only the services listed by the applied batches",
                "additive schema migrations remain in place and are unused by previous code",
            ],
        },
        "production_tree_modified_by_this_command": False,
        "file_contents_read_from_production": capture_predeploy,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()

    manifest = build(args.spec if args.spec.is_absolute() else ROOT / args.spec)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)

    if args.verify:
        existing = json.loads(args.verify.read_text(encoding="utf-8"))
        drift = [
            entry["source"]
            for batch, recorded in zip(manifest["batches"], existing.get("batches", []))
            for entry, recorded_entry in zip(batch["files"], recorded.get("files", []))
            if entry["source_sha256"] != recorded_entry.get("source_sha256")
        ]
        print(json.dumps({"drift": drift}, indent=2, sort_keys=True))
        return 1 if drift else 0

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
