#!/usr/bin/env python3
"""Deploy, verify, or roll back one batch of a generated deployment manifest.

Every file is backed up before it is overwritten, every backup's SHA256 is
recorded next to it, and every deployed file is re-hashed and compared against
the manifest afterwards. A mismatch aborts the batch instead of continuing.

    deploy_manifest_batch.py --manifest M --batch 1-selfheal --verify-only
    deploy_manifest_batch.py --manifest M --batch 3-application --apply
    deploy_manifest_batch.py --manifest M --batch 3-application --rollback

``--apply`` never restarts a service on its own: the manifest's
``restart_required`` list is printed and left to the operator step, so a
deployment and a restart stay separately auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_target(path: str) -> bytes | None:
    """Read a target that may be root-owned, without copying it anywhere."""
    result = subprocess.run(
        ["sudo", "cat", path], capture_output=True, check=False
    )
    return result.stdout if result.returncode == 0 else None


def _batch(manifest: dict, name: str) -> dict:
    for batch in manifest["batches"]:
        if batch["batch"] == name:
            return batch
    raise SystemExit(f"unknown batch:{name} (have: " + ",".join(
        b["batch"] for b in manifest["batches"]) + ")")


def _deployable(batch: dict) -> list[dict]:
    return [entry for entry in batch["files"] if entry.get("kind", "file") == "file"]


def verify(manifest: dict, batch: dict) -> int:
    drift = 0
    for entry in _deployable(batch):
        current = _read_target(entry["target"])
        if current is None:
            print(f"ABSENT  {entry['target']}")
            drift += 1
            continue
        digest = _sha256_bytes(current)
        if digest == entry["source_sha256"]:
            print(f"OK      {entry['target']}")
        else:
            print(f"DRIFT   {entry['target']} deployed={digest[:16]} "
                  f"manifest={entry['source_sha256'][:16]}")
            drift += 1
    print(f"verified={len(_deployable(batch))} drift={drift}")
    return 1 if drift else 0


def apply(manifest: dict, batch: dict) -> int:
    backup_dir = Path(manifest["backup_root"]) / manifest["manifest_key"] / batch["batch"]
    subprocess.run(["sudo", "mkdir", "-p", str(backup_dir)], check=True)
    records = []
    for entry in _deployable(batch):
        source = ROOT / entry["source"]
        expected = _sha256_bytes(source.read_bytes())
        if expected != entry["source_sha256"]:
            raise SystemExit(
                f"source drifted from the manifest:{entry['source']}; "
                "regenerate the manifest before deploying"
            )
        target = entry["target"]
        existing = _read_target(target)
        backup = backup_dir / (Path(target).name + ".bak")
        if existing is None:
            print(f"NEW     {target} (no prior file to back up)")
            records.append({"target": target, "backup": None, "previous_sha256": None})
        else:
            subprocess.run(["sudo", "cp", "-p", target, str(backup)], check=True)
            records.append(
                {
                    "target": target,
                    "backup": str(backup),
                    "previous_sha256": _sha256_bytes(existing),
                }
            )
            print(f"BACKUP  {target} -> {backup}")
        owner = entry.get("owner") or "root:root"
        mode = entry.get("mode") or "0644"
        subprocess.run(
            ["sudo", "install", "-D", "-o", owner.split(":")[0], "-g",
             owner.split(":")[-1], "-m", mode, str(source), target],
            check=True,
        )
        deployed = _read_target(target)
        if deployed is None or _sha256_bytes(deployed) != entry["source_sha256"]:
            raise SystemExit(f"post-deploy SHA256 mismatch:{target}; roll back now")
        print(f"DEPLOY  {target} {entry['source_sha256'][:16]} OK")

    ledger = backup_dir / "deployed.json"
    payload = json.dumps(
        {
            "manifest_key": manifest["manifest_key"],
            "source_commit": manifest["source_commit"],
            "batch": batch["batch"],
            "files": records,
        },
        indent=2,
        sort_keys=True,
    )
    subprocess.run(
        ["sudo", "tee", str(ledger)], input=payload.encode("utf-8"),
        stdout=subprocess.DEVNULL, check=True,
    )
    print(f"ledger  {ledger}")
    if batch.get("systemd_reload_required"):
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        print("systemd daemon-reload done")
    if batch.get("restart_required"):
        print("RESTART REQUIRED (left to the operator step): "
              + ",".join(batch["restart_required"]))
    print("health_check: " + batch.get("health_check", ""))
    return 0


def rollback(manifest: dict, batch: dict) -> int:
    backup_dir = Path(manifest["backup_root"]) / manifest["manifest_key"] / batch["batch"]
    ledger_bytes = _read_target(str(backup_dir / "deployed.json"))
    if ledger_bytes is None:
        raise SystemExit(f"no deployment ledger at {backup_dir}/deployed.json")
    ledger = json.loads(ledger_bytes)
    for record in ledger["files"]:
        if not record["backup"]:
            print(f"SKIP    {record['target']} (was new; remove manually if wanted)")
            continue
        subprocess.run(["sudo", "cp", "-p", record["backup"], record["target"]], check=True)
        restored = _read_target(record["target"])
        if restored is None or _sha256_bytes(restored) != record["previous_sha256"]:
            raise SystemExit(f"rollback SHA256 mismatch:{record['target']}")
        print(f"RESTORE {record['target']} {record['previous_sha256'][:16]} OK")
    if batch.get("systemd_reload_required"):
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        print("systemd daemon-reload done")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--batch", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--rollback", action="store_true")
    args = parser.parse_args()

    path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    manifest = json.loads(path.read_text(encoding="utf-8"))
    batch = _batch(manifest, args.batch)
    if not shutil.which("sudo"):
        raise SystemExit("sudo is required to read and write deployment targets")
    if args.verify_only:
        return verify(manifest, batch)
    if args.apply:
        return apply(manifest, batch)
    return rollback(manifest, batch)


if __name__ == "__main__":
    raise SystemExit(main())
