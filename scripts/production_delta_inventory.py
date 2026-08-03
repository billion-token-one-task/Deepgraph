#!/usr/bin/env python3
"""Read-only inventory of the production runtime's delta from its snapshot.

The production tree at ``/home/billion-token/Deepgraph`` is a hybrid runtime
snapshot, not a clean checkout. This command classifies *paths only* so an
operator can decide what to port, what belongs in a protected deployment
manifest, and what is disposable.

It never reads file contents, never writes to the production tree, never runs a
Git command that mutates state (``--no-optional-locks`` keeps the index
untouched), and never prints business data. Output is path names and counts.

Classification:

1. ``source``    -- code that must be ported to master before deployment;
2. ``runtime_config`` -- environment/config that belongs in a protected
   deployment manifest and never in Git;
3. ``generated``  -- artifacts, backups, logs, caches, temporary files;
4. ``review``     -- obsolete or unrecognized paths needing operator review.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import PurePosixPath


DEFAULT_TREE = "/home/billion-token/Deepgraph"

SOURCE_SUFFIXES = {".py", ".sql", ".toml", ".sh", ".service", ".timer", ".ps1"}
SOURCE_ROOTS = {
    "agents",
    "contracts",
    "db",
    "deploy",
    "ingestion",
    "meta_harness",
    "orchestrator",
    "plugins",
    "scripts",
    "web",
    "tests",
    "compat",
}
CONFIG_NAMES = {".env", ".env.local", "deepgraph.toml", "cla-signers.json"}
CONFIG_SUFFIXES = {".env"}
GENERATED_ROOTS = {
    "logs",
    "workspace",
    "artifacts",
    "backups",
    "output",
    "outputs",
    "figures",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "node_modules",
}
GENERATED_SUFFIXES = {
    ".log",
    ".pyc",
    ".pdf",
    ".png",
    ".jpg",
    ".zip",
    ".tar",
    ".gz",
    ".dump",
    ".sqlite",
    ".db",
    ".bak",
    ".tmp",
    ".pptx",
    ".docx",
}


def classify(path: str) -> str:
    pure = PurePosixPath(path)
    parts = pure.parts
    name = pure.name
    suffix = pure.suffix.lower()

    # Credential-bearing files are never disposable, whatever their suffix:
    # ".env.bak-*" copies still hold live secrets and stay operator-owned.
    if name.startswith(".env"):
        return "runtime_config"
    if any(part in GENERATED_ROOTS for part in parts):
        return "generated"
    if suffix in GENERATED_SUFFIXES:
        return "generated"
    if name in CONFIG_NAMES or suffix in CONFIG_SUFFIXES:
        return "runtime_config"
    if parts and parts[0] in SOURCE_ROOTS and suffix in SOURCE_SUFFIXES:
        return "source"
    if len(parts) == 1 and suffix in SOURCE_SUFFIXES:
        return "source"
    if len(parts) == 1 and suffix in {".md", ".txt"}:
        return "source"
    return "review"


def porcelain(tree: str) -> list[tuple[str, str]]:
    """Return (status, path) pairs without mutating the repository."""
    completed = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "-C",
            tree,
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    entries: list[tuple[str, str]] = []
    for line in completed.stdout.splitlines():
        if len(line) < 4:
            continue
        status = line[:2].strip() or "?"
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        entries.append((status, path))
    return entries


def inventory(tree: str, *, examples: int = 8) -> dict:
    entries = porcelain(tree)
    buckets: dict[str, list[str]] = {
        "source": [],
        "runtime_config": [],
        "generated": [],
        "review": [],
    }
    status_counts: Counter[str] = Counter()
    for status, path in entries:
        status_counts[status] += 1
        buckets[classify(path)].append(path)
    head = subprocess.run(
        ["git", "--no-optional-locks", "-C", tree, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "--no-optional-locks", "-C", tree, "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "tree": tree,
        "head": head,
        "branch": branch,
        "file_contents_read": False,
        "production_tree_modified": False,
        "total_paths": len(entries),
        "status_counts": dict(sorted(status_counts.items())),
        "counts": {name: len(paths) for name, paths in sorted(buckets.items())},
        "examples": {
            name: sorted(paths)[:examples] for name, paths in sorted(buckets.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", default=DEFAULT_TREE)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            inventory(args.tree, examples=args.examples),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
