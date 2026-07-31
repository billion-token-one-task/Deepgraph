#!/usr/bin/env python3
"""Audit that a CGGR method shard is contract-compatible with a canonical run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONTRACT_FIELDS = (
    "model_id",
    "seeds",
    "seed_values",
    "max_examples_per_dataset_seed",
    "cost_lambda",
    "decoding",
    "reasoning_budget",
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _target_names(config: dict) -> list[str]:
    names: list[str] = []
    for row in config.get("targets") or []:
        if isinstance(row, dict):
            text = str(row.get("name") or "").strip()
        else:
            text = str(row or "").strip()
        if text:
            names.append(text)
    return names


def _method_set(config: dict) -> set[str]:
    return {str(item).strip() for item in config.get("methods") or [] if str(item).strip()}


def audit(canonical_workdir: Path, shard_workdir: Path) -> dict:
    blockers: list[str] = []
    warnings: list[str] = []
    canonical_config = _load_json(canonical_workdir / "results" / "run_config.json")
    shard_config = _load_json(shard_workdir / "results" / "run_config.json")
    shard_spec = _load_json(shard_workdir / "spec" / "shard_config.json")

    if not isinstance(canonical_config, dict):
        blockers.append("missing or invalid canonical results/run_config.json")
        canonical_config = {}
    if not isinstance(shard_config, dict):
        blockers.append("missing or invalid shard results/run_config.json")
        shard_config = {}
    if not isinstance(shard_spec, dict):
        blockers.append("missing or invalid shard spec/shard_config.json")
        shard_spec = {}

    for field in CONTRACT_FIELDS:
        if canonical_config.get(field) != shard_config.get(field):
            blockers.append(f"contract field mismatch: {field}")

    canonical_targets = _target_names(canonical_config)
    shard_targets = _target_names(shard_config)
    if canonical_targets != shard_targets:
        blockers.append("target list mismatch")

    shard_axes = shard_config.get("shard_axes") if isinstance(shard_config.get("shard_axes"), dict) else {}
    if not shard_config.get("sharded_run") or not shard_axes.get("method"):
        blockers.append("shard run_config is not marked as a method shard")
    if shard_axes.get("target") or shard_axes.get("seed"):
        blockers.append("shard changes target or seed axis; only method shards are supported")

    expected_methods = {str(item).strip() for item in shard_spec.get("method_subset") or [] if str(item).strip()}
    observed_methods = _method_set(shard_config)
    if expected_methods and observed_methods != expected_methods:
        blockers.append("shard method subset does not match spec/shard_config.json")
    if not expected_methods:
        blockers.append("shard_config.json has no method_subset")

    canonical_methods = _method_set(canonical_config)
    missing_from_canonical = sorted(method for method in observed_methods if method not in canonical_methods)
    if missing_from_canonical:
        warnings.append(
            "shard has methods not present in canonical run_config: "
            + ", ".join(missing_from_canonical)
        )

    return {
        "ok": not blockers,
        "canonical_workdir": str(canonical_workdir),
        "shard_workdir": str(shard_workdir),
        "blockers": blockers,
        "warnings": warnings,
        "canonical_methods": sorted(canonical_methods),
        "shard_methods": sorted(observed_methods),
        "contract_fields": list(CONTRACT_FIELDS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical_workdir", type=Path)
    parser.add_argument("shard_workdir", type=Path)
    args = parser.parse_args()
    result = audit(args.canonical_workdir, args.shard_workdir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
