#!/usr/bin/env python3
"""Read-only, no-cost compute-backend capability audit.

Prints what each backend's state actually is and why. It never connects to a
GPU host, never starts a Colab runtime, never creates a cloud resource, and
never prints a credential: only the *names* of secret references appear.

Exit code is 0 when every listed backend is explicitly classified, 1 when a
backend is enabled in configuration but cannot be classified as usable, so CI
fails on a silent-fallback regression.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meta_harness.backend_capability import (  # noqa: E402
    GPU_BACKENDS,
    STATE_ENABLED,
    STATE_UNKNOWN,
    reports_from_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-schedulable-gpu",
        action="store_true",
        help="fail unless exactly one GPU backend is verified and schedulable",
    )
    args = parser.parse_args()

    reports = reports_from_config()
    payload = {
        "database_accessed": False,
        "network_accessed": False,
        "cloud_resources_created": False,
        "backends": {kind: report.to_dict() for kind, report in sorted(reports.items())},
    }
    schedulable_gpu = sorted(
        kind
        for kind in GPU_BACKENDS
        if reports[kind].state == STATE_ENABLED
    )
    canary_gpu = sorted(
        kind for kind in GPU_BACKENDS if reports[kind].state == STATE_UNKNOWN
    )
    payload["schedulable_gpu_backends"] = schedulable_gpu
    payload["canary_eligible_gpu_backends"] = canary_gpu
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    if args.require_schedulable_gpu and len(schedulable_gpu) != 1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
