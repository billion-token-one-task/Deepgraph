#!/usr/bin/env python
"""Standalone experiment runner: run the validation loop for a specific run with debug options."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import database as db


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the validation loop for a specific experiment run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Normal execution with streaming output
  python scripts/run_experiment.py --run-id 123 --stream

  # Single iteration only (quick test)
  python scripts/run_experiment.py --run-id 123 --single-iter --phase reproduction

  # Select GPU device explicitly
  python scripts/run_experiment.py --run-id 123 --gpu 1 --stream

  # Full benchmark completion mode
  python scripts/run_experiment.py --run-id 123 --full-benchmark
""",
    )
    parser.add_argument("--run-id", type=int, required=True, help="experiment_runs.id to run")
    parser.add_argument("--stream", action="store_true", help="Stream subprocess output to console in real-time")
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES override (e.g. '0' or '0,1')")
    parser.add_argument("--full-benchmark", action="store_true", help="Run full benchmark completion mode")
    parser.add_argument("--single-iter", action="store_true", help="Stop after a single iteration")
    parser.add_argument("--phase", type=str, default=None, choices=["reproduction", "hypothesis"], help="Phase to run (with --single-iter)")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.stream:
        os.environ["DEEPGRAPH_STREAM_OUTPUT"] = "1"

    db.init_db()

    execution_context: dict = {
        "worker": _build_local_worker(args.gpu),
        "job": {},
        "full_benchmark": args.full_benchmark,
        "stream_output": args.stream,
    }

    if args.full_benchmark:
        from agents.validation_loop import run_full_benchmark_completion
        result = run_full_benchmark_completion(args.run_id, execution_context=execution_context)
    else:
        from agents.validation_loop import run_validation_loop
        result = run_validation_loop(args.run_id, execution_context=execution_context)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        verdict = result.get("verdict", "unknown") if isinstance(result, dict) else "unknown"
        print(f"\nValidation loop completed for run {args.run_id}")
        print(f"  verdict: {verdict}")
        if isinstance(result, dict):
            for key in ("effect_pct", "best_metric", "error"):
                if result.get(key) is not None:
                    print(f"  {key}: {result[key]}")


def _build_local_worker(gpu: str | None) -> dict:
    """Build a synthetic worker dict for standalone execution."""
    import socket
    hostname = socket.gethostname()
    device = gpu or os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    metadata = json.dumps({"visible_device": device, "backend": "local"})
    return {
        "id": f"{hostname}:gpu{device}",
        "hostname": hostname,
        "gpu_index": 0,
        "gpu_model": "local",
        "total_mem_gb": 0,
        "status": "busy",
        "metadata": metadata,
        "visible_device": device,
    }


if __name__ == "__main__":
    main()
