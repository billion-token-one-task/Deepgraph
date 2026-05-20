#!/usr/bin/env python
"""End-to-end: forge + validate for an insight, bypassing the gpu_scheduler queue."""
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
        description="Forge and validate an experiment end-to-end for a deep insight.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Full pipeline for insight 42 on GPU 0 with streaming
  python scripts/run_full_pipeline.py --insight-id 42 --gpu 0 --stream

  # Full pipeline with full benchmark
  python scripts/run_full_pipeline.py --insight-id 42 --gpu 0 --full-benchmark
""",
    )
    parser.add_argument("--insight-id", type=int, required=True, help="deep_insights.id to forge and run")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--stream", action="store_true", help="Stream subprocess output to console")
    parser.add_argument("--full-benchmark", action="store_true", help="Run full benchmark after validation")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.stream:
        os.environ["DEEPGRAPH_STREAM_OUTPUT"] = "1"

    db.init_db()

    # Phase 1: Forge
    print(f"=== Forging experiment for insight {args.insight_id} ===")
    from agents.experiment_forge import forge_experiment

    forge_result = forge_experiment(args.insight_id)
    run_id = forge_result.get("run_id")
    workdir = forge_result.get("workdir")
    if not run_id:
        print(f"Forge failed: {forge_result.get('error', 'unknown')}")
        sys.exit(1)
    print(f"  run_id:  {run_id}")
    print(f"  workdir: {workdir}")

    # Phase 2: Validate
    print(f"\n=== Running validation loop for run {run_id} ===")
    import socket
    hostname = socket.gethostname()
    metadata = json.dumps({"visible_device": args.gpu, "backend": "local"})
    execution_context: dict = {
        "worker": {
            "id": f"{hostname}:gpu{args.gpu}",
            "hostname": hostname,
            "gpu_index": 0,
            "gpu_model": "local",
            "total_mem_gb": 0,
            "status": "busy",
            "metadata": metadata,
        },
        "job": {},
        "full_benchmark": args.full_benchmark,
        "stream_output": args.stream,
    }

    from agents.validation_loop import run_validation_loop
    result = run_validation_loop(run_id, execution_context=execution_context)

    # Phase 3: Optional full benchmark
    if args.full_benchmark and isinstance(result, dict) and result.get("verdict") == "confirmed":
        print(f"\n=== Running full benchmark for run {run_id} ===")
        from agents.validation_loop import run_full_benchmark_completion
        benchmark_result = run_full_benchmark_completion(run_id, execution_context=execution_context)
        result["full_benchmark"] = benchmark_result

    if args.json:
        output = {"forge": forge_result, "validation": result}
        print(json.dumps(output, indent=2, default=str))
    else:
        verdict = result.get("verdict", "unknown") if isinstance(result, dict) else "unknown"
        print(f"\n=== Pipeline complete ===")
        print(f"  insight: {args.insight_id}")
        print(f"  run_id:  {run_id}")
        print(f"  verdict: {verdict}")
        if isinstance(result, dict):
            for key in ("effect_pct", "best_metric", "error"):
                if result.get(key) is not None:
                    print(f"  {key}: {result[key]}")


if __name__ == "__main__":
    main()
