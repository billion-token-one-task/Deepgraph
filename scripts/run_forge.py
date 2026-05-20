#!/usr/bin/env python
"""Standalone experiment forge: create a workspace for an insight without the full pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import database as db


def main() -> None:
    parser = argparse.ArgumentParser(description="Forge an experiment workspace for a deep insight.")
    parser.add_argument("--insight-id", type=int, required=True, help="deep_insights.id to forge")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    args = parser.parse_args()

    db.init_db()
    from agents.experiment_forge import forge_experiment

    result = forge_experiment(args.insight_id)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        run_id = result.get("run_id")
        workdir = result.get("workdir")
        print(f"Forged experiment for insight {args.insight_id}")
        print(f"  run_id:  {run_id}")
        print(f"  workdir: {workdir}")
        if result.get("error"):
            print(f"  error:   {result['error']}")


if __name__ == "__main__":
    main()
