#!/usr/bin/env python3
"""Wide deep-insight pass: re-harvest SQL signals, then Tier 1 + Tier 2 at bulk limits.

Uses env-tunable DISCOVERY_BULK_* in config.py. LLM cost is much higher than the
default scheduler pass.

Usage (from repo root):
  cd deepgraph && python3.12 run_bulk_deep_insights.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure package root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from orchestrator.discovery_scheduler import run_bulk_deep_insights  # noqa: E402


def main() -> None:
    print(
        "[BULK] Starting run_bulk_deep_insights at",
        datetime.now(timezone.utc).isoformat(),
        flush=True,
    )
    out = run_bulk_deep_insights()
    log_path = Path(__file__).resolve().parent / "bulk_discovery_result.json"
    log_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("[BULK] Done. Summary written to", log_path, flush=True)
    print(
        "Tier1 stored:",
        len(out.get("tier1") or []),
        "Tier2 stored:",
        len(out.get("tier2") or []),
        flush=True,
    )


if __name__ == "__main__":
    main()
