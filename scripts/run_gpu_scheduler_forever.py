"""Run the DeepGraph GPU scheduler as a small foreground service."""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator import gpu_scheduler


def main() -> int:
    result = gpu_scheduler.start()
    print(result, flush=True)
    status = result.get("status") if isinstance(result, dict) else ""
    if str(status).startswith("already_running"):
        return 0
    while True:
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
