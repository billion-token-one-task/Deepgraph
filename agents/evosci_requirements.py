"""Central gating when strict EvoScientist mode is enabled.

When DEEPGRAPH_REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS=true:
  - novelty verification must conclude novelty_status == 'novel'
  - EvoScientist deep research must produce evoscientist_workdir/final_report.md
  - SciForge (forge + validation loop) is blocked until both hold.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from config import REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS


def evosci_binary_path() -> Path:
    """Resolve EvoSci executable (Windows Scripts\\ or POSIX bin/)."""
    home = Path.home()
    if os.name == "nt":
        win = home / "EvoScientist" / ".venv" / "Scripts" / "EvoSci.exe"
        if win.exists():
            return win
    return home / "EvoScientist" / ".venv" / "bin" / "EvoSci"


def evosci_installed() -> bool:
    return evosci_binary_path().exists()


# Flags every headless EvoScientist invocation needs: approve tool calls
# without a prompt, no thinking display, plain CLI rendering. The workdir and
# the prompt are per-call and are appended by evosci_command.
#
# These are hard-coded against a CLI this repository does not own, and they
# drifted: three call sites each passed a `--auto-mode` that the installed
# EvoScientist no longer accepts, so from some upgrade onwards *every*
# pre-insert Tier-2 review died on a usage error before doing any work, and
# reported only "evoscientist_review_failed". Keeping the command in one place
# means the next drift breaks one function, and test_evosci_command_contract
# checks these flags against the installed binary's own help output.
EVOSCI_HEADLESS_FLAGS: tuple[str, ...] = (
    "--auto-approve",
    "--no-thinking",
    "--ui",
    "cli",
)


def evosci_command(*, workdir: str | Path, prompt: str) -> list[str]:
    """Build one headless EvoScientist invocation."""

    return [
        str(evosci_binary_path()),
        "--workdir",
        str(workdir),
        *EVOSCI_HEADLESS_FLAGS,
        "-p",
        prompt,
    ]


def final_report_ready(insight: dict[str, Any] | None) -> bool:
    if not insight:
        return False
    wd = str(insight.get("evoscientist_workdir") or "").strip()
    if not wd:
        return False
    path = Path(wd) / "final_report.md"
    try:
        return path.is_file() and path.stat().st_size > 100
    except OSError:
        return False


def evosci_strict_gate_insight(insight: dict[str, Any] | None) -> dict | None:
    """If strict mode blocks progress, return an error dict for APIs; else None."""
    if not REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS:
        return None
    if not insight:
        return {"error": "Deep insight not found", "route": "blocked"}

    if not evosci_installed():
        return {
            "error": (
                "EvoScientist is required (DEEPGRAPH_REQUIRE_EVOSCIENTIST_FOR_EXPERIMENTS=true) "
                f"but EvoSci was not found at {evosci_binary_path()}"
            ),
            "route": "blocked",
        }

    novelty = (insight.get("novelty_status") or "").strip()
    if novelty in {"", "unchecked", "verifying"}:
        return {
            "error": "Complete EvoScientist novelty verification before SciForge (strict mode).",
            "route": "blocked",
        }
    if novelty == "exists":
        return {
            "error": "Insight marked as existing prior work; SciForge disabled (strict mode).",
            "route": "blocked",
        }
    if novelty == "partially_exists":
        return {
            "error": (
                "novelty_status=partially_exists: refine the insight or re-run verification; "
                "SciForge disabled until novelty is 'novel' (strict mode)."
            ),
            "route": "blocked",
        }
    if novelty != "novel":
        return {
            "error": f"Novelty status {novelty!r} is not eligible for SciForge under strict EvoScientist mode.",
            "route": "blocked",
        }

    if not final_report_ready(insight):
        return {
            "error": (
                "EvoScientist final_report.md is missing or empty. "
                "Finish deep research before SciForge (strict mode)."
            ),
            "route": "blocked",
        }

    return None
