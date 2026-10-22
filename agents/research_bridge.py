"""Bridge: DeepGraph Insights → EvoScientist Research Proposals.

Takes a DeepGraph insight (with its supporting evidence from the knowledge graph)
and produces an enriched research proposal that EvoScientist can execute as a
full research plan with experiment design, code, and analysis.
"""

import json
import os
import subprocess
import time
from pathlib import Path



ACTIVE_SESSION_LOG_WINDOW_SECONDS = 15 * 60
TERMINAL_LOG_MARKERS = (
    "BadRequestError:",
    "Traceback (most recent call last)",
    "UnicodeEncodeError:",
    "invalid_request_error",
    "Error code:",
)


def _pid_file_path(workdir: Path) -> Path:
    return workdir / "evoscientist.pid"


def _read_session_pid(workdir: Path) -> int | None:
    pid_path = _pid_file_path(workdir)
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_is_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        try:
            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return f'"{pid}"' in proc.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def write_session_pid(workdir: Path, pid: int) -> None:
    _pid_file_path(workdir).write_text(f"{pid}\n", encoding="utf-8")


def _log_has_terminal_error(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    except OSError:
        return False
    return any(marker in tail for marker in TERMINAL_LOG_MARKERS)


def active_research_session(workdir: str | Path, *, log_window_seconds: int = ACTIVE_SESSION_LOG_WINDOW_SECONDS) -> dict | None:
    workdir = Path(workdir)
    if not workdir.exists():
        return None
    final_report = workdir / "final_report.md"
    if final_report.exists():
        return None
    pid = _read_session_pid(workdir)
    if _pid_is_running(pid):
        return {"workdir": str(workdir), "pid": pid, "active": True, "reason": "pid_running"}
    log_path = workdir / "evoscientist.log"
    if log_path.exists():
        age_seconds = max(0.0, time.time() - log_path.stat().st_mtime)
        if pid and _log_has_terminal_error(log_path):
            return None
        if age_seconds <= max(1, log_window_seconds):
            return {
                "workdir": str(workdir),
                "pid": pid,
                "active": True,
                "reason": "recent_log_activity_after_launcher_exit" if pid else "recent_log_activity",
                "log_age_seconds": age_seconds,
            }
    return None


def _json_load(value, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _json_list(value) -> list:
    loaded = _json_load(value, [])
    return loaded if isinstance(loaded, list) else []


def _json_dict(value) -> dict:
    loaded = _json_load(value, {})
    return loaded if isinstance(loaded, dict) else {}


def _dedupe(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def get_research_status(workdir: str) -> dict:
    """Check the status of an EvoScientist research session."""
    workdir = Path(workdir)
    status = {"workdir": str(workdir), "exists": workdir.exists()}

    if not workdir.exists():
        return status

    pid = _read_session_pid(workdir)
    status["pid"] = pid
    status["pid_running"] = _pid_is_running(pid)
    active = active_research_session(workdir)
    status["active"] = bool(active)
    if active:
        status["active_reason"] = active.get("reason")
        if active.get("log_age_seconds") is not None:
            status["log_age_seconds"] = round(float(active["log_age_seconds"]), 2)

    # Check for output files
    for fname in ["todos.md", "plan.md", "final_report.md", "research_proposal.md"]:
        fpath = workdir / fname
        if fpath.exists():
            status[fname] = {
                "exists": True,
                "size": fpath.stat().st_size,
                "preview": fpath.read_text(encoding="utf-8")[:500],
            }

    # Check log
    log_path = workdir / "evoscientist.log"
    if log_path.exists():
        content = log_path.read_text(encoding="utf-8")
        status["log_lines"] = len(content.split("\n"))
        status["log_tail"] = content[-1000:]

    return status
