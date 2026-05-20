"""Monitor live experiment progress from artifacts and process signals."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from db import database as db


@dataclass
class ProgressReport:
    role: str = "ExperimentProgressMonitor"
    run_id: int = 0
    workdir: str = ""
    alive: bool = False
    subprocess_running: bool = False
    subprocess_command: str | None = None
    elapsed_seconds: float | None = None
    time_budget_seconds: int | None = None
    predictions_path: str | None = None
    predictions_lines: int = 0
    predictions_bytes: int = 0
    predictions_age_seconds: float | None = None
    predictions_growth_lines_per_min: float | None = None
    expected_lines_per_round: int | None = None
    coverage_pct: float | None = None
    datasets_seen: list[str] = field(default_factory=list)
    methods_seen: list[str] = field(default_factory=list)
    iteration_packets: list[dict[str, Any]] = field(default_factory=list)
    crash_streak: int = 0
    status: str = "unknown"  # running | stale | idle | complete | crashed
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def _scan_predictions_meta(path: Path, *, sample_limit: int = 5000) -> tuple[set[str], set[str]]:
    datasets: set[str] = set()
    methods: set[str] = set()
    if not path.exists():
        return datasets, methods
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for idx, line in enumerate(handle):
                if idx >= sample_limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("dataset"):
                    datasets.add(str(row["dataset"]))
                if row.get("method"):
                    methods.add(str(row["method"]))
    except OSError:
        pass
    return datasets, methods


def _expected_lines_from_config(run_config: dict[str, Any]) -> int | None:
    targets = run_config.get("targets") if isinstance(run_config.get("targets"), list) else []
    if not targets:
        return None
    per_target = 0
    for target in targets:
        if not isinstance(target, dict):
            continue
        try:
            per_target += int(target.get("max_eval_examples") or 0)
        except (TypeError, ValueError):
            continue
    env_report_path = None
    try:
        seeds = int((run_config.get("seeds") or 3))
    except (TypeError, ValueError):
        seeds = 3
    methods = run_config.get("methods") if isinstance(run_config.get("methods"), list) else []
    method_count = max(len(methods), 3)
    if per_target <= 0:
        return None
    return per_target * max(seeds, 1) * method_count


def _train_subprocess_running(code_dir: Path | None) -> tuple[bool, str | None]:
    if not code_dir:
        return False, None
    cwd = str(code_dir.resolve())
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,cmd"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False, None
    for line in (result.stdout or "").splitlines():
        if "train.py" not in line:
            continue
        if cwd in line or "deepgraph_ideas" in line:
            return True, line.strip()[:200]
    return False, None


def _load_iteration_packets(workdir: Path) -> list[dict[str, Any]]:
    packet_dir = workdir / "results" / "iteration_packets"
    if not packet_dir.is_dir():
        return []
    packets: list[dict[str, Any]] = []
    for path in sorted(packet_dir.glob("*.json")):
        payload = _read_json(path)
        if payload:
            payload["packet_file"] = path.name
            packets.append(payload)
    return packets


def _crash_streak(packets: list[dict[str, Any]]) -> int:
    streak = 0
    for packet in reversed(packets):
        if packet.get("status") == "crash":
            streak += 1
        else:
            break
    return streak


def inspect_run(run_id: int, *, previous_lines: int | None = None, previous_at: float | None = None) -> ProgressReport:
    """Build a progress snapshot for an experiment run."""
    row = db.fetchone("SELECT * FROM experiment_runs WHERE id=?", (run_id,))
    if not row:
        return ProgressReport(run_id=run_id, status="missing", notes=["experiment run not found"])

    workdir = Path(row["workdir"] or "")
    report = ProgressReport(
        run_id=run_id,
        workdir=str(workdir),
        alive=workdir.is_dir(),
        time_budget_seconds=None,
    )

    proxy = {}
    if row.get("proxy_config"):
        try:
            proxy = json.loads(row["proxy_config"]) if isinstance(row["proxy_config"], str) else dict(row["proxy_config"])
        except (TypeError, json.JSONDecodeError):
            proxy = {}
    try:
        report.time_budget_seconds = int(
            proxy.get("full_benchmark_time_budget_seconds")
            or proxy.get("time_budget_seconds")
            or 0
        ) or None
    except (TypeError, ValueError):
        report.time_budget_seconds = None

    if row.get("started_at"):
        try:
            from datetime import datetime

            started = datetime.fromisoformat(str(row["started_at"]).replace("Z", "+00:00"))
            report.elapsed_seconds = max(0.0, time.time() - started.timestamp())
        except (TypeError, ValueError):
            pass

    results_dir = workdir / "results"
    pred_path = results_dir / "raw_predictions.jsonl"
    report.predictions_path = str(pred_path) if pred_path.exists() else None
    report.predictions_lines = _count_lines(pred_path)
    if pred_path.exists():
        stat = pred_path.stat()
        report.predictions_bytes = stat.st_size
        report.predictions_age_seconds = max(0.0, time.time() - stat.st_mtime)
        if previous_lines is not None and previous_at is not None:
            delta_lines = report.predictions_lines - previous_lines
            delta_min = max((time.time() - previous_at) / 60.0, 1e-6)
            if delta_lines > 0:
                report.predictions_growth_lines_per_min = delta_lines / delta_min

    run_config = _read_json(results_dir / "run_config.json")
    report.expected_lines_per_round = _expected_lines_from_config(run_config)
    if report.expected_lines_per_round and report.predictions_lines:
        report.coverage_pct = min(
            100.0,
            100.0 * report.predictions_lines / max(report.expected_lines_per_round, 1),
        )

    datasets, methods = _scan_predictions_meta(pred_path)
    report.datasets_seen = sorted(datasets)
    report.methods_seen = sorted(methods)

    code_dir = workdir / "code" if (workdir / "code").is_dir() else None
    running, cmd = _train_subprocess_running(code_dir)
    report.subprocess_running = running
    report.subprocess_command = cmd

    report.iteration_packets = _load_iteration_packets(workdir)
    report.crash_streak = _crash_streak(report.iteration_packets)

    run_status = str(row.get("status") or "")
    if run_status in {"completed", "failed"}:
        report.status = run_status
    elif report.subprocess_running or (report.predictions_growth_lines_per_min or 0) > 1:
        report.status = "running"
    elif report.predictions_lines > 0 and (report.predictions_age_seconds or 0) > 900:
        report.status = "stale"
        report.notes.append("predictions file not updated for >15 minutes")
    elif report.predictions_lines > 0:
        report.status = "idle"
        report.notes.append("artifacts present but no active train.py process")
    elif report.crash_streak >= 2:
        report.status = "crashed"
        report.notes.append(f"{report.crash_streak} consecutive crash packets")
    else:
        report.status = "unknown"

    if report.subprocess_running:
        report.notes.append("train.py subprocess active")
    if report.coverage_pct is not None:
        report.notes.append(f"prediction coverage ~{report.coverage_pct:.0f}% of one full round")

    return report
