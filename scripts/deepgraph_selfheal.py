#!/usr/bin/env python3
"""Source-controlled DeepGraph self-heal watchdog.

Replaces the untracked ``/usr/local/bin/deepgraph-selfheal.sh``. The decision
itself lives in :mod:`orchestrator.selfheal_policy`; this runner only collects
signals and applies the single action the policy returns.

Signal collection rules:

* status/count-only SQL. No business row, paper text, claim, or user data is
  read or logged.
* no credential, database URL, or host address is ever printed. The database
  password is passed to ``psql`` through the environment only.
* any signal that cannot be observed is reported as unknown, and the policy
  treats unknown as "do nothing".

Usage:
    deepgraph_selfheal.py [--dry-run] [--json]

``--dry-run`` collects signals and prints the decision without restarting
anything and without touching the cooldown state. It is the supported way to
verify a deployment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.selfheal_policy import (  # noqa: E402
    HEALTH_FAILED,
    HEALTH_OK,
    HEALTH_UNKNOWN,
    SelfHealDecision,
    SelfHealPolicy,
    SelfHealSignals,
    decide,
    next_consecutive_failures,
)


DEFAULT_RUNTIME_ENV = "/home/billion-token/Deepgraph/.env"
DEFAULT_STATE_DIR = "/var/lib/deepgraph-selfheal"
DEFAULT_WEB_SERVICE = "deepgraph-web.service"
DEFAULT_PROCESS_PATTERN = "Deepgraph/main.py"
DEFAULT_HEALTH_URL = "http://127.0.0.1:8080/api/meta"
DEFAULT_LOG = "/var/log/deepgraph-selfheal.log"
PROVIDER_ISSUE_MARKERS = (
    "cooling down",
    "auth failed (401)",
    "no llm providers",
    "额度",
)

TRUE_VALUES = {"1", "true", "yes", "on"}

# One row of counters. Every column is an aggregate; no row content is selected.
COUNTS_SQL = """
SELECT
  coalesce((
    SELECT extract(epoch FROM now() - max(created_at))::bigint
    FROM (
      SELECT max(created_at) AS created_at FROM claims
      UNION ALL SELECT max(created_at) FROM insights
      UNION ALL SELECT max(created_at) FROM results
      UNION ALL SELECT max(created_at) FROM graph_relations
    ) AS fresh
  ), -1) AS output_age_seconds,
  (SELECT count(*) FROM resource_grants
     WHERE status='active' AND expires_at > now()) AS active_grants,
  (SELECT count(*) FROM auto_research_jobs
     WHERE status IN ('verifying','researching','running_experiment')) AS running_jobs,
  (SELECT count(*) FROM auto_research_jobs
     WHERE stage='awaiting_portfolio_decision' OR status='blocked')
     AS awaiting_jobs
"""


def parse_env_file(path: str) -> dict[str, str]:
    """Read KEY=VALUE lines. Values are never logged by this module."""
    values: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env_flag(values: dict[str, str], key: str, default: bool) -> bool:
    raw = values.get(key, os.environ.get(key, ""))
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in TRUE_VALUES


def parse_counts(raw: str) -> dict[str, int] | None:
    """Parse the single ``psql -At`` row into counters, or None if unusable."""
    fields = [part.strip() for part in str(raw).strip().split("|")]
    if len(fields) != 4:
        return None
    try:
        parsed = [int(float(value)) for value in fields]
    except (TypeError, ValueError):
        return None
    age = parsed[0]
    return {
        "output_age_seconds": age if age >= 0 else -1,
        "active_grants": max(0, parsed[1]),
        "running_jobs": max(0, parsed[2]),
        "awaiting_jobs": max(0, parsed[3]),
    }


def provider_issue_in_log(tail_text: str) -> bool:
    lowered = str(tail_text).lower()
    return any(marker in lowered for marker in PROVIDER_ISSUE_MARKERS)


def _run(command: list[str], *, env: dict[str, str] | None = None, timeout: int = 20):
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
        check=False,
    )


def process_running(pattern: str) -> bool:
    try:
        return _run(["pgrep", "-f", pattern], timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def probe_health(url: str, *, timeout: int = 10) -> str:
    """Return ok / failed / unknown. A transport error is *unknown*, not failed.

    Only an answered request proves anything about the application: a 5xx or a
    connection refused on a live process is a real failure, while DNS/socket
    errors on a loopback probe are treated as an unobservable signal.
    """
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HEALTH_OK if 200 <= int(response.status) < 400 else HEALTH_FAILED
    except urllib.error.HTTPError as exc:
        return HEALTH_FAILED if int(exc.code) >= 500 else HEALTH_OK
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", "")).lower()
        if "refused" in reason or "reset" in reason or "timed out" in reason:
            return HEALTH_FAILED
        return HEALTH_UNKNOWN
    except (OSError, ValueError):
        return HEALTH_UNKNOWN


def read_counts(database_url: str, *, psql: str) -> dict[str, int] | None:
    """Run the aggregate query. Returns None when the signal is unavailable."""
    if not database_url:
        return None
    parsed = urlsplit(database_url)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        return None
    environment = dict(os.environ)
    if parsed.password:
        environment["PGPASSWORD"] = unquote(parsed.password)
    command = [
        psql,
        "-h",
        parsed.hostname,
        "-p",
        str(parsed.port or 5432),
        "-U",
        unquote(parsed.username or ""),
        "-d",
        parsed.path.lstrip("/") or "postgres",
        "-At",
        "-c",
        COUNTS_SQL,
    ]
    try:
        completed = _run(command, env=environment, timeout=25)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return parse_counts(completed.stdout)


def tail_text(path: str, *, lines: int = 200) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 64 * 1024))
            data = handle.read()
    except OSError:
        return ""
    return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-lines:])


class SelfHealState:
    """Cooldown and health-failure bookkeeping on disk."""

    def __init__(self, directory: str):
        self.path = Path(directory) / "state.json"

    def load(self) -> dict:
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def save(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)


def collect_signals(
    *,
    runtime_env: str,
    process_pattern: str,
    health_url: str,
    web_log: str,
    psql: str,
    previous_state: dict,
    now: float,
) -> SelfHealSignals:
    env_values = parse_env_file(runtime_env)
    health_status = probe_health(health_url)
    counts = read_counts(env_values.get("DEEPGRAPH_DATABASE_URL", ""), psql=psql)
    last_restart = previous_state.get("last_restart_epoch")
    since_restart = (
        int(now - float(last_restart))
        if isinstance(last_restart, (int, float)) and float(last_restart) > 0
        else None
    )
    awaiting_jobs = int((counts or {}).get("awaiting_jobs") or 0)
    running_jobs = int((counts or {}).get("running_jobs") or 0)
    age = int((counts or {}).get("output_age_seconds", -1))
    return SelfHealSignals(
        web_process_running=process_running(process_pattern),
        health_status=health_status,
        health_consecutive_failures=next_consecutive_failures(
            int(previous_state.get("health_consecutive_failures") or 0),
            health_status=health_status,
        ),
        auto_research_enabled=env_flag(
            env_values, "DEEPGRAPH_AUTO_RESEARCH_ENABLED", True
        ),
        auto_pipeline_enabled=env_flag(
            env_values, "DEEPGRAPH_AUTO_PIPELINE_ENABLED", False
        ),
        active_resource_grants=int((counts or {}).get("active_grants") or 0),
        running_jobs=running_jobs,
        awaiting_authority=bool(awaiting_jobs > 0 and running_jobs == 0),
        awaiting_authority_reasons=(
            ("portfolio_or_grant_decision_pending",) if awaiting_jobs else ()
        ),
        output_age_seconds=(age if counts is not None and age >= 0 else None),
        provider_issue=provider_issue_in_log(tail_text(web_log)),
        seconds_since_last_restart=since_restart,
    )


def ensure_dependencies(services: list[str], *, log) -> None:
    """Cheap dependency nudge. Never touches the web service itself."""
    for service in services:
        if not service:
            continue
        active = _run(["systemctl", "is-active", "--quiet", service], timeout=10)
        if active.returncode != 0:
            started = _run(["systemctl", "start", service], timeout=30)
            log(f"dependency_start service={service} rc={started.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--runtime-env",
        default=os.environ.get("DEEPGRAPH_SELFHEAL_ENV_FILE", DEFAULT_RUNTIME_ENV),
    )
    parser.add_argument(
        "--state-dir",
        default=os.environ.get("DEEPGRAPH_SELFHEAL_STATE_DIR", DEFAULT_STATE_DIR),
    )
    parser.add_argument(
        "--health-url",
        default=os.environ.get("DEEPGRAPH_SELFHEAL_HEALTH_URL", DEFAULT_HEALTH_URL),
    )
    parser.add_argument(
        "--process-pattern",
        default=os.environ.get(
            "DEEPGRAPH_SELFHEAL_PROCESS_PATTERN", DEFAULT_PROCESS_PATTERN
        ),
    )
    parser.add_argument(
        "--web-service",
        default=os.environ.get("DEEPGRAPH_SELFHEAL_WEB_SERVICE", DEFAULT_WEB_SERVICE),
    )
    parser.add_argument(
        "--web-log",
        default=os.environ.get(
            "DEEPGRAPH_SELFHEAL_WEB_LOG",
            "/home/billion-token/Deepgraph/logs/web-systemd.log",
        ),
    )
    parser.add_argument(
        "--log-file",
        default=os.environ.get("DEEPGRAPH_SELFHEAL_LOG", DEFAULT_LOG),
    )
    parser.add_argument("--psql", default=os.environ.get("DEEPGRAPH_PSQL", "psql"))
    parser.add_argument(
        "--dependency-service",
        action="append",
        default=None,
        help="service to keep started (repeatable); never the web service",
    )
    parser.add_argument("--stall-seconds", type=int, default=45 * 60)
    parser.add_argument("--cooldown-seconds", type=int, default=30 * 60)
    parser.add_argument("--health-failure-threshold", type=int, default=3)
    args = parser.parse_args()

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
        try:
            with open(args.log_file, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass
        if not args.json:
            print(line, flush=True)

    dependencies = (
        args.dependency_service
        if args.dependency_service is not None
        else ["deepgraph-postgres.service", "grobid-gateway.socket"]
    )
    if not args.dry_run:
        ensure_dependencies(
            [service for service in dependencies if service != args.web_service],
            log=log,
        )

    state = SelfHealState(args.state_dir)
    previous = state.load()
    now = time.time()
    signals = collect_signals(
        runtime_env=args.runtime_env,
        process_pattern=args.process_pattern,
        health_url=args.health_url,
        web_log=args.web_log,
        psql=args.psql,
        previous_state=previous,
        now=now,
    )
    policy = SelfHealPolicy(
        stall_seconds=args.stall_seconds,
        cooldown_seconds=args.cooldown_seconds,
        health_failure_threshold=args.health_failure_threshold,
    )
    decision: SelfHealDecision = decide(signals, policy=policy)

    report = {
        "decision": decision.action,
        "reason_code": decision.reason_code,
        "details": decision.details,
        "signals": asdict(signals),
        "dry_run": bool(args.dry_run),
    }
    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    log(f"decision={decision.action} reason={decision.reason_code}")

    if args.dry_run:
        return 0

    next_state = dict(previous)
    next_state["health_consecutive_failures"] = signals.health_consecutive_failures
    next_state["last_decision"] = decision.reason_code
    next_state["last_run_epoch"] = int(now)
    if decision.should_restart:
        restarted = _run(["systemctl", "restart", args.web_service], timeout=120)
        log(
            f"restart service={args.web_service} rc={restarted.returncode} "
            f"reason={decision.reason_code}"
        )
        next_state["last_restart_epoch"] = int(now)
        next_state["health_consecutive_failures"] = 0
    state.save(next_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
