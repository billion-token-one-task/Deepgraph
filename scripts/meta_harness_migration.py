#!/usr/bin/env python3
"""Guarded migration planner/runner for PostgreSQL schema upgrades.

Dry-run is the default and performs no database access. Isolated restores use
the normal ``--apply`` path. A deliberately separate, one-time live-local path
exists only for the on-host ``deepgraph`` database after the web service has
been stopped and a verified custom-format backup has been recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = ROOT / "db" / "migrations"
# Ordered. A later migration is only additive on top of its predecessors.
MIGRATION_KEYS = (
    "0001_meta_harness_v1",
    "0002_topic_gate_and_frontier_authority",
)
MIGRATION_KEY = MIGRATION_KEYS[0]
MIGRATION = MIGRATIONS_DIR / f"{MIGRATION_KEY}.sql"


def migration_path(migration_key: str) -> Path:
    """Resolve one reviewed migration file by key. Unknown keys fail closed."""
    if migration_key not in MIGRATION_KEYS:
        raise SystemExit(f"unknown migration key:{migration_key}")
    path = MIGRATIONS_DIR / f"{migration_key}.sql"
    if not path.is_file():
        raise SystemExit(f"migration file is missing:{path}")
    return path
ACK = "I_UNDERSTAND_THIS_WRITES_AN_ISOLATED_RESTORE"
LIVE_LOCAL_ACK = "I_UNDERSTAND_THIS_WRITES_LIVE_LOCAL_DEEPGRAPH"
ISOLATED_NAME_MARKERS = ("test", "ci", "canary", "sandbox", "staging", "restore", "shadow")
LIVE_LOCAL_HOST = "127.0.0.1"
LIVE_LOCAL_PORT = 5433
LIVE_LOCAL_DATABASE = "deepgraph"
LIVE_LOCAL_SERVICE = "deepgraph-web.service"
LIVE_LOCAL_BACKUP_DIRECTORY = Path("/home/ec2-user")
DESTRUCTIVE_SQL = re.compile(
    r"\b(DROP|TRUNCATE|DELETE|UPDATE\s+(?!research_agendas|resource_grants|agenda_resource_ledger)|"
    r"ALTER\s+COLUMN|RENAME\s+TO)\b",
    re.IGNORECASE,
)


def _statements(sql: str) -> list[str]:
    lines = [
        line
        for line in sql.splitlines()
        if line.strip() and not line.lstrip().startswith("--")
    ]
    return [statement.strip() for statement in "\n".join(lines).split(";") if statement.strip()]


def migration_plan(migration_key: str = MIGRATION_KEY) -> dict:
    path = migration_path(migration_key)
    sql = path.read_text(encoding="utf-8")
    executable_sql = ";\n".join(_statements(sql))
    destructive = sorted(
        set(match.group(0) for match in DESTRUCTIVE_SQL.finditer(executable_sql))
    )
    return {
        "migration": str(path),
        "migration_key": migration_key,
        "sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
        "bytes": len(sql.encode("utf-8")),
        "statement_count": len(_statements(sql)),
        "destructive_tokens": destructive,
        "database_accessed": False,
    }


def _validate_isolated_url(url: str) -> None:
    parsed = urlsplit(url)
    database_name = parsed.path.lstrip("/").lower()
    if parsed.scheme not in {"postgres", "postgresql"} or not database_name:
        raise SystemExit("migration URL must name an explicit PostgreSQL database")
    if not any(marker in database_name for marker in ISOLATED_NAME_MARKERS):
        raise SystemExit(
            "refusing write: database name does not identify an isolated restore"
        )
    production_url = os.environ.get("DEEPGRAPH_DATABASE_URL", "").strip()
    if production_url and production_url == url:
        raise SystemExit("refusing write: migration URL equals DEEPGRAPH_DATABASE_URL")


def _validate_live_local_url(url: str) -> None:
    """Allow exactly the operator-approved local service database endpoint."""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise SystemExit("live-local migration URL has an invalid PostgreSQL port") from exc
    database_name = parsed.path.lstrip("/")
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise SystemExit("live-local migration URL must use PostgreSQL")
    if parsed.hostname != LIVE_LOCAL_HOST or port != LIVE_LOCAL_PORT:
        raise SystemExit(
            "refusing live-local migration: target must be "
            f"{LIVE_LOCAL_HOST}:{LIVE_LOCAL_PORT}"
        )
    if database_name != LIVE_LOCAL_DATABASE:
        raise SystemExit(
            "refusing live-local migration: database must be "
            f"{LIVE_LOCAL_DATABASE}"
        )


def _require_service_stopped() -> None:
    """Fail closed unless systemd confirms the managed writer is inactive."""
    try:
        result = subprocess.run(
            ["systemctl", "show", LIVE_LOCAL_SERVICE, "-p", "ActiveState", "--value"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SystemExit("cannot verify deepgraph-web.service state") from exc
    if result.returncode != 0 or result.stdout.strip() != "inactive":
        raise SystemExit(
            "refusing live-local migration: deepgraph-web.service must be inactive"
        )


def _verified_backup(backup_file: str, backup_sha256: str) -> dict[str, str]:
    """Verify the stated backup without accepting a disposable /tmp artifact."""
    backup_path = Path(backup_file).resolve()
    expected = backup_sha256.lower()
    if backup_path.parent != LIVE_LOCAL_BACKUP_DIRECTORY:
        raise SystemExit(
            "refusing live-local migration: backup must be a file directly in "
            f"{LIVE_LOCAL_BACKUP_DIRECTORY}"
        )
    if not backup_path.is_file():
        raise SystemExit("refusing live-local migration: backup file does not exist")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise SystemExit("live-local migration requires a 64-character backup SHA256")
    actual = hashlib.sha256(backup_path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit("refusing live-local migration: backup SHA256 mismatch")
    return {"backup_file": str(backup_path), "backup_sha256": actual}


def apply_to_isolated_restore(
    url: str,
    *,
    source_commit: str,
    migration_key: str = MIGRATION_KEY,
) -> dict:
    _validate_isolated_url(url)
    plan = migration_plan(migration_key)
    if plan["destructive_tokens"]:
        raise SystemExit(
            "refusing migration with destructive SQL: "
            + ",".join(plan["destructive_tokens"])
        )
    try:
        import psycopg
    except ImportError as exc:
        raise SystemExit("psycopg is required in the isolated CI environment") from exc
    sql = migration_path(migration_key).read_text(encoding="utf-8")
    with psycopg.connect(url, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '5s'")
            cur.execute("SET LOCAL statement_timeout = '120s'")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS deepgraph_schema_migrations (
                    migration_key TEXT PRIMARY KEY,
                    source_commit TEXT NOT NULL,
                    checksum_sha256 TEXT NOT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                ALTER TABLE deepgraph_schema_migrations
                ADD COLUMN IF NOT EXISTS checksum_sha256 TEXT
                """
            )
            cur.execute(
                """
                SELECT checksum_sha256
                FROM deepgraph_schema_migrations
                WHERE migration_key=%s
                """,
                (migration_key,),
            )
            existing = cur.fetchone()
            if existing:
                if str(existing[0]) != plan["sha256"]:
                    raise RuntimeError("migration checksum mismatch")
                conn.rollback()
                return {**plan, "status": "already_applied", "database_accessed": True}
            for statement in _statements(sql):
                cur.execute(statement)
            cur.execute(
                """
                INSERT INTO deepgraph_schema_migrations
                    (migration_key, source_commit, checksum_sha256)
                VALUES (%s, %s, %s)
                """,
                (migration_key, source_commit, plan["sha256"]),
            )
        conn.commit()
    return {**plan, "status": "applied", "database_accessed": True}


def apply_to_live_local(
    url: str,
    *,
    source_commit: str,
    backup_file: str,
    backup_sha256: str,
    migration_key: str = MIGRATION_KEY,
) -> dict:
    """Apply once to the explicitly-authorized local ``deepgraph`` database.

    This intentionally does not reuse the isolated-restore guard: its separate
    checks make a live-local operation auditable and prevent a mere database
    name change from converting a restore migration into a live write.
    """
    if os.environ.get("DEEPGRAPH_ALLOW_LIVE_LOCAL_MIGRATION") != "1":
        raise SystemExit(
            "refusing live-local migration: set "
            "DEEPGRAPH_ALLOW_LIVE_LOCAL_MIGRATION=1"
        )
    _validate_live_local_url(url)
    backup = _verified_backup(backup_file, backup_sha256)
    _require_service_stopped()
    plan = migration_plan(migration_key)
    if plan["destructive_tokens"]:
        raise SystemExit(
            "refusing migration with destructive SQL: "
            + ",".join(plan["destructive_tokens"])
        )
    try:
        import psycopg
    except ImportError as exc:
        raise SystemExit("psycopg is required for the live-local migration") from exc
    sql = migration_path(migration_key).read_text(encoding="utf-8")
    with psycopg.connect(url, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '5s'")
            cur.execute("SET LOCAL statement_timeout = '120s'")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS deepgraph_schema_migrations (
                    migration_key TEXT PRIMARY KEY,
                    source_commit TEXT NOT NULL,
                    checksum_sha256 TEXT NOT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                ALTER TABLE deepgraph_schema_migrations
                ADD COLUMN IF NOT EXISTS checksum_sha256 TEXT
                """
            )
            cur.execute(
                """
                SELECT checksum_sha256
                FROM deepgraph_schema_migrations
                WHERE migration_key=%s
                """,
                (migration_key,),
            )
            existing = cur.fetchone()
            if existing:
                if str(existing[0]) != plan["sha256"]:
                    raise RuntimeError("migration checksum mismatch")
                conn.rollback()
                return {
                    **plan,
                    **backup,
                    "status": "already_applied",
                    "safety_mode": "live_local",
                    "database_accessed": True,
                }
            for statement in _statements(sql):
                cur.execute(statement)
            cur.execute(
                """
                INSERT INTO deepgraph_schema_migrations
                    (migration_key, source_commit, checksum_sha256)
                VALUES (%s, %s, %s)
                """,
                (migration_key, source_commit, plan["sha256"]),
            )
        conn.commit()
    return {
        **plan,
        **backup,
        "status": "applied",
        "safety_mode": "live_local",
        "database_accessed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="apply to an isolated restore")
    mode.add_argument(
        "--apply-live-local",
        action="store_true",
        help="apply once to 127.0.0.1:5433/deepgraph after live-local checks",
    )
    parser.add_argument("--confirm-isolated-restore")
    parser.add_argument("--confirm-live-local-deepgraph")
    parser.add_argument("--source-commit")
    parser.add_argument("--backup-file")
    parser.add_argument("--backup-sha256")
    parser.add_argument(
        "--migration-key",
        default=MIGRATION_KEY,
        choices=list(MIGRATION_KEYS),
        help="which reviewed migration to plan or apply",
    )
    args = parser.parse_args()
    if not args.apply and not args.apply_live_local:
        print(migration_plan(args.migration_key))
        return 0
    if not args.source_commit or not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("--source-commit must be the 40-character candidate commit hash")
    url = os.environ.get("DEEPGRAPH_MIGRATION_DATABASE_URL", "").strip()
    if not url:
        parser.error("DEEPGRAPH_MIGRATION_DATABASE_URL is required")
    if args.apply_live_local:
        if args.confirm_live_local_deepgraph != LIVE_LOCAL_ACK:
            parser.error(
                "--confirm-live-local-deepgraph must equal "
                f"{LIVE_LOCAL_ACK}"
            )
        if not args.backup_file or not args.backup_sha256:
            parser.error("live-local migration requires --backup-file and --backup-sha256")
        result = apply_to_live_local(
            url,
            source_commit=args.source_commit,
            backup_file=args.backup_file,
            backup_sha256=args.backup_sha256,
            migration_key=args.migration_key,
        )
        print(result)
        return 0
    if args.confirm_isolated_restore != ACK:
        parser.error(f"--confirm-isolated-restore must equal {ACK}")
    result = apply_to_isolated_restore(
        url,
        source_commit=args.source_commit,
        migration_key=args.migration_key,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
