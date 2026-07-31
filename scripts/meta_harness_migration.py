#!/usr/bin/env python3
"""Guarded migration planner/runner for an isolated PostgreSQL restore.

Dry-run is the default and performs no network or database access. Applying the
migration requires an explicit acknowledgement, a dedicated environment
variable, an isolated-looking database name, and a source commit supplied by
the operator.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "db" / "migrations" / "0001_meta_harness_v1.sql"
MIGRATION_KEY = "0001_meta_harness_v1"
ACK = "I_UNDERSTAND_THIS_WRITES_AN_ISOLATED_RESTORE"
ISOLATED_NAME_MARKERS = ("test", "ci", "canary", "sandbox", "staging", "restore", "shadow")
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


def migration_plan() -> dict:
    sql = MIGRATION.read_text(encoding="utf-8")
    executable_sql = ";\n".join(_statements(sql))
    destructive = sorted(
        set(match.group(0) for match in DESTRUCTIVE_SQL.finditer(executable_sql))
    )
    return {
        "migration": str(MIGRATION),
        "migration_key": MIGRATION_KEY,
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


def apply_to_isolated_restore(url: str, *, source_commit: str) -> dict:
    _validate_isolated_url(url)
    plan = migration_plan()
    if plan["destructive_tokens"]:
        raise SystemExit(
            "refusing migration with destructive SQL: "
            + ",".join(plan["destructive_tokens"])
        )
    try:
        import psycopg
    except ImportError as exc:
        raise SystemExit("psycopg is required in the isolated CI environment") from exc
    sql = MIGRATION.read_text(encoding="utf-8")
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
                (MIGRATION_KEY,),
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
                (MIGRATION_KEY, source_commit, plan["sha256"]),
            )
        conn.commit()
    return {**plan, "status": "applied", "database_accessed": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm-isolated-restore")
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    if not args.apply:
        print(migration_plan())
        return 0
    if args.confirm_isolated_restore != ACK:
        parser.error(f"--confirm-isolated-restore must equal {ACK}")
    if not args.source_commit or not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("--source-commit must be the 40-character candidate commit hash")
    url = os.environ.get("DEEPGRAPH_MIGRATION_DATABASE_URL", "").strip()
    if not url:
        parser.error("DEEPGRAPH_MIGRATION_DATABASE_URL is required")
    result = apply_to_isolated_restore(url, source_commit=args.source_commit)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
