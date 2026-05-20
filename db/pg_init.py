#!/usr/bin/env python3
"""Initialize PostgreSQL schema from schema_postgres.sql (run once per database).

Usage:
  export DEEPGRAPH_DATABASE_URL=postgresql://user:pass@localhost:5432/deepgraph
  python -m db.pg_init

When DEEPGRAPH_DATABASE_URL is set, the application uses PostgreSQL via psycopg (see db/database.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import psycopg
except ImportError:
    print("Install psycopg: pip install 'psycopg[binary]>=3.1'", file=sys.stderr)
    raise


def main() -> None:
    url = os.getenv("DEEPGRAPH_DATABASE_URL", "").strip()
    if not url:
        print("Set DEEPGRAPH_DATABASE_URL", file=sys.stderr)
        sys.exit(1)
    schema_path = Path(__file__).with_name("schema_postgres.sql")
    if not schema_path.is_file():
        print(f"Missing {schema_path}", file=sys.stderr)
        sys.exit(1)
    sql_lines = []
    for line in schema_path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("--"):
            continue
        sql_lines.append(line)
    sql_text = "\n".join(sql_lines)
    # Split on semicolons — crude but works for our DDL (no ; inside strings)
    statements = [s.strip() for s in sql_text.split(";") if s.strip()]
    with psycopg.connect(url, autocommit=True) as conn:
        with conn.cursor() as cur:
            pending = statements
            for _ in range(8):
                next_pending = []
                progress = 0
                for stmt in pending:
                    try:
                        cur.execute(stmt)
                        progress += 1
                    except Exception as e:
                        # Idempotent re-runs: skip "already exists"
                        if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                            progress += 1
                            continue
                        next_pending.append((stmt, e))
                if not next_pending:
                    break
                if progress == 0:
                    stmt, err = next_pending[0]
                    print(f"Statement failed: {stmt[:120]}... Error: {err}", file=sys.stderr)
                    raise err
                pending = [stmt for stmt, _ in next_pending]
    print("PostgreSQL schema applied OK.", flush=True)


if __name__ == "__main__":
    main()
