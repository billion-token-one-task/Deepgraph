"""Database connection and helpers.

Runtime: set DEEPGRAPH_DATABASE_URL to use PostgreSQL (psycopg3); otherwise SQLite (DEEPGRAPH_DB_PATH).
Application SQL uses ``?`` placeholders; they are translated to ``%s`` for PostgreSQL.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from config import DB_PATH, DATABASE_URL, PGVECTOR_EMBEDDING_DIM
from ingestion.arxiv_ids import arxiv_base_id
from db.sql_dialect import to_postgres

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None
    dict_row = None  # type: ignore[misc, assignment]

_local = threading.local()
_pg_init_lock = threading.Lock()
_pg_init_done = False
_backend_notice_lock = threading.Lock()
_backend_notice_emitted = False


def _use_pg() -> bool:
    return bool((DATABASE_URL or "").strip())


def _adapt_sql(sql: str) -> str:
    if not _use_pg():
        return sql
    return to_postgres(sql)


def _redact_database_url(url: str) -> str:
    parts = urlsplit((url or "").strip())
    if not parts.scheme:
        return ""
    netloc = parts.hostname or ""
    if parts.username:
        netloc = parts.username
        if parts.password:
            netloc += ":***"
        if parts.hostname:
            netloc += f"@{parts.hostname}"
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def describe_backend() -> dict[str, Any]:
    backend = "postgresql" if _use_pg() else "sqlite"
    info: dict[str, Any] = {
        "backend": backend,
        "database_url_configured": _use_pg(),
        "sqlite_path": str(DB_PATH),
        "sqlite_exists": DB_PATH.exists(),
    }
    if _use_pg():
        info["target"] = _redact_database_url(DATABASE_URL)
        info["shadow_sqlite_path"] = str(DB_PATH)
        info["shadow_sqlite_exists"] = DB_PATH.exists()
    else:
        info["target"] = str(DB_PATH)
    return info


def _emit_backend_notice_once() -> None:
    global _backend_notice_emitted
    if _backend_notice_emitted:
        return
    with _backend_notice_lock:
        if _backend_notice_emitted:
            return
        info = describe_backend()
        print(f"[DB] Backend={info['backend']} target={info['target']}", flush=True)
        if info["backend"] == "postgresql" and info.get("shadow_sqlite_exists"):
            print(
                f"[DB] WARNING: legacy SQLite file still exists at {info['shadow_sqlite_path']} "
                "but runtime is using PostgreSQL. Do not use the SQLite file as source of truth.",
                flush=True,
            )
        _backend_notice_emitted = True


def _pg_connect():
    if psycopg is None or dict_row is None:
        raise RuntimeError("PostgreSQL requested but psycopg is not installed. pip install 'psycopg[binary]>=3.1'")
    return psycopg.connect(DATABASE_URL.strip(), row_factory=dict_row, autocommit=False)


def get_conn():
    """Thread-local connection (SQLite or PostgreSQL)."""
    if _use_pg():
        if not hasattr(_local, "pg_conn") or _local.pg_conn is None:
            _local.pg_conn = _pg_connect()
        return _local.pg_conn
    sc = getattr(_local, "sqlite_conn", None)
    if sc is not None:
        try:
            sc.execute("SELECT 1")
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            sc = None
    if sc is None:
        _local.sqlite_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.sqlite_conn.row_factory = sqlite3.Row
        _local.sqlite_conn.execute("PRAGMA journal_mode=WAL")
        _local.sqlite_conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = _local.sqlite_conn  # backward compat: tests patch _local.conn
    return _local.sqlite_conn


def _apply_postgres_schema_file() -> None:
    schema_path = Path(__file__).with_name("schema_postgres.sql")
    if not schema_path.is_file():
        raise FileNotFoundError(schema_path)
    sql_text = schema_path.read_text(encoding="utf-8")
    statements = [s.strip() for s in sql_text.split(";") if s.strip() and not s.strip().startswith("--")]
    conn = get_conn()
    with conn.cursor() as cur:
        # Startup should not block indefinitely on existing hot tables when the
        # schema is already mostly present. Best-effort index creation is enough.
        cur.execute("SET lock_timeout = '5s'")
        cur.execute("SET statement_timeout = '60s'")
        for stmt in statements:
            cur.execute("SAVEPOINT schema_stmt")
            try:
                cur.execute(stmt)
                cur.execute("RELEASE SAVEPOINT schema_stmt")
            except Exception as e:
                msg = str(e).lower()
                normalized = " ".join(stmt.lower().split())
                best_effort_stmt = (
                    normalized.startswith("create index if not exists")
                    or normalized.startswith("create unique index if not exists")
                    or (
                        " if not exists " in f" {normalized} "
                        and (
                            normalized.startswith("alter table ")
                            or normalized.startswith("create table ")
                            or normalized.startswith("create extension ")
                        )
                    )
                )
                if (
                    "already exists" in msg
                    or "duplicate" in msg
                    or "extension" in msg
                    or ("lock timeout" in msg and best_effort_stmt)
                    or ("canceling statement due to lock timeout" in msg and best_effort_stmt)
                ):
                    cur.execute("ROLLBACK TO SAVEPOINT schema_stmt")
                    cur.execute("RELEASE SAVEPOINT schema_stmt")
                    if "lock timeout" in msg and best_effort_stmt:
                        print(f"[DB] Skipping locked startup schema statement: {stmt[:120]}", flush=True)
                    continue
                cur.execute("ROLLBACK TO SAVEPOINT schema_stmt")
                cur.execute("RELEASE SAVEPOINT schema_stmt")
                raise
    conn.commit()


def _table_exists(name: str) -> bool:
    if _use_pg():
        row = fetchone(
            """SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = ?
            ) AS ok""",
            (name,),
        )
        return bool(row and row.get("ok"))
    row = get_conn().execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _column_names(table: str) -> set[str]:
    if _use_pg():
        rows = fetchall(
            """SELECT column_name AS name FROM information_schema.columns
               WHERE table_schema = 'public' AND table_name = ?""",
            (table,),
        )
        return {r["name"] for r in rows}
    rows = get_conn().execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"] for row in rows}


def _execute_startup_statement(
    conn,
    stmt: str,
    params: tuple = (),
    *,
    best_effort_if_locked: bool = False,
) -> bool:
    if not _use_pg():
        conn.execute(stmt, params)
        return True
    with conn.cursor() as cur:
        cur.execute("SAVEPOINT startup_stmt")
        try:
            cur.execute(stmt, params)
            cur.execute("RELEASE SAVEPOINT startup_stmt")
            return True
        except Exception as e:
            msg = str(e).lower()
            cur.execute("ROLLBACK TO SAVEPOINT startup_stmt")
            cur.execute("RELEASE SAVEPOINT startup_stmt")
            if best_effort_if_locked and (
                "lock timeout" in msg or "canceling statement due to lock timeout" in msg
            ):
                print(f"[DB] Skipping locked startup schema statement: {stmt[:120]}", flush=True)
                return False
            raise


def _ensure_columns(table: str, additions: dict[str, str]) -> None:
    if not _table_exists(table):
        return
    cols = _column_names(table)
    conn = get_conn()
    for name, ddl in additions.items():
        if name not in cols:
            _execute_startup_statement(
                conn,
                f"ALTER TABLE {table} ADD COLUMN {name} {ddl}",
                best_effort_if_locked=_use_pg(),
            )
    if not _use_pg():
        conn.commit()


def _ensure_legacy_migrations() -> None:
    if _use_pg():
        return
    conn = get_conn()

    if _table_exists("patterns"):
        additions = {
            "abstraction_level": "TEXT",
            "node_id": "TEXT REFERENCES taxonomy_nodes(id)",
            "source_claims": "TEXT",
        }
        _ensure_columns("patterns", additions)

    conn.execute(
        """CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT NOT NULL REFERENCES taxonomy_nodes(id),
            insight_type TEXT NOT NULL,
            title TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            evidence TEXT,
            experiment TEXT,
            impact TEXT,
            novelty_score REAL DEFAULT 0,
            feasibility_score REAL DEFAULT 0,
            supporting_papers TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_insights_node ON insights(node_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_insights_type ON insights(insight_type)")
    if _table_exists("patterns"):
        conn.execute("CREATE INDEX IF NOT EXISTS idx_patterns_node ON patterns(node_id)")
    conn.commit()


def _ensure_vnext_migrations() -> None:
    conn = get_conn()

    _ensure_columns(
        "deep_insights",
        {
            "signal_mix": "TEXT",
            "mechanism_type": "TEXT",
            "evidence_packet": "TEXT",
            "evidence_plan": "TEXT",
            "experimentability": "TEXT",
            "resource_class": "TEXT DEFAULT 'cpu'",
            "submission_status": "TEXT DEFAULT 'not_started'",
            "workspace_root": "TEXT",
            "experiment_root": "TEXT",
            "plan_root": "TEXT",
            "paper_root": "TEXT",
            "canonical_run_id": "INTEGER",
        },
    )
    _ensure_columns(
        "experiment_runs",
        {
            "resource_class": "TEXT DEFAULT 'cpu'",
            "submission_bundle_id": "INTEGER",
        },
    )
    _ensure_columns(
        "auto_research_jobs",
        {
            "resource_class": "TEXT DEFAULT 'cpu'",
            "scheduler_priority": "INTEGER DEFAULT 0",
            "assigned_worker": "TEXT",
            "artifact_bundle_id": "INTEGER",
        },
    )

    _execute_startup_statement(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_deep_insights_mechanism ON deep_insights(mechanism_type)",
        best_effort_if_locked=_use_pg(),
    )
    _execute_startup_statement(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_deep_insights_resource ON deep_insights(resource_class)",
        best_effort_if_locked=_use_pg(),
    )
    _execute_startup_statement(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_deep_insights_canonical_run ON deep_insights(canonical_run_id)",
        best_effort_if_locked=_use_pg(),
    )
    _execute_startup_statement(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_auto_research_jobs_resource ON auto_research_jobs(resource_class)",
        best_effort_if_locked=_use_pg(),
    )
    conn.commit()


def _ensure_grounding_schema() -> None:
    """Add cite-and-verify columns for claims and results (existing DBs)."""
    _ensure_columns(
        "claims",
        {
            "source_quote": "TEXT",
            "char_start": "INTEGER",
            "char_end": "INTEGER",
            "grounding_status": "TEXT",
            "grounding_score": "REAL",
            "grounding_source_field": "TEXT",
        },
    )
    _ensure_columns(
        "results",
        {
            "source_quote": "TEXT",
            "char_start": "INTEGER",
            "char_end": "INTEGER",
            "grounding_status": "TEXT",
            "grounding_score": "REAL",
            "grounding_source_field": "TEXT",
        },
    )
    get_conn().commit()


def _ensure_claim_dedup_schema() -> None:
    _ensure_columns(
        "claims",
        {
            "claim_key": "TEXT",
            "source_paper_ids": "TEXT",
            "source_node_ids": "TEXT",
        },
    )
    _ensure_columns(
        "experimental_claims",
        {
            "source_paper_ids": "TEXT",
            "source_node_ids": "TEXT",
        },
    )
    _ensure_columns(
        "results",
        {
            "result_key": "TEXT",
        },
    )
    conn = get_conn()
    _execute_startup_statement(
        conn,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_claims_claim_key ON claims(claim_key)",
        best_effort_if_locked=_use_pg(),
    )
    _execute_startup_statement(
        conn,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_results_result_key ON results(result_key)",
        best_effort_if_locked=_use_pg(),
    )
    if _use_pg():
        try:
            _execute_startup_statement(
                conn,
                f"ALTER TABLE claims ADD COLUMN IF NOT EXISTS embedding_vector vector({PGVECTOR_EMBEDDING_DIM})",
                best_effort_if_locked=True,
            )
            _execute_startup_statement(
                conn,
                "CREATE INDEX IF NOT EXISTS idx_claims_embedding_vector "
                "ON claims USING hnsw (embedding_vector vector_cosine_ops)",
                best_effort_if_locked=True,
            )
        except Exception:
            pass
    conn.commit()


def _ensure_pipeline_event_schema() -> None:
    conn = get_conn()
    if _use_pg():
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_events (
                id BIGSERIAL PRIMARY KEY,
                event_type TEXT NOT NULL,
                entity_type TEXT,
                entity_id TEXT,
                dedupe_key TEXT,
                payload TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_event_consumers (
                consumer_name TEXT PRIMARY KEY,
                last_event_id BIGINT DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    else:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                entity_type TEXT,
                entity_id TEXT,
                dedupe_key TEXT,
                payload TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_event_consumers (
                consumer_name TEXT PRIMARY KEY,
                last_event_id INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    _execute_startup_statement(
        conn,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_pipeline_events_dedupe ON pipeline_events(dedupe_key)",
        best_effort_if_locked=_use_pg(),
    )
    _execute_startup_statement(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_pipeline_events_type_id ON pipeline_events(event_type, id)",
        best_effort_if_locked=_use_pg(),
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_stage_checkpoints (
            paper_id TEXT NOT NULL REFERENCES papers(id),
            stage TEXT NOT NULL,
            payload TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (paper_id, stage)
        )
        """
    )
    _execute_startup_statement(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_paper_stage_checkpoints_stage ON paper_stage_checkpoints(stage)",
        best_effort_if_locked=_use_pg(),
    )
    conn.commit()


def _sync_postgres_sequences() -> None:
    """Align BIGSERIAL/serial sequences after bulk imports from SQLite."""
    if not _use_pg():
        return
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    table_name,
                    column_name,
                    pg_get_serial_sequence(format('%I.%I', table_schema, table_name), column_name) AS sequence_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND column_default LIKE 'nextval(%'
                """
            )
            rows = cur.fetchall()

        with conn.cursor() as cur:
            for row in rows:
                table_name = row["table_name"] if isinstance(row, dict) else row[0]
                column_name = row["column_name"] if isinstance(row, dict) else row[1]
                sequence_name = row["sequence_name"] if isinstance(row, dict) else row[2]
                if not sequence_name:
                    continue

                safe_table = str(table_name).replace('"', '""')
                safe_column = str(column_name).replace('"', '""')
                cur.execute(f'SELECT MAX("{safe_column}") AS max_id FROM "{safe_table}"')
                max_row = cur.fetchone()
                max_id = (max_row["max_id"] if isinstance(max_row, dict) else max_row[0]) or 0
                if max_id:
                    cur.execute("SELECT setval(%s, %s, true)", (sequence_name, int(max_id)))
                else:
                    cur.execute("SELECT setval(%s, %s, false)", (sequence_name, 1))
    except Exception as e:
        if "lock timeout" in str(e).lower():
            print("[DB] Skipping startup sequence sync due to lock timeout", flush=True)
            try:
                conn.rollback()
            except Exception:
                pass
            return
        raise


def init_db():
    if _use_pg():
        global _pg_init_done
        if _pg_init_done:
            _emit_backend_notice_once()
            return
        with _pg_init_lock:
            if _pg_init_done:
                _emit_backend_notice_once()
                return
            conn = get_conn()
            conn.execute("SET lock_timeout = '5s'")
            conn.execute("SET statement_timeout = '60s'")
            try:
                _apply_postgres_schema_file()
                _ensure_vnext_migrations()
                _ensure_grounding_schema()
                schema_feedback = Path(__file__).parent / "schema_insight_feedback.sql"
                if schema_feedback.exists():
                    # insight_feedback may contain SQLite-only fragments; apply best-effort
                    try:
                        for stmt in schema_feedback.read_text().split(";"):
                            s = stmt.strip()
                            if s and not s.startswith("--"):
                                try:
                                    get_conn().execute(s)
                                except Exception:
                                    pass
                        get_conn().commit()
                    except Exception:
                        pass
                _ensure_insight_feedback_schema()
                _ensure_papers_checkpoint_columns()
                _ensure_claim_dedup_schema()
                _ensure_pipeline_event_schema()
                _sync_postgres_sequences()
                get_conn().commit()
                _pg_init_done = True
            finally:
                try:
                    conn.execute("SET lock_timeout = '0'")
                    conn.execute("SET statement_timeout = '0'")
                    conn.commit()
                except Exception:
                    pass
        _emit_backend_notice_once()
        return

    conn = get_conn()
    _ensure_legacy_migrations()
    schema_path = Path(__file__).parent / "schema.sql"
    conn.executescript(schema_path.read_text())
    schema_v2_path = Path(__file__).parent / "schema_v2.sql"
    if schema_v2_path.exists():
        conn.executescript(schema_v2_path.read_text())
    _ensure_vnext_migrations()
    _ensure_grounding_schema()
    schema_feedback = Path(__file__).parent / "schema_insight_feedback.sql"
    if schema_feedback.exists():
        conn.executescript(schema_feedback.read_text())
    _ensure_insight_feedback_schema()
    _ensure_papers_checkpoint_columns()
    _ensure_claim_dedup_schema()
    _ensure_pipeline_event_schema()
    conn.commit()
    _emit_backend_notice_once()


def _ensure_papers_checkpoint_columns() -> None:
    _ensure_columns(
        "papers",
        {
            "processing_stage": "TEXT DEFAULT 'ingested'",
            "arxiv_base_id": "TEXT",
            "appendix_text": "TEXT",
            "processing_attempts": "INTEGER DEFAULT 0",
            "stage_last_error": "TEXT",
            "stage_locked_by": "TEXT",
            "stage_started_at": "TIMESTAMP",
            "stage_completed_at": "TIMESTAMP",
        },
    )
    conn = get_conn()
    _execute_startup_statement(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_papers_arxiv_base ON papers(arxiv_base_id)",
        best_effort_if_locked=_use_pg(),
    )
    _execute_startup_statement(
        conn,
        "CREATE INDEX IF NOT EXISTS idx_papers_processing_stage ON papers(processing_stage)",
        best_effort_if_locked=_use_pg(),
    )
    conn.commit()


def _ensure_insight_feedback_schema() -> None:
    """Provenance + outcome columns for insights / deep_insights (existing DBs)."""
    _ensure_columns(
        "insights",
        {
            "generation_run_id": "TEXT",
            "source_signal_ids": "TEXT",
            "source_paper_ids": "TEXT",
            "source_node_ids": "TEXT",
            "prompt_version": "TEXT",
            "model_version": "TEXT",
            "exemplars_used": "TEXT",
            "token_cost_usd": "REAL",
            "wall_clock_seconds": "REAL",
            "outcome": "TEXT DEFAULT 'pending'",
            "outcome_reason": "TEXT",
            "outcome_updated_at": "TIMESTAMP",
            "human_score": "REAL",
            "human_notes": "TEXT",
        },
    )
    _ensure_columns(
        "deep_insights",
        {
            "generation_run_id": "TEXT",
            "source_signal_ids": "TEXT",
            "source_paper_ids": "TEXT",
            "prompt_version": "TEXT",
            "model_version": "TEXT",
            "exemplars_used": "TEXT",
            "token_cost_usd": "REAL",
            "wall_clock_seconds": "REAL",
            "outcome": "TEXT DEFAULT 'pending'",
            "outcome_reason": "TEXT",
            "outcome_updated_at": "TIMESTAMP",
            "human_score": "REAL",
            "human_notes": "TEXT",
        },
    )
    get_conn().commit()


def execute(sql: str, params: tuple = ()):
    sql_a = _adapt_sql(sql)
    conn = get_conn()
    if _use_pg():
        cur = conn.cursor()
        cur.execute(sql_a, params or ())
        return cur
    return conn.execute(sql_a, params)


def executemany(sql: str, params_list: list):
    sql_a = _adapt_sql(sql)
    conn = get_conn()
    if _use_pg():
        cur = conn.cursor()
        cur.executemany(sql_a, params_list)
        return cur
    return conn.executemany(sql_a, params_list)


def commit():
    get_conn().commit()


def rollback():
    get_conn().rollback()


def fetchone(sql: str, params: tuple = ()) -> dict | None:
    cur = execute(sql, params)
    row = cur.fetchone()
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    return dict(row)


def fetchall(sql: str, params: tuple = ()) -> list[dict]:
    cur = execute(sql, params)
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append(r if isinstance(r, dict) else dict(r))
    return out


def insert_returning_id(sql: str, params: tuple) -> int:
    """Run INSERT ... RETURNING id (SQLite 3.35+ and PostgreSQL)."""
    row = fetchone(sql, params)
    if not row or row.get("id") is None:
        raise RuntimeError("INSERT RETURNING id produced no id")
    return int(row["id"])


def _dump_json(value: Any) -> str:
    return json.dumps(value if value is not None else [])


def _load_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _claim_dedup_key(claim: dict) -> str:
    payload = {
        "paper_id": claim.get("paper_id"),
        "claim_text": _normalize_text(claim.get("claim_text")),
        "claim_type": claim.get("claim_type"),
        "method_name": _normalize_text(claim.get("method_name")),
        "dataset_name": _normalize_text(claim.get("dataset_name")),
        "metric_name": _normalize_text(claim.get("metric_name")),
        "metric_value": claim.get("metric_value"),
        "evidence_location": _normalize_text(claim.get("evidence_location")),
        "source_quote": _normalize_text(claim.get("source_quote")),
        "char_start": claim.get("char_start"),
        "char_end": claim.get("char_end"),
        "grounding_source_field": _normalize_text(claim.get("grounding_source_field")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def emit_pipeline_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    entity_type: str | None = None,
    entity_id: str | None = None,
    dedupe_key: str | None = None,
) -> int:
    payload_json = json.dumps(payload, ensure_ascii=False, default=str)
    if dedupe_key:
        row = fetchone(
            """
            INSERT INTO pipeline_events (event_type, entity_type, entity_id, dedupe_key, payload)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(dedupe_key) DO UPDATE SET
              event_type=excluded.event_type,
              entity_type=excluded.entity_type,
              entity_id=excluded.entity_id,
              payload=excluded.payload,
              created_at=CURRENT_TIMESTAMP
            RETURNING id
            """,
            (event_type, entity_type, entity_id, dedupe_key, payload_json),
        )
    else:
        row = fetchone(
            """
            INSERT INTO pipeline_events (event_type, entity_type, entity_id, payload)
            VALUES (?, ?, ?, ?)
            RETURNING id
            """,
            (event_type, entity_type, entity_id, payload_json),
        )
    commit()
    return int(row["id"]) if row and row.get("id") is not None else 0


def fetch_pipeline_events(
    consumer_name: str,
    *,
    limit: int = 100,
    event_types: list[str] | None = None,
) -> list[dict]:
    consumer = fetchone("SELECT last_event_id FROM pipeline_event_consumers WHERE consumer_name=?", (consumer_name,))
    last_event_id = int(consumer["last_event_id"]) if consumer and consumer.get("last_event_id") is not None else 0
    params: list[Any] = [last_event_id]
    sql = "SELECT * FROM pipeline_events WHERE id > ?"
    if event_types:
        placeholders = ", ".join("?" for _ in event_types)
        sql += f" AND event_type IN ({placeholders})"
        params.extend(event_types)
    sql += " ORDER BY id ASC LIMIT ?"
    params.append(limit)
    return fetchall(sql, tuple(params))


def ack_pipeline_events(consumer_name: str, last_event_id: int) -> None:
    execute(
        """
        INSERT INTO pipeline_event_consumers (consumer_name, last_event_id)
        VALUES (?, ?)
        ON CONFLICT(consumer_name) DO UPDATE SET
          last_event_id=excluded.last_event_id,
          updated_at=CURRENT_TIMESTAMP
        """,
        (consumer_name, last_event_id),
    )
    commit()


def record_paper_checkpoint(paper_id: str, stage: str, payload: Any, *, error_message: str | None = None) -> None:
    execute(
        """
        INSERT INTO paper_stage_checkpoints (paper_id, stage, payload, error_message)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(paper_id, stage) DO UPDATE SET
          payload=excluded.payload,
          error_message=excluded.error_message,
          updated_at=CURRENT_TIMESTAMP
        """,
        (paper_id, stage, json.dumps(payload, ensure_ascii=False, default=str), error_message),
    )
    commit()


def get_paper_checkpoint(paper_id: str, stage: str) -> dict | None:
    row = fetchone("SELECT * FROM paper_stage_checkpoints WHERE paper_id=? AND stage=?", (paper_id, stage))
    if not row:
        return None
    row["payload"] = _load_json(row.get("payload"), {})
    return row


def start_paper_stage(paper_id: str, stage: str, *, worker: str | None = None) -> None:
    execute(
        """
        UPDATE papers
        SET status='processing',
            processing_stage=?,
            processing_attempts=COALESCE(processing_attempts, 0) + 1,
            stage_locked_by=?,
            stage_started_at=CURRENT_TIMESTAMP,
            stage_last_error=NULL,
            updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (stage, worker, paper_id),
    )
    commit()


def mark_paper_stage_failure(paper_id: str, stage: str, error_message: str, *, retryable: bool = True) -> None:
    execute(
        """
        UPDATE papers
        SET status=?,
            processing_stage=?,
            stage_last_error=?,
            stage_completed_at=CURRENT_TIMESTAMP,
            updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        ("failed_retryable" if retryable else "error", stage, error_message, paper_id),
    )
    record_paper_checkpoint(paper_id, stage, {"error": error_message}, error_message=error_message)


def insert_paper(paper: dict) -> str:
    base = arxiv_base_id(paper["id"])
    existing = fetchone(
        """
        SELECT id, arxiv_base_id
        FROM papers
        WHERE id=? OR arxiv_base_id=?
        ORDER BY CASE WHEN id=? THEN 0 ELSE 1 END
        LIMIT 1
        """,
        (paper["id"], base, paper["id"]),
    )
    canonical_id = existing["id"] if existing else paper["id"]
    if existing and existing["id"] != paper["id"]:
        execute(
            """
            UPDATE papers
            SET title=?,
                authors=?,
                abstract=?,
                categories=?,
                published_date=?,
                pdf_url=COALESCE(NULLIF(?, ''), pdf_url),
                arxiv_base_id=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (
                paper["title"],
                json.dumps(paper.get("authors", [])),
                paper.get("abstract", ""),
                json.dumps(paper.get("categories", [])),
                paper.get("published_date", ""),
                paper.get("pdf_url", ""),
                base,
                canonical_id,
            ),
        )
    else:
        execute(
            """INSERT INTO papers (id, title, authors, abstract, categories, published_date, pdf_url, arxiv_base_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (id) DO UPDATE SET
                 title=excluded.title,
                 authors=excluded.authors,
                 abstract=excluded.abstract,
                 categories=excluded.categories,
                 published_date=excluded.published_date,
                 pdf_url=COALESCE(NULLIF(excluded.pdf_url, ''), papers.pdf_url),
                 arxiv_base_id=excluded.arxiv_base_id,
                 updated_at=CURRENT_TIMESTAMP""",
            (
                paper["id"],
                paper["title"],
                json.dumps(paper.get("authors", [])),
                paper.get("abstract", ""),
                json.dumps(paper.get("categories", [])),
                paper.get("published_date", ""),
                paper.get("pdf_url", ""),
                base,
            ),
        )
    commit()
    emit_pipeline_event(
        "paper_ingested",
        {"paper_id": canonical_id, "raw_paper_id": paper["id"], "arxiv_base_id": base, "title": paper.get("title")},
        entity_type="paper",
        entity_id=canonical_id,
        dedupe_key=f"paper_ingested:{base}",
    )
    return canonical_id


def update_paper_text(paper_id: str, full_text: str, *, appendix_text: str = ""):
    base = arxiv_base_id(paper_id)
    execute(
        """UPDATE papers SET full_text=?, appendix_text=?, status='ingested', arxiv_base_id=COALESCE(arxiv_base_id, ?),
           processing_stage='text_ready', stage_completed_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP WHERE id=?""",
        (full_text, appendix_text, base, paper_id),
    )
    commit()
    emit_pipeline_event(
        "paper_text_ready",
        {
            "paper_id": paper_id,
            "text_length": len(full_text or ""),
            "appendix_length": len(appendix_text or ""),
        },
        entity_type="paper",
        entity_id=paper_id,
        dedupe_key=f"paper_text_ready:{paper_id}",
    )


def update_paper_processing_stage(paper_id: str, stage: str, *, worker: str | None = None) -> None:
    execute(
        """
        UPDATE papers
        SET processing_stage=?,
            stage_locked_by=?,
            stage_completed_at=CURRENT_TIMESTAMP,
            stage_last_error=NULL,
            updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (stage, worker, paper_id),
    )
    commit()


def update_paper_status(paper_id: str, status: str, error_msg: str = None, token_cost: int = 0):
    execute(
        """
        UPDATE papers
        SET status=?,
            error_msg=?,
            stage_last_error=COALESCE(?, stage_last_error),
            token_cost=token_cost+?,
            updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (status, error_msg, error_msg, token_cost, paper_id),
    )
    commit()


def insert_claim(claim: dict) -> int:
    claim_key = claim.get("claim_key") or _claim_dedup_key(claim)
    existing = fetchone("SELECT id FROM claims WHERE claim_key=?", (claim_key,))
    payload = (
        claim["paper_id"],
        claim["claim_text"],
        claim.get("claim_type"),
        claim.get("method_name"),
        claim.get("dataset_name"),
        claim.get("metric_name"),
        claim.get("metric_value"),
        claim.get("evidence_location"),
        json.dumps(claim.get("conditions", {})),
        claim.get("source_quote"),
        claim.get("char_start"),
        claim.get("char_end"),
        claim.get("grounding_status"),
        claim.get("grounding_score"),
        claim.get("grounding_source_field"),
        json.dumps(claim.get("source_paper_ids", [])),
        json.dumps(claim.get("source_node_ids", [])),
        claim_key,
    )
    if existing:
        rid = int(existing["id"])
        execute(
            """UPDATE claims
               SET paper_id=?, claim_text=?, claim_type=?, method_name=?, dataset_name=?,
                   metric_name=?, metric_value=?, evidence_location=?, conditions=?,
                   source_quote=?, char_start=?, char_end=?, grounding_status=?,
                   grounding_score=?, grounding_source_field=?, source_paper_ids=?, source_node_ids=?
               WHERE id=?""",
            payload[:-1] + (rid,),
        )
    else:
        rid = insert_returning_id(
            """INSERT INTO claims (paper_id, claim_text, claim_type, method_name, dataset_name,
               metric_name, metric_value, evidence_location, conditions,
               source_quote, char_start, char_end, grounding_status, grounding_score, grounding_source_field,
               source_paper_ids, source_node_ids, claim_key)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               RETURNING id""",
            payload,
        )
    commit()
    return rid


def insert_method(method: dict):
    execute(
        """INSERT INTO methods (name, category, description, key_innovation, first_paper_id, builds_on)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT (name) DO NOTHING""",
        (
            method["name"],
            method.get("category"),
            method.get("description"),
            method.get("key_innovation"),
            method.get("first_paper_id"),
            json.dumps(method.get("builds_on", [])),
        ),
    )
    commit()


def insert_pattern(pattern: dict) -> int:
    rid = insert_returning_id(
        """INSERT INTO patterns (pattern_text, pattern_type, domain_count, domains, claim_ids)
           VALUES (?, ?, ?, ?, ?)
           RETURNING id""",
        (
            pattern["pattern_text"],
            pattern.get("pattern_type"),
            pattern.get("domain_count", 1),
            json.dumps(pattern.get("domains", [])),
            json.dumps(pattern.get("claim_ids", [])),
        ),
    )
    commit()
    return rid


def insert_contradiction(c: dict) -> int:
    rid = insert_returning_id(
        """INSERT INTO contradictions (claim_a_id, claim_b_id, description, condition_diff, hypothesis, severity)
           VALUES (?, ?, ?, ?, ?, ?)
           RETURNING id""",
        (
            c["claim_a_id"],
            c["claim_b_id"],
            c["description"],
            c.get("condition_diff"),
            c.get("hypothesis"),
            c.get("severity", "medium"),
        ),
    )
    commit()
    return rid


def insert_gap(g: dict) -> int:
    rid = insert_returning_id(
        """INSERT INTO gaps (problem_pattern_id, solution_pattern_id, gap_description,
           missing_domain, evidence_papers, research_proposal, value_score)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           RETURNING id""",
        (
            g.get("problem_pattern_id"),
            g.get("solution_pattern_id"),
            g["gap_description"],
            g.get("missing_domain"),
            json.dumps(g.get("evidence_papers", [])),
            g.get("research_proposal"),
            g.get("value_score"),
        ),
    )
    commit()
    return rid


def upsert_paper_insight(paper_id: str, insight: dict):
    execute(
        """INSERT INTO paper_insights
           (paper_id, plain_summary, problem_statement, approach_summary, work_type,
            key_findings, limitations, open_questions)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(paper_id) DO UPDATE SET
             plain_summary=excluded.plain_summary,
             problem_statement=excluded.problem_statement,
             approach_summary=excluded.approach_summary,
             work_type=excluded.work_type,
             key_findings=excluded.key_findings,
             limitations=excluded.limitations,
             open_questions=excluded.open_questions,
             updated_at=CURRENT_TIMESTAMP""",
        (
            paper_id,
            insight.get("plain_summary"),
            insight.get("problem_statement"),
            insight.get("approach_summary"),
            insight.get("work_type"),
            _dump_json(insight.get("key_findings", [])),
            _dump_json(insight.get("limitations", [])),
            _dump_json(insight.get("open_questions", [])),
        ),
    )
    commit()


def get_paper_insight(paper_id: str) -> dict | None:
    row = fetchone("SELECT * FROM paper_insights WHERE paper_id=?", (paper_id,))
    if not row:
        return None
    row["key_findings"] = _load_json(row.get("key_findings"), [])
    row["limitations"] = _load_json(row.get("limitations"), [])
    row["open_questions"] = _load_json(row.get("open_questions"), [])
    return row


def upsert_node_summary(summary: dict):
    execute(
        """INSERT INTO node_summaries
           (node_id, audience, overview, why_it_matters, what_people_are_building,
            common_patterns, common_methods, common_datasets, current_gaps,
            starter_questions, generated_from_papers, paper_count, result_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(node_id) DO UPDATE SET
             audience=excluded.audience,
             overview=excluded.overview,
             why_it_matters=excluded.why_it_matters,
             what_people_are_building=excluded.what_people_are_building,
             common_patterns=excluded.common_patterns,
             common_methods=excluded.common_methods,
             common_datasets=excluded.common_datasets,
             current_gaps=excluded.current_gaps,
             starter_questions=excluded.starter_questions,
             generated_from_papers=excluded.generated_from_papers,
             paper_count=excluded.paper_count,
             result_count=excluded.result_count,
             generated_at=CURRENT_TIMESTAMP,
             updated_at=CURRENT_TIMESTAMP""",
        (
            summary["node_id"],
            summary.get("audience", "general"),
            summary.get("overview"),
            summary.get("why_it_matters"),
            _dump_json(summary.get("what_people_are_building", [])),
            _dump_json(summary.get("common_patterns", [])),
            _dump_json(summary.get("common_methods", [])),
            _dump_json(summary.get("common_datasets", [])),
            _dump_json(summary.get("current_gaps", [])),
            _dump_json(summary.get("starter_questions", [])),
            _dump_json(summary.get("generated_from_papers", [])),
            summary.get("paper_count", 0),
            summary.get("result_count", 0),
        ),
    )
    commit()


def get_node_summary(node_id: str) -> dict | None:
    row = fetchone("SELECT * FROM node_summaries WHERE node_id=?", (node_id,))
    if not row:
        return None
    for key in (
        "what_people_are_building",
        "common_patterns",
        "common_methods",
        "common_datasets",
        "current_gaps",
        "starter_questions",
        "generated_from_papers",
    ):
        row[key] = _load_json(row.get(key), [])
    return row


def get_stats() -> dict:
    stats = {
        "papers_total": fetchone("SELECT COUNT(*) as c FROM papers")["c"],
        "papers_processed": fetchone("SELECT COUNT(*) as c FROM papers WHERE status IN ('extracted','abstracted','reasoned')")["c"],
        "claims_total": fetchone("SELECT COUNT(*) as c FROM claims")["c"],
        "patterns_total": fetchone("SELECT COUNT(*) as c FROM patterns")["c"],
        "contradictions_total": fetchone("SELECT COUNT(*) as c FROM contradictions")["c"],
        "gaps_total": fetchone("SELECT COUNT(*) as c FROM gaps")["c"],
        "tokens_consumed": fetchone("SELECT COALESCE(SUM(token_cost),0) as c FROM papers")["c"],
    }
    try:
        stats["results_total"] = fetchone("SELECT COUNT(*) as c FROM results")["c"]
        stats["taxonomy_assignments"] = fetchone("SELECT COUNT(*) as c FROM paper_taxonomy")["c"]
        stats["matrix_gaps_total"] = fetchone("SELECT COUNT(*) as c FROM matrix_gaps")["c"]
        stats["paper_insights_total"] = fetchone("SELECT COUNT(*) as c FROM paper_insights")["c"]
        stats["node_summaries_total"] = fetchone("SELECT COUNT(*) as c FROM node_summaries")["c"]
        stats["graph_entities_total"] = fetchone("SELECT COUNT(*) as c FROM graph_entities")["c"]
        stats["graph_relations_total"] = fetchone("SELECT COUNT(*) as c FROM graph_relations")["c"]
        stats["node_graph_summaries_total"] = fetchone("SELECT COUNT(*) as c FROM node_graph_summaries")["c"]
        stats["node_opportunities_total"] = fetchone("SELECT COUNT(*) as c FROM node_opportunities")["c"]
        stats["entity_resolutions_total"] = fetchone("SELECT COUNT(*) as c FROM entity_resolutions")["c"]
        stats["merge_candidates_pending_total"] = fetchone("SELECT COUNT(*) as c FROM entity_merge_candidates WHERE status='pending'")["c"]
        stats["insights_total"] = fetchone("SELECT COUNT(*) as c FROM insights")["c"]
        stats["deep_insights_total"] = fetchone("SELECT COUNT(*) as c FROM deep_insights")["c"]
        stats["deep_insights_tier1"] = fetchone("SELECT COUNT(*) as c FROM deep_insights WHERE tier=1")["c"]
        stats["deep_insights_tier2"] = fetchone("SELECT COUNT(*) as c FROM deep_insights WHERE tier=2")["c"]
        stats["experiment_runs_total"] = fetchone("SELECT COUNT(*) as c FROM experiment_runs")["c"]
        stats["experiment_runs_completed"] = fetchone("SELECT COUNT(*) as c FROM experiment_runs WHERE status='completed'")["c"]
        stats["experimental_claims_total"] = fetchone("SELECT COUNT(*) as c FROM experimental_claims")["c"]
        stats["experiments_confirmed"] = fetchone("SELECT COUNT(*) as c FROM experiment_runs WHERE hypothesis_verdict='confirmed'")["c"]
        stats["experiments_refuted"] = fetchone("SELECT COUNT(*) as c FROM experiment_runs WHERE hypothesis_verdict='refuted'")["c"]
        stats["gpu_workers_total"] = fetchone("SELECT COUNT(*) as c FROM gpu_workers")["c"]
        stats["gpu_jobs_total"] = fetchone("SELECT COUNT(*) as c FROM gpu_jobs")["c"]
        stats["gpu_jobs_running"] = fetchone("SELECT COUNT(*) as c FROM gpu_jobs WHERE status='running'")["c"]
        stats["manuscript_runs_total"] = fetchone("SELECT COUNT(*) as c FROM manuscript_runs")["c"]
        stats["submission_bundles_total"] = fetchone("SELECT COUNT(*) as c FROM submission_bundles")["c"]
    except Exception:
        stats["results_total"] = 0
        stats["taxonomy_assignments"] = 0
        stats["matrix_gaps_total"] = 0
        stats["paper_insights_total"] = 0
        stats["node_summaries_total"] = 0
        stats["graph_entities_total"] = 0
        stats["graph_relations_total"] = 0
        stats["node_graph_summaries_total"] = 0
        stats["node_opportunities_total"] = 0
        stats["entity_resolutions_total"] = 0
        stats["merge_candidates_pending_total"] = 0
        stats["insights_total"] = 0
        stats["deep_insights_total"] = 0
        stats["deep_insights_tier1"] = 0
        stats["deep_insights_tier2"] = 0
        stats["experiment_runs_total"] = 0
        stats["experiment_runs_completed"] = 0
        stats["experimental_claims_total"] = 0
        stats["experiments_confirmed"] = 0
        stats["experiments_refuted"] = 0
        stats["gpu_workers_total"] = 0
        stats["gpu_jobs_total"] = 0
        stats["gpu_jobs_running"] = 0
        stats["manuscript_runs_total"] = 0
        stats["submission_bundles_total"] = 0
    return stats


def table_exists(name: str) -> bool:
    """Whether a table exists (SQLite or PostgreSQL)."""
    return _table_exists(name)


def column_names(table: str) -> set[str]:
    """Column names for a table (SQLite or PostgreSQL)."""
    return _column_names(table)


def use_postgres() -> bool:
    """True when DEEPGRAPH_DATABASE_URL is set (PostgreSQL backend)."""
    return _use_pg()


def sql_created_after_hours(hours: int) -> str:
    """SQL fragment: ``created_at`` more recent than N hours (dialect-specific)."""
    if _use_pg():
        return f"created_at > NOW() - INTERVAL '{hours} hours'"
    return f"created_at > datetime('now', '-{hours} hours')"


def sql_updated_after_seconds(seconds: int) -> str:
    """SQL fragment: ``updated_at`` more recent than N seconds (dialect-specific)."""
    if _use_pg():
        return f"updated_at > NOW() - INTERVAL '{seconds} seconds'"
    return f"updated_at > datetime('now', '-{seconds} seconds')"
