"""Database connection and helpers."""
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from config import DB_PATH

_local = threading.local()


def get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db():
    conn = get_conn()
    schema_path = Path(__file__).parent / "schema.sql"
    conn.executescript(schema_path.read_text())
    conn.commit()


def execute(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    return get_conn().execute(sql, params)


def executemany(sql: str, params_list: list) -> sqlite3.Cursor:
    return get_conn().executemany(sql, params_list)


def commit():
    get_conn().commit()


def fetchone(sql: str, params: tuple = ()) -> dict | None:
    row = execute(sql, params).fetchone()
    return dict(row) if row else None


def fetchall(sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in execute(sql, params).fetchall()]


def _dump_json(value: Any) -> str:
    return json.dumps(value if value is not None else [])


def _load_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def insert_paper(paper: dict) -> str:
    execute(
        """INSERT OR IGNORE INTO papers (id, title, authors, abstract, categories, published_date, pdf_url)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (paper["id"], paper["title"], json.dumps(paper.get("authors", [])),
         paper.get("abstract", ""), json.dumps(paper.get("categories", [])),
         paper.get("published_date", ""), paper.get("pdf_url", ""))
    )
    commit()
    return paper["id"]


def update_paper_text(paper_id: str, full_text: str):
    execute("UPDATE papers SET full_text=?, status='ingested', updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (full_text, paper_id))
    commit()


def update_paper_status(paper_id: str, status: str, error_msg: str = None, token_cost: int = 0):
    execute(
        "UPDATE papers SET status=?, error_msg=?, token_cost=token_cost+?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (status, error_msg, token_cost, paper_id))
    commit()


def insert_claim(claim: dict) -> int:
    cur = execute(
        """INSERT INTO claims (paper_id, claim_text, claim_type, method_name, dataset_name,
           metric_name, metric_value, evidence_location, conditions)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (claim["paper_id"], claim["claim_text"], claim.get("claim_type"),
         claim.get("method_name"), claim.get("dataset_name"),
         claim.get("metric_name"), claim.get("metric_value"),
         claim.get("evidence_location"), json.dumps(claim.get("conditions", {})))
    )
    commit()
    return cur.lastrowid


def insert_method(method: dict):
    execute(
        """INSERT OR IGNORE INTO methods (name, category, description, key_innovation, first_paper_id, builds_on)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (method["name"], method.get("category"), method.get("description"),
         method.get("key_innovation"), method.get("first_paper_id"),
         json.dumps(method.get("builds_on", [])))
    )
    commit()


def insert_pattern(pattern: dict) -> int:
    cur = execute(
        """INSERT INTO patterns (pattern_text, pattern_type, domain_count, domains, claim_ids)
           VALUES (?, ?, ?, ?, ?)""",
        (pattern["pattern_text"], pattern.get("pattern_type"),
         pattern.get("domain_count", 1), json.dumps(pattern.get("domains", [])),
         json.dumps(pattern.get("claim_ids", [])))
    )
    commit()
    return cur.lastrowid


def insert_contradiction(c: dict) -> int:
    cur = execute(
        """INSERT INTO contradictions (claim_a_id, claim_b_id, description, condition_diff, hypothesis, severity)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (c["claim_a_id"], c["claim_b_id"], c["description"],
         c.get("condition_diff"), c.get("hypothesis"), c.get("severity", "medium"))
    )
    commit()
    return cur.lastrowid


def insert_gap(g: dict) -> int:
    cur = execute(
        """INSERT INTO gaps (problem_pattern_id, solution_pattern_id, gap_description,
           missing_domain, evidence_papers, research_proposal, value_score)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (g.get("problem_pattern_id"), g.get("solution_pattern_id"),
         g["gap_description"], g.get("missing_domain"),
         json.dumps(g.get("evidence_papers", [])),
         g.get("research_proposal"), g.get("value_score"))
    )
    commit()
    return cur.lastrowid


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
    # New tables (may not exist yet during migration)
    try:
        stats["results_total"] = fetchone("SELECT COUNT(*) as c FROM results")["c"]
        stats["taxonomy_assignments"] = fetchone("SELECT COUNT(*) as c FROM paper_taxonomy")["c"]
        stats["matrix_gaps_total"] = fetchone("SELECT COUNT(*) as c FROM matrix_gaps")["c"]
        stats["paper_insights_total"] = fetchone("SELECT COUNT(*) as c FROM paper_insights")["c"]
        stats["node_summaries_total"] = fetchone("SELECT COUNT(*) as c FROM node_summaries")["c"]
    except Exception:
        stats["results_total"] = 0
        stats["taxonomy_assignments"] = 0
        stats["matrix_gaps_total"] = 0
        stats["paper_insights_total"] = 0
        stats["node_summaries_total"] = 0
    return stats
