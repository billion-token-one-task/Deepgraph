"""Problem-first discovery and closed-loop signal feedback."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable

from agents.signal_harvester import (
    HARVEST_SIGNAL_ROLES,
    get_problem_signals,
    get_solution_signals,
    signal_ref_from_row,
    structural_score,
)
from agents.agenda_relevance import agenda_scope_terms
from agents.agenda_repository import row_to_agenda
from db import database as db
from db.evidence_graph import upsert_entity
from meta_harness.scientific_authority import positive_decision_authorized

MIN_SUPPORT = 2
MAX_ATTEMPTS = 3

TABLE_TO_SOURCE_TYPE = {
    "contradiction_clusters": "contradiction",
    "performance_plateaus": "plateau",
    "protocol_artifacts": "protocol_artifact",
    "negative_space_gaps": "negative_space_gap",
    "claim_method_gaps": "claim_method_gap",
    "mechanism_mismatches": "mechanism_mismatch",
}

TABLE_TO_MECHANISM_TYPE = {
    "contradiction_clusters": "mechanism_mismatch",
    "performance_plateaus": "plateau",
    "protocol_artifacts": "protocol_artifact",
    "negative_space_gaps": "negative_space_gap",
    "claim_method_gaps": "claim_method_gap",
    "mechanism_mismatches": "mechanism_mismatch",
}


def _require_agenda_id(value: Any) -> int:
    try:
        agenda_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("agenda_id is required for problem-first discovery") from exc
    if agenda_id <= 0:
        raise ValueError("agenda_id must be a positive integer")
    return agenda_id


def _json_load(value: Any, default: Any):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return default
    return default


def _json_list(value: Any) -> list:
    loaded = _json_load(value, [])
    return loaded if isinstance(loaded, list) else []


def _json_dict(value: Any) -> dict:
    loaded = _json_load(value, {})
    return loaded if isinstance(loaded, dict) else {}


def _json_dump(value: Any) -> str:
    return json.dumps(value if value is not None else [], ensure_ascii=False, sort_keys=True, default=str)


def _dedupe(values: list[Any]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _source_refs(value: Any) -> list[dict]:
    payload = _json_dict(value)
    refs = payload.get("signals") if isinstance(payload, dict) else None
    if isinstance(refs, list):
        return [dict(ref) for ref in refs if isinstance(ref, dict)]
    if isinstance(value, list):
        return [dict(ref) for ref in value if isinstance(ref, dict)]
    return []


def _text_blob(problem: dict) -> str:
    return " ".join(
        str(problem.get(key) or "")
        for key in (
            "problem_statement",
            "formal_statement",
            "source_evidence",
            "desideratum",
            "current_failure_mode",
            "evidence_summary",
            "title",
        )
    ).lower()


def _load_signal_row(source_ref: dict) -> dict | None:
    table = source_ref.get("table")
    if table not in HARVEST_SIGNAL_ROLES:
        return None
    if source_ref.get("content_hash"):
        row = db.fetchone(f"SELECT * FROM {table} WHERE content_hash=?", (source_ref["content_hash"],))
        if row:
            return row
    if source_ref.get("id") is not None:
        return db.fetchone(f"SELECT * FROM {table} WHERE id=?", (source_ref["id"],))
    return None


def can_compile_to_experiment(problem: dict) -> float:
    text = _text_blob(problem)
    if not text:
        return 0.0
    score = 0.0
    if any(term in text for term in ("experiment", "test", "evaluate", "measure", "ablation", "benchmark")):
        score += 0.45
    if any(term in text for term in ("metric", "accuracy", "f1", "loss", "latency", "auc", "score")):
        score += 0.25
    if any(term in text for term in ("falsif", "threshold", "whether", "under condition", "if ")):
        score += 0.2
    if "?" in text or problem.get("central_question"):
        score += 0.1
    return min(1.0, score)


def cross_paper_conflict(problem: dict) -> float:
    ref = _json_dict(problem.get("source_signal_ref"))
    table = ref.get("table") or problem.get("_signal_table") or ""
    paper_ids = _json_list(problem.get("paper_ids")) or problem.get("_paper_ids") or []
    if table == "contradiction_clusters":
        return 1.0 if len(set(paper_ids)) >= 2 else 0.6
    if table == "mechanism_mismatches":
        return 0.8 if len(set(paper_ids)) >= 2 else 0.4
    if table in {"protocol_artifacts", "negative_space_gaps", "claim_method_gaps"}:
        return 0.45 if len(set(paper_ids)) >= 2 else 0.25
    if table == "performance_plateaus":
        return 0.35
    return 0.0


def small_scale_feasible(problem: dict) -> float:
    text = _text_blob(problem)
    difficulty = str(problem.get("difficulty") or "").lower()
    resource = str(problem.get("resource_class") or "").lower()
    if resource == "cpu" or difficulty == "easy":
        return 1.0
    if any(term in text for term in ("small", "subset", "proxy", "ablation", "inference", "cpu")):
        return 0.85
    if difficulty == "hard" or "multi-gpu" in text or "large-scale" in text:
        return 0.35
    return 0.65


def problem_quality_score(problem: dict) -> float:
    paper_ids = _json_list(problem.get("paper_ids")) or problem.get("_paper_ids") or []
    paper_ids = [str(pid) for pid in paper_ids if str(pid).strip()]
    if not paper_ids:
        return 0.0
    try:
        support_count = int(problem.get("support_count") or len(set(paper_ids)))
    except (TypeError, ValueError):
        support_count = len(set(paper_ids))
    score_penalty = -2.0 if support_count < MIN_SUPPORT else 0.0
    evidence = math.log(1 + max(support_count, len(set(paper_ids))))
    score = (
        can_compile_to_experiment(problem) * 2.0
        + evidence * 1.0
        + cross_paper_conflict(problem) * 1.5
        + small_scale_feasible(problem) * 1.0
        + score_penalty
    )
    return round(max(0.0, score), 4)


def _statement_for_signal(signal: dict) -> str:
    table = signal.get("_signal_table") or ""
    if table == "contradiction_clusters":
        return (
            f"Resolve whether the conflict cluster '{signal.get('theme')}' reflects a real "
            "boundary condition, protocol difference, or method-family failure."
        )
    if table == "performance_plateaus":
        return (
            f"Explain why top methods on {signal.get('node_id')} / {signal.get('dataset_name')} "
            f"[{signal.get('metric_name')}] have converged within {signal.get('spread_pct')}%."
        )
    if table == "protocol_artifacts":
        return f"Determine whether protocol artifact '{signal.get('artifact_type')}' is driving measured progress: {signal.get('summary')}"
    if table == "negative_space_gaps":
        return f"Turn the recurring gap '{signal.get('gap_type')}' into a falsifiable experiment: {signal.get('summary')}"
    if table == "claim_method_gaps":
        return f"Test the missing mechanism behind strong claims in {signal.get('node_id')}: {signal.get('summary')}"
    if table == "mechanism_mismatches":
        return f"Disambiguate competing mechanisms for '{signal.get('theme')}' with a decisive experiment."
    return signal.get("summary") or signal.get("theme") or str(table or "research problem")


def promote_to_problem(signal: dict) -> dict:
    table = signal.get("_signal_table") or ""
    node_ids = signal.get("_node_ids") or []
    paper_ids = signal.get("_paper_ids") or []
    source_ref = signal.get("_source_ref") or signal_ref_from_row(table, signal)
    try:
        support_count = int(
            signal.get("support_count")
            or signal.get("paper_count")
            or signal.get("cluster_size")
            or len(paper_ids)
        )
    except (TypeError, ValueError):
        support_count = len(paper_ids)
    problem = {
        "problem_statement": _statement_for_signal(signal),
        "source_signal_ref": source_ref,
        "node_ids": _dedupe(node_ids),
        "paper_ids": _dedupe(paper_ids),
        "support_count": support_count,
        "status": "open",
        "attempts_count": 0,
        "ruled_out_approaches": [],
        "_signal_table": table,
        "_structural_score": structural_score(table, signal),
    }
    problem["problem_quality_score"] = problem_quality_score(problem)
    return problem


def text_similarity(left: str, right: str) -> float:
    left_tokens = {token for token in str(left or "").lower().split() if token}
    right_tokens = {token for token in str(right or "").lower().split() if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def match_problem_to_research_problem(problem: dict, candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    target_nodes = {str(node) for node in problem.get("related_node_ids") or problem.get("node_ids") or [] if str(node).strip()}
    mechanism = str(problem.get("mechanism_type") or problem.get("source_type") or "").lower()
    title_blob = " ".join(
        str(problem.get(key) or "")
        for key in ("title", "formal_statement", "problem_statement", "source_evidence")
    )
    best = None
    best_score = -1.0
    for candidate in candidates:
        candidate_nodes = {str(node) for node in candidate.get("node_ids") or [] if str(node).strip()}
        source_ref = candidate.get("source_signal_ref") or {}
        source_table = str(source_ref.get("table") or "").lower()
        score = 0.0
        if target_nodes and candidate_nodes:
            score += len(target_nodes & candidate_nodes) / max(1, len(target_nodes | candidate_nodes)) * 3.0
        if mechanism and mechanism in source_table:
            score += 2.5
        score += text_similarity(title_blob, candidate.get("problem_statement") or "") * 2.0
        score += float(candidate.get("problem_quality_score") or 0) * 0.1
        if score > best_score:
            best = candidate
            best_score = score
    return best


def hydrate_problem_candidate(problem: dict) -> dict:
    source_ref = problem.get("source_signal_ref") or {}
    source_row = _load_signal_row(source_ref) or {}
    table = str(source_ref.get("table") or "")
    node_ids = _dedupe(problem.get("node_ids") or [])
    paper_ids = _dedupe(problem.get("paper_ids") or [])
    source_text = (
        source_row.get("summary")
        or source_row.get("theme")
        or source_row.get("artifact_type")
        or source_row.get("gap_type")
        or source_row.get("shared_factor")
        or problem.get("problem_statement")
        or ""
    )
    source_type = TABLE_TO_SOURCE_TYPE.get(table, table or "problem")
    mechanism_type = TABLE_TO_MECHANISM_TYPE.get(table, source_type)
    non_numeric_evidence = []
    if source_row.get("summary"):
        non_numeric_evidence.append(str(source_row.get("summary")))
    if source_row.get("theme"):
        non_numeric_evidence.append(str(source_row.get("theme")))
    if source_row.get("artifact_type"):
        non_numeric_evidence.append(f"artifact_type={source_row.get('artifact_type')}")
    if source_row.get("gap_type"):
        non_numeric_evidence.append(f"gap_type={source_row.get('gap_type')}")
    support_count = int(problem.get("support_count") or len(paper_ids) or 0)
    quality = float(problem.get("problem_quality_score") or problem_quality_score(problem))
    return {
        "id": problem.get("id"),
        "title": str(problem.get("problem_statement") or "")[:160] or f"{source_type} problem",
        "source_type": source_type,
        "source_evidence": source_text,
        "formal_statement": problem.get("problem_statement") or source_text,
        "current_failure_mode": source_text,
        "desideratum": f"Produce an experimentally falsifiable resolution for {source_type}.",
        "central_question": problem.get("problem_statement") or source_text,
        "motivation": source_text,
        "result_that_would_change_belief": (
            "A small-scale controlled experiment that clearly changes the observed failure pattern."
        ),
        "mechanism_type": mechanism_type,
        "non_numeric_evidence": _dedupe(non_numeric_evidence)[:6],
        "difficulty": "medium" if quality >= 1.5 else "hard",
        "impact_scope": f"{support_count} supporting papers across {max(1, len(node_ids))} taxonomy areas",
        "related_node_ids": node_ids,
        "research_problem_id": problem.get("id"),
        "problem_statement": problem.get("problem_statement"),
        "source_signal_refs": {
            "signals": [source_ref] if source_ref else [],
            "node_ids": node_ids,
            "paper_ids": paper_ids,
        },
        "source_paper_ids": paper_ids,
        "ruled_out_approaches": problem.get("ruled_out_approaches") or [],
        "problem_quality_score": quality,
    }


def select_problem_first_candidates(
    limit: int = 8,
    *,
    agenda_id: int,
    refresh: bool = True,
) -> list[dict]:
    agenda_id = _require_agenda_id(agenda_id)
    candidates = []
    if refresh:
        candidates = discover_research_problems(
            limit=max(limit * 2, limit),
            agenda_id=agenda_id,
            persist=True,
        )
    if not candidates:
        rows = db.fetchall(
            """
            SELECT *
            FROM research_problems
            WHERE agenda_id=?
              AND status IN ('open', 'exploring')
              AND attempts_count < ?
            ORDER BY problem_quality_score DESC, updated_at ASC
            LIMIT ?
            """,
            (agenda_id, MAX_ATTEMPTS, max(limit * 2, limit)),
        )
        candidates = [_row_to_problem(row) for row in rows]
    return [hydrate_problem_candidate(candidate) for candidate in candidates[:limit]]


def upsert_research_problem(problem: dict, *, agenda_id: int) -> int:
    agenda_id = _require_agenda_id(agenda_id)
    source_ref = problem.get("source_signal_ref") or {}
    source_ref_json = _json_dump(source_ref)
    node_ids_json = _json_dump(problem.get("node_ids") or [])
    paper_ids_json = _json_dump(problem.get("paper_ids") or [])
    ruled_out_json = _json_dump(problem.get("ruled_out_approaches") or [])
    score = float(problem.get("problem_quality_score") or problem_quality_score(problem))
    existing = db.fetchone(
        "SELECT id FROM research_problems WHERE agenda_id=? AND source_signal_ref=?",
        (agenda_id, source_ref_json),
    )
    if existing:
        rid = int(existing["id"])
        db.execute(
            """
            UPDATE research_problems
            SET problem_statement=?, node_ids=?, paper_ids=?, problem_quality_score=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=? AND agenda_id=?
            """,
            (problem.get("problem_statement"), node_ids_json, paper_ids_json, score, rid, agenda_id),
        )
        db.commit()
        return rid
    rid = db.insert_returning_id(
        """
        INSERT INTO research_problems
          (agenda_id, problem_statement, source_signal_ref, node_ids, paper_ids,
           problem_quality_score, status, attempts_count, ruled_out_approaches)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        (
            agenda_id,
            problem.get("problem_statement"),
            source_ref_json,
            node_ids_json,
            paper_ids_json,
            score,
            problem.get("status") or "open",
            int(problem.get("attempts_count") or 0),
            ruled_out_json,
        ),
    )
    db.commit()
    return rid


def discover_research_problems(
    limit: int = 20,
    *,
    agenda_id: int,
    persist: bool = True,
) -> list[dict]:
    agenda_id = _require_agenda_id(agenda_id)
    signals = get_problem_signals(limit=max(limit * 2, limit))
    candidates = [promote_to_problem(signal) for signal in signals]
    agenda_row = (
        db.fetchone("SELECT * FROM research_agendas WHERE id=?", (agenda_id,))
        if db.table_exists("research_agendas")
        else None
    )
    if agenda_row:
        agenda = row_to_agenda(agenda_row)
        terms = agenda_scope_terms(agenda)
        scoped = [
            problem
            for problem in candidates
            if any(term in _text_blob(problem) for term in terms)
        ]
        if scoped:
            candidates = scoped
        else:
            direction = str(agenda.description or agenda.name).strip()
            focus = ", ".join(terms)
            candidates = [
                {
                    "problem_statement": (
                        f"Within the research direction '{direction}', determine which "
                        f"falsifiable intervention concerning {focus or direction} can "
                        "improve a measurable baseline under the declared resource budget, "
                        "and identify the smallest controlled experiment that would refute it."
                    ),
                    "source_signal_ref": {
                        "kind": "agenda_direction",
                        "axis": "effectiveness",
                        "agenda_id": agenda_id,
                        "agenda_version": agenda.version,
                    },
                    "node_ids": [],
                    "paper_ids": [],
                    "support_count": 0,
                    "status": "open",
                    "attempts_count": 0,
                    "ruled_out_approaches": [],
                    "problem_quality_score": 1.0,
                },
                {
                    "problem_statement": (
                        f"Within the research direction '{direction}', identify the "
                        f"boundary conditions and resource/robustness trade-offs for "
                        f"{focus or direction}; design a measurable comparison whose "
                        "negative result would rule out at least one plausible approach."
                    ),
                    "source_signal_ref": {
                        "kind": "agenda_direction",
                        "axis": "boundary_conditions",
                        "agenda_id": agenda_id,
                        "agenda_version": agenda.version,
                    },
                    "node_ids": [],
                    "paper_ids": [],
                    "support_count": 0,
                    "status": "open",
                    "attempts_count": 0,
                    "ruled_out_approaches": [],
                    "problem_quality_score": 0.9,
                },
            ]
    candidates.sort(key=lambda item: item.get("problem_quality_score") or 0, reverse=True)
    out = candidates[:limit]
    if persist:
        for problem in out:
            problem["agenda_id"] = agenda_id
            problem["id"] = upsert_research_problem(problem, agenda_id=agenda_id)
    return out


def _row_to_problem(row: dict) -> dict:
    out = dict(row)
    out["source_signal_ref"] = _json_dict(row.get("source_signal_ref"))
    out["node_ids"] = _json_list(row.get("node_ids"))
    out["paper_ids"] = _json_list(row.get("paper_ids"))
    out["ruled_out_approaches"] = _json_list(row.get("ruled_out_approaches"))
    return out


def select_next_problem(*, agenda_id: int) -> dict | None:
    agenda_id = _require_agenda_id(agenda_id)
    row = db.fetchone(
        """
        SELECT * FROM research_problems
        WHERE agenda_id=?
          AND status IN ('open', 'exploring')
          AND attempts_count < ?
        ORDER BY problem_quality_score DESC, updated_at ASC
        LIMIT 1
        """,
        (agenda_id, MAX_ATTEMPTS),
    )
    if not row:
        discovered = discover_research_problems(limit=5, agenda_id=agenda_id, persist=True)
        if not discovered:
            return None
        return discovered[0]
    return _row_to_problem(row)


def _approach_summary(signal: dict) -> str:
    table = signal.get("_signal_table")
    if table == "method_transfer":
        return signal.get("title") or signal.get("hypothesis") or "method transfer"
    if table == "hidden_variable_bridges":
        return f"Use shared factor {signal.get('shared_factor')} between {signal.get('node_a_id')} and {signal.get('node_b_id')}."
    if table == "node_entity_overlap":
        return f"Explore shared entities between {signal.get('node_a_id')} and {signal.get('node_b_id')}."
    if table == "pattern_matches":
        return f"Transfer convergent pattern {signal.get('pattern_a_id')} <-> {signal.get('pattern_b_id')}."
    return signal.get("summary") or signal.get("title") or str(table or "candidate approach")


def propose_approach(problem: dict, solution_signals: list[dict] | None = None, ruled_out: list[dict] | None = None) -> dict | None:
    solution_signals = solution_signals or get_solution_signals(problem, limit=30)
    ruled_text = " ".join(str(item.get("approach") or item.get("summary") or "") for item in (ruled_out or []))
    for signal in solution_signals:
        summary = _approach_summary(signal)
        if summary and summary in ruled_text:
            continue
        source_ref = signal.get("_source_ref") or {}
        return {
            "summary": summary,
            "source_signal_refs": {"signals": [source_ref] if source_ref else []},
            "hypothesis_node": (signal.get("_node_ids") or [None])[0],
            "raw_signal": signal,
        }
    return None


def record_solution(
    problem: dict,
    approach: dict,
    result: dict,
    *,
    agenda_id: int,
) -> bool:
    agenda_id = _require_agenda_id(agenda_id)
    pid = problem.get("id")
    if not pid or not positive_decision_authorized(
        agenda_id=agenda_id,
        run_id=result.get("run_id"),
    ):
        return False
    db.execute(
        """
        UPDATE research_problems
        SET status='solved', updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND agenda_id=?
        """,
        (pid, agenda_id),
    )
    db.commit()
    return True


def append_ruled_out_approach(
    problem_id: int,
    approach: dict,
    result: dict,
    *,
    agenda_id: int,
) -> None:
    agenda_id = _require_agenda_id(agenda_id)
    row = db.fetchone(
        """
        SELECT ruled_out_approaches
        FROM research_problems
        WHERE id=? AND agenda_id=?
        """,
        (problem_id, agenda_id),
    )
    current = _json_list(row.get("ruled_out_approaches") if row else None)
    item = {
        "approach": approach.get("summary") or approach.get("name") or "",
        "failed_under": result.get("conditions") or {},
        "effect_size": result.get("effect_size"),
        "verdict": result.get("verdict"),
    }
    if item["approach"] and not any(entry.get("approach") == item["approach"] for entry in current if isinstance(entry, dict)):
        current.append(item)
    db.execute(
        """
        UPDATE research_problems
        SET ruled_out_approaches=?,
            attempts_count=attempts_count + 1,
            status=CASE WHEN attempts_count + 1 >= ? THEN 'abandoned' ELSE 'exploring' END,
            updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND agenda_id=?
        """,
        (_json_dump(current), MAX_ATTEMPTS, problem_id, agenda_id),
    )
    db.commit()


def update_signal_posterior(
    source_signal_refs: Any,
    verdict: str,
    *,
    agenda_id: int,
    run_id: int | None,
    experimental_claim_id: int | None,
    effect_size: float | None,
    p_value: float | None,
    conditions: dict,
) -> list[dict]:
    """Record agenda-local feedback without mutating shared ingestion signals."""
    agenda_id = _require_agenda_id(agenda_id)
    verdict = str(verdict or "").lower()
    if verdict in {"confirmed", "reproduced"}:
        outcome = "supported"
    elif verdict == "refuted":
        outcome = "refuted"
    else:
        return []
    updates = []
    for ref in _source_refs(source_signal_refs):
        table = ref.get("table")
        if table not in HARVEST_SIGNAL_ROLES:
            continue
        content_hash = ref.get("content_hash")
        row = None
        if content_hash:
            row = db.fetchone(f"SELECT * FROM {table} WHERE content_hash=?", (content_hash,))
        if not row and ref.get("id") is not None:
            row = db.fetchone(f"SELECT * FROM {table} WHERE id=?", (ref.get("id"),))
            content_hash = row.get("content_hash") if row else content_hash
        if not row or not content_hash:
            continue
        idempotency_material = {
            "agenda_id": agenda_id,
            "run_id": run_id,
            "experimental_claim_id": experimental_claim_id,
            "signal_table": table,
            "content_hash": content_hash,
            "verdict": outcome,
        }
        idempotency_key = hashlib.sha256(
            _json_dump(idempotency_material).encode("utf-8")
        ).hexdigest()
        db.execute(
            """
            INSERT INTO agenda_signal_outcomes
                (agenda_id, run_id, experimental_claim_id, signal_table,
                 signal_content_hash, verdict, effect_size, p_value,
                 conditions_json, idempotency_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (agenda_id, idempotency_key) DO NOTHING
            """,
            (
                agenda_id,
                run_id,
                experimental_claim_id,
                table,
                content_hash,
                outcome,
                effect_size,
                p_value,
                _json_dump(conditions),
                idempotency_key,
            ),
        )
        counts = db.fetchone(
            """
            SELECT
                SUM(CASE WHEN verdict='supported' THEN 1 ELSE 0 END)
                    AS confirm_count,
                SUM(CASE WHEN verdict='refuted' THEN 1 ELSE 0 END)
                    AS refute_count
            FROM agenda_signal_outcomes
            WHERE agenda_id=? AND signal_table=? AND signal_content_hash=?
            """,
            (agenda_id, table, content_hash),
        )
        confirm = int((counts or {}).get("confirm_count") or 0)
        refute = int((counts or {}).get("refute_count") or 0)
        alpha0 = 1.0 + min(10.0, max(0.0, structural_score(table, row)))
        beta0 = 1.0
        posterior = alpha0 + confirm
        posterior = posterior / (posterior + beta0 + refute)
        updates.append(
            {
                "table": table,
                "content_hash": content_hash,
                "agenda_id": agenda_id,
                "confirm_count": confirm,
                "refute_count": refute,
                "empirical_posterior": round(posterior, 6),
            }
        )
    if updates:
        db.commit()
    return updates


def _empirical_entity_id(run_id: int | None, claim_id: int | None) -> str:
    key = f"run:{run_id or 'none'}:claim:{claim_id or 'none'}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"empirical_result:{digest}"


def _insert_evidence_edge(payload: dict) -> None:
    agenda_id = _require_agenda_id(payload.get("agenda_id"))
    existing = db.fetchone(
        """
        SELECT id FROM experimental_evidence_edges
        WHERE agenda_id=?
          AND COALESCE(run_id, -1)=COALESCE(?, -1)
          AND COALESCE(deep_insight_id, -1)=COALESCE(?, -1)
          AND COALESCE(target_kind, '')=COALESCE(?, '')
          AND COALESCE(target_id, '')=COALESCE(?, '')
          AND COALESCE(relation, '')=COALESCE(?, '')
        """,
        (
            agenda_id,
            payload.get("run_id"),
            payload.get("deep_insight_id"),
            payload.get("target_kind"),
            payload.get("target_id"),
            payload.get("relation"),
        ),
    )
    if existing:
        return
    db.execute(
        """
        INSERT INTO experimental_evidence_edges
          (agenda_id, experimental_claim_id, run_id, deep_insight_id,
           research_problem_id, empirical_entity_id, target_kind, target_id,
           relation, verdict, effect_size, conditions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            agenda_id,
            payload.get("experimental_claim_id"),
            payload.get("run_id"),
            payload.get("deep_insight_id"),
            payload.get("research_problem_id"),
            payload.get("empirical_entity_id"),
            payload.get("target_kind"),
            payload.get("target_id"),
            payload.get("relation"),
            payload.get("verdict"),
            payload.get("effect_size"),
            _json_dump(payload.get("conditions") or {}),
        ),
    )


def writeback_experiment_result(
    *,
    agenda_id: int,
    run_id: int | None,
    deep_insight_id: int,
    verdict: str,
    effect_size: float | None = None,
    p_value: float | None = None,
    conditions: dict | None = None,
    source_signal_refs: Any = None,
    experimental_claim_id: int | None = None,
) -> dict:
    agenda_id = _require_agenda_id(agenda_id)
    insight = db.fetchone(
        "SELECT * FROM deep_insights WHERE id=? AND agenda_id=?",
        (deep_insight_id, agenda_id),
    )
    if not insight:
        return {"updated_signals": [], "evidence_edges": 0, "reason": "missing_deep_insight"}
    if run_id is not None:
        run = db.fetchone(
            "SELECT agenda_id FROM experiment_runs WHERE id=?",
            (int(run_id),),
        )
        if not run or int(run.get("agenda_id") or 0) != agenda_id:
            raise ValueError("experiment result writeback run scope mismatch")
    refs = source_signal_refs or insight.get("source_signal_refs")
    positive_authorized = (
        str(verdict).lower() in {"confirmed", "supported", "reproduced"}
        and positive_decision_authorized(agenda_id=agenda_id, run_id=run_id)
    )
    feedback_verdict = (
        "confirmed"
        if positive_authorized
        else ("refuted" if str(verdict).lower() == "refuted" else "inconclusive")
    )
    signal_updates = update_signal_posterior(
        refs,
        feedback_verdict,
        agenda_id=agenda_id,
        run_id=run_id,
        experimental_claim_id=experimental_claim_id,
        effect_size=effect_size,
        p_value=p_value,
        conditions=conditions or {},
    )
    research_problem_id = insight.get("research_problem_id")
    result = {
        "verdict": verdict,
        "conditions": conditions or {},
        "effect_size": effect_size,
        "p_value": p_value,
    }
    if research_problem_id:
        if str(verdict).lower() == "refuted":
            approach = {
                "summary": (insight.get("proposed_method") or insight.get("title") or "")[:500]
            }
            append_ruled_out_approach(
                int(research_problem_id),
                approach,
                result,
                agenda_id=agenda_id,
            )
        elif positive_authorized:
            record_solution(
                {"id": int(research_problem_id), "agenda_id": agenda_id},
                {},
                {**result, "run_id": run_id},
                agenda_id=agenda_id,
            )

    edge_count = 0
    if str(verdict).lower() == "refuted":
        entity_id = _empirical_entity_id(run_id, experimental_claim_id)
        description = (
            f"Experimental result for deep_insight {deep_insight_id} was refuted"
            f" under {json.dumps(conditions or {}, ensure_ascii=False, default=str)}."
        )
        upsert_entity(
            {
                "id": entity_id,
                "canonical_name": f"Refuted experiment {run_id or experimental_claim_id or deep_insight_id}",
                "entity_type": "empirical_result",
                "description": description,
                "metadata": {
                    "run_id": run_id,
                    "deep_insight_id": deep_insight_id,
                    "experimental_claim_id": experimental_claim_id,
                    "verdict": verdict,
                    "effect_size": effect_size,
                    "p_value": p_value,
                    "conditions": conditions or {},
                },
            }
        )
        base_payload = {
            "agenda_id": agenda_id,
            "experimental_claim_id": experimental_claim_id,
            "run_id": run_id,
            "deep_insight_id": deep_insight_id,
            "research_problem_id": research_problem_id,
            "empirical_entity_id": entity_id,
            "verdict": verdict,
            "effect_size": effect_size,
            "conditions": conditions or {},
        }
        targets = [
            {"target_kind": "deep_insight", "target_id": str(deep_insight_id), "relation": "refutes"}
        ]
        for node_id in _json_list(insight.get("source_node_ids")):
            targets.append(
                {"target_kind": "taxonomy_node", "target_id": str(node_id), "relation": "negative_evidence_for"}
            )
        for ref in _source_refs(refs):
            if ref.get("content_hash") and ref.get("table"):
                targets.append(
                    {
                        "target_kind": f"signal:{ref['table']}",
                        "target_id": str(ref["content_hash"]),
                        "relation": "negative_evidence_for",
                    }
                )
        for target in targets:
            _insert_evidence_edge({**base_payload, **target})
            edge_count += 1
        db.commit()
    return {
        "updated_signals": signal_updates,
        "evidence_edges": edge_count,
        "positive_feedback_authorized": positive_authorized,
        "feedback_suppressed_reason": (
            None
            if positive_authorized or str(verdict).lower() == "refuted"
            else "scientifically_decided_supported_transition_required"
        ),
    }


def run_experiment_worker(problem: dict, approach: dict, *, run_id: int | None = None) -> dict:
    """Compatibility wrapper for problem-first callers.

    Direct execution is no longer a compatibility behavior. A staged approach
    may return an inconclusive packet, but an existing run must enter through
    the grant-scoped durable ComputeScheduler path.
    """
    if run_id is None:
        return {
            "verdict": "inconclusive",
            "conditions": {"reason": "no_run_id_supplied"},
            "effect_size": None,
            "run_id": None,
            "deep_insight_id": None,
            "source_signal_refs": approach.get("source_signal_refs") or {},
        }
    raise PermissionError(
        "direct problem-first validation is disabled; submit the existing "
        "run through ComputeScheduler with a valid ResourceGrant"
    )


def problem_first_cycle(
    *,
    agenda_id: int,
    max_attempts: int = MAX_ATTEMPTS,
    worker: Callable[[dict, dict], dict] | None = None,
) -> dict:
    agenda_id = _require_agenda_id(agenda_id)
    problem = select_next_problem(agenda_id=agenda_id)
    if not problem:
        return {"status": "no_problem"}
    pid = problem.get("id")
    if pid:
        db.execute(
            "UPDATE research_problems SET status='exploring', updated_at=CURRENT_TIMESTAMP "
            "WHERE id=? AND agenda_id=?",
            (pid, agenda_id),
        )
        db.commit()
    attempts = int(problem.get("attempts_count") or 0)
    while attempts < max_attempts:
        approach = propose_approach(problem, ruled_out=problem.get("ruled_out_approaches") or [])
        if not approach:
            return {"status": "no_approach", "problem": problem}
        if worker is None:
            return {"status": "approach_ready", "problem": problem, "approach": approach}
        result = worker(problem, approach)
        verdict = str(result.get("verdict") or "").lower()
        if verdict in {"confirmed", "supported", "reproduced"}:
            if record_solution(
                problem,
                approach,
                result,
                agenda_id=agenda_id,
            ):
                return {
                    "status": "solved",
                    "problem": problem,
                    "approach": approach,
                    "result": result,
                }
            return {
                "status": "awaiting_scientific_decision",
                "problem": problem,
                "approach": approach,
                "result": result,
            }
        if verdict == "refuted":
            if pid:
                append_ruled_out_approach(
                    int(pid),
                    approach,
                    result,
                    agenda_id=agenda_id,
                )
            writeback_experiment_result(
                agenda_id=agenda_id,
                run_id=result.get("run_id"),
                deep_insight_id=int(result.get("deep_insight_id") or 0),
                verdict=verdict,
                effect_size=result.get("effect_size"),
                conditions=result.get("conditions") or {},
                source_signal_refs=result.get("source_signal_refs") or approach.get("source_signal_refs"),
            )
        elif pid:
            tracked_result = dict(result)
            tracked_result["verdict"] = verdict or "inconclusive"
            append_ruled_out_approach(
                int(pid),
                approach,
                tracked_result,
                agenda_id=agenda_id,
            )
        attempts += 1
    return {"status": "attempt_limit", "problem": problem}
