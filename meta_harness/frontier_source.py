"""Agenda-scoped Frontier snapshots assembled from the live evidence graph.

This module does not use an LLM and does not infer scientific conclusions.
It reads the problem's explicitly linked papers and experimental evidence,
then content-addresses the retrieved rows.  An independent evaluator must
still supply the assessment fields used by the Frontier gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from contracts.meta_harness import FrontierPacket
from db import database as db
from meta_harness.frontier_builder import (
    FrontierBuildError,
    RetrievalSnapshot,
    build_frontier_packet,
)


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FrontierBuildError("evidence graph contains malformed JSON") from exc
    if not isinstance(parsed, list):
        raise FrontierBuildError("evidence graph list field is not an array")
    return parsed


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paper_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": str(row.get("id") or ""),
        "title": str(row.get("title") or ""),
        "published_date": str(row.get("published_date") or ""),
        "status": str(row.get("status") or ""),
        "result_count": int(row.get("result_count") or 0),
        "sota_result_count": int(row.get("sota_result_count") or 0),
        "grounded_claim_count": int(row.get("grounded_claim_count") or 0),
    }


@dataclass(frozen=True)
class FrontierAssessment:
    """Independent assessment applied to a content-addressed retrieval."""

    problem_status: str
    contribution_delta: dict[str, Any]
    why_not_obsolete: str
    minimum_falsification_experiment: dict[str, Any]
    evaluator: str
    provider: str
    model: str
    prompt_version: str
    coverage_start: str
    coverage_end: str

    def validate(self) -> None:
        required = {
            "problem_status": self.problem_status,
            "why_not_obsolete": self.why_not_obsolete,
            "evaluator": self.evaluator,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "coverage_start": self.coverage_start,
            "coverage_end": self.coverage_end,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise FrontierBuildError(
                "frontier assessment is incomplete:" + ",".join(sorted(missing))
            )
        if not self.contribution_delta:
            raise FrontierBuildError("frontier assessment requires contribution_delta")
        if not self.minimum_falsification_experiment:
            raise FrontierBuildError(
                "frontier assessment requires a minimum falsification experiment"
            )


class EvidenceGraphFrontierSource:
    """Read-only adapter from agenda-scoped graph rows to ``FrontierPacket``."""

    def _problem(self, *, agenda_id: int, research_problem_id: int) -> dict:
        problem = db.fetchone(
            """
            SELECT id, agenda_id, problem_statement, source_signal_ref,
                   node_ids, paper_ids, status, ruled_out_approaches,
                   created_at, updated_at
            FROM research_problems
            WHERE id=? AND agenda_id=?
            """,
            (research_problem_id, agenda_id),
        )
        if not problem:
            raise FrontierBuildError("agenda-scoped research problem not found")
        return problem

    def _papers(self, paper_ids: list[str]) -> list[dict]:
        if not paper_ids:
            raise FrontierBuildError(
                "research problem has no explicitly linked evidence papers"
            )
        placeholders = ",".join("?" for _ in paper_ids)
        rows = db.fetchall(
            f"""
            SELECT p.id, p.title, p.published_date, p.status,
                   COUNT(DISTINCT r.id) AS result_count,
                   COUNT(DISTINCT CASE WHEN r.is_sota=1 THEN r.id END)
                       AS sota_result_count,
                   COUNT(DISTINCT CASE
                       WHEN c.grounding_status='verified' THEN c.id END)
                       AS grounded_claim_count
            FROM papers AS p
            LEFT JOIN results AS r ON r.paper_id=p.id
            LEFT JOIN claims AS c ON c.paper_id=p.id
            WHERE p.id IN ({placeholders})
            GROUP BY p.id, p.title, p.published_date, p.status
            ORDER BY p.published_date DESC, p.id
            """,
            tuple(paper_ids),
        )
        found = {str(row.get("id") or "") for row in rows}
        missing = sorted(set(paper_ids) - found)
        if missing:
            raise FrontierBuildError(
                "linked evidence papers are missing:" + ",".join(missing)
            )
        return rows

    def _benchmark_rows(self, paper_ids: list[str]) -> list[dict]:
        placeholders = ",".join("?" for _ in paper_ids)
        return db.fetchall(
            f"""
            SELECT r.id, r.paper_id, r.method_name, r.dataset_name,
                   r.metric_name, r.metric_value, r.metric_unit, r.is_sota,
                   r.evidence_location, r.grounding_status
            FROM results AS r
            WHERE r.paper_id IN ({placeholders})
              AND r.metric_name IS NOT NULL
              AND r.metric_value IS NOT NULL
            ORDER BY r.is_sota DESC, r.created_at DESC, r.id DESC
            LIMIT 100
            """,
            tuple(paper_ids),
        )

    def _negative_rows(
        self, *, agenda_id: int, research_problem_id: int
    ) -> list[dict]:
        return db.fetchall(
            """
            SELECT id, relation, verdict, effect_size, conditions,
                   run_id, deep_insight_id, target_kind, target_id, created_at
            FROM experimental_evidence_edges
            WHERE agenda_id=? AND research_problem_id=?
              AND (
                  verdict IN ('refuted', 'inconclusive', 'invalid')
                  OR relation IN ('refutes', 'contradicts', 'negative_result')
              )
            ORDER BY created_at DESC, id DESC
            LIMIT 100
            """,
            (agenda_id, research_problem_id),
        )

    def evidence_briefing(
        self,
        *,
        agenda_id: int,
        research_problem_id: int,
    ) -> dict[str, Any]:
        """Read-only view of the evidence an evaluator is allowed to see.

        Exactly the explicitly linked rows: the agenda-scoped problem, its
        linked papers, their benchmark results and the recorded negative
        evidence. Nothing is inferred, retrieved from the open web, or borrowed
        from another agenda, and the returned ``query_ref`` content-addresses
        the snapshot so the resulting packet stays independently verifiable.
        """
        if int(agenda_id) <= 0 or int(research_problem_id) <= 0:
            raise FrontierBuildError("frontier scope ids must be positive")
        problem = self._problem(
            agenda_id=int(agenda_id),
            research_problem_id=int(research_problem_id),
        )
        paper_ids = [
            str(value).strip()
            for value in _json_list(problem.get("paper_ids"))
            if str(value).strip()
        ]
        paper_rows = self._papers(paper_ids)
        briefing = {
            "agenda_id": int(agenda_id),
            "research_problem_id": int(research_problem_id),
            "problem_statement": str(problem.get("problem_statement") or ""),
            "problem_status": str(problem.get("status") or ""),
            "ruled_out_approaches": _json_list(problem.get("ruled_out_approaches")),
            "papers": [_paper_ref(row) for row in paper_rows],
            "benchmarks": [dict(row) for row in self._benchmark_rows(paper_ids)],
            "negative_evidence": [
                dict(row)
                for row in self._negative_rows(
                    agenda_id=int(agenda_id),
                    research_problem_id=int(research_problem_id),
                )
            ],
        }
        briefing["query_ref"] = "deepgraph:evidence-graph:sha256:" + _canonical_hash(
            briefing
        )
        return briefing

    def build(
        self,
        *,
        agenda_id: int,
        research_problem_id: int,
        assessment: FrontierAssessment,
        retrieved_at: str | None = None,
    ) -> FrontierPacket:
        if int(agenda_id) <= 0 or int(research_problem_id) <= 0:
            raise FrontierBuildError("frontier scope ids must be positive")
        assessment.validate()
        problem = self._problem(
            agenda_id=int(agenda_id),
            research_problem_id=int(research_problem_id),
        )
        paper_ids = [
            str(value).strip()
            for value in _json_list(problem.get("paper_ids"))
            if str(value).strip()
        ]
        paper_rows = self._papers(paper_ids)
        benchmark_rows = self._benchmark_rows(paper_ids)
        negative_rows = self._negative_rows(
            agenda_id=int(agenda_id),
            research_problem_id=int(research_problem_id),
        )

        paper_refs = [_paper_ref(row) for row in paper_rows]
        benchmark_refs = [dict(row) for row in benchmark_rows]
        negative_refs = [dict(row) for row in negative_rows]
        retrieval_content = {
            "agenda_id": int(agenda_id),
            "research_problem_id": int(research_problem_id),
            "problem": {
                key: problem.get(key)
                for key in (
                    "id",
                    "agenda_id",
                    "problem_statement",
                    "source_signal_ref",
                    "node_ids",
                    "paper_ids",
                    "status",
                    "ruled_out_approaches",
                    "updated_at",
                )
            },
            "papers": paper_refs,
            "benchmarks": benchmark_refs,
            "negative_evidence": negative_refs,
        }
        query_ref = "deepgraph:evidence-graph:sha256:" + _canonical_hash(
            retrieval_content
        )
        strongest = sorted(
            paper_refs,
            key=lambda row: (
                int(row["sota_result_count"]),
                int(row["grounded_claim_count"]),
                str(row["published_date"]),
            ),
            reverse=True,
        )[:20]
        obsolete_evidence: list[dict[str, Any]] = []
        persisted_status = str(problem.get("status") or "")
        if persisted_status in {"duplicate", "obsolete", "solved"}:
            obsolete_evidence.append(
                {
                    "research_problem_id": int(research_problem_id),
                    "status": persisted_status,
                    "source_signal_ref": str(problem.get("source_signal_ref") or ""),
                }
            )
        effective_status = (
            persisted_status
            if persisted_status in {"duplicate", "obsolete", "solved"}
            else assessment.problem_status
        )
        snapshot = RetrievalSnapshot(
            retrieved_at=retrieved_at or _utc_now(),
            date_start=assessment.coverage_start,
            date_end=assessment.coverage_end,
            source_indexes=("deepgraph.postgresql.evidence_graph",),
            query_refs=(query_ref,),
            strongest_recent_work=tuple(strongest),
            latest_benchmarks=tuple(benchmark_refs[:30]),
            nearest_prior_art=tuple(paper_refs),
            obsolete_or_duplicate_evidence=tuple(obsolete_evidence),
            counterevidence_and_negative_results=tuple(negative_refs),
        )
        return build_frontier_packet(
            agenda_id=int(agenda_id),
            research_problem_id=int(research_problem_id),
            snapshot=snapshot,
            problem_status=effective_status,
            contribution_delta=assessment.contribution_delta,
            why_not_obsolete=assessment.why_not_obsolete,
            minimum_falsification_experiment=(
                assessment.minimum_falsification_experiment
            ),
            evaluator=assessment.evaluator,
            provider=assessment.provider,
            model=assessment.model,
            prompt_version=assessment.prompt_version,
        )
