"""Build a FrontierPacket from an explicit, auditable retrieval snapshot."""

from __future__ import annotations

from dataclasses import dataclass

from contracts.meta_harness import FrontierPacket


class FrontierBuildError(ValueError):
    pass


@dataclass(frozen=True)
class RetrievalSnapshot:
    retrieved_at: str
    date_start: str
    date_end: str
    source_indexes: tuple[str, ...]
    query_refs: tuple[str, ...]
    strongest_recent_work: tuple[dict, ...]
    latest_benchmarks: tuple[dict, ...]
    nearest_prior_art: tuple[dict, ...]
    obsolete_or_duplicate_evidence: tuple[dict, ...] = ()
    counterevidence_and_negative_results: tuple[dict, ...] = ()

    def coverage(self) -> dict:
        if not self.source_indexes or not self.query_refs:
            raise FrontierBuildError(
                "retrieval coverage requires source indexes and immutable query refs"
            )
        if not self.date_start or not self.date_end:
            raise FrontierBuildError("retrieval coverage requires an explicit date range")
        return {
            "date_start": self.date_start,
            "date_end": self.date_end,
            "source_indexes": list(self.source_indexes),
            "query_refs": list(self.query_refs),
        }


def build_frontier_packet(
    *,
    agenda_id: int,
    research_problem_id: int,
    snapshot: RetrievalSnapshot,
    problem_status: str,
    contribution_delta: dict,
    why_not_obsolete: str,
    minimum_falsification_experiment: dict,
    evaluator: str,
    provider: str,
    model: str,
    prompt_version: str,
) -> FrontierPacket:
    """Create a validated packet; it does not retrieve or invent evidence."""
    packet = FrontierPacket(
        agenda_id=agenda_id,
        research_problem_id=research_problem_id,
        retrieved_at=snapshot.retrieved_at,
        coverage=snapshot.coverage(),
        problem_status=problem_status,
        strongest_recent_work=list(snapshot.strongest_recent_work),
        latest_benchmarks=list(snapshot.latest_benchmarks),
        nearest_prior_art=list(snapshot.nearest_prior_art),
        contribution_delta=dict(contribution_delta),
        obsolete_or_duplicate_evidence=list(
            snapshot.obsolete_or_duplicate_evidence
        ),
        counterevidence_and_negative_results=list(
            snapshot.counterevidence_and_negative_results
        ),
        why_not_obsolete=why_not_obsolete,
        minimum_falsification_experiment=dict(minimum_falsification_experiment),
        evaluator=evaluator,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
    )
    packet.validate()
    return packet
