"""Declarative high-level agent boundaries.

The existing module paths stay import-compatible. These boundaries are the
stable ownership map for the larger agent folders under ``agents/``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentBoundary:
    key: str
    title: str
    folder: str
    responsibility: str
    config_sections: tuple[str, ...]
    modules: tuple[str, ...]
    scripts: tuple[str, ...] = ()


AGENT_BOUNDARIES: tuple[AgentBoundary, ...] = (
    AgentBoundary(
        key="paper_extraction",
        title="Paper Extraction Agent",
        folder="agents/paper_extraction",
        responsibility="Discover papers, parse PDFs, extract structured claims, and audit source completeness.",
        config_sections=("profile", "arxiv", "grobid", "pdf", "llm", "paths"),
        modules=(
            "ingestion.arxiv_client",
            "ingestion.arxiv_ids",
            "ingestion.grobid_tei",
            "ingestion.pdf_parser",
            "agents.extraction_agent",
            "agents.claim_grounding",
            "agents.paper_completeness",
            "agents.reference_corpus_audit",
        ),
    ),
    AgentBoundary(
        key="graph_construction",
        title="Graph Construction Agent",
        folder="agents/graph_construction",
        responsibility="Build and maintain the evidence graph, taxonomy, opportunity signals, and graph feedback loop.",
        config_sections=("database", "graph", "discovery", "pipeline", "paths"),
        modules=(
            "db.evidence_graph",
            "db.opportunity_engine",
            "db.taxonomy",
            "agents.taxonomy_expander",
            "agents.domain_summary_agent",
            "agents.signal_harvester",
            "agents.knowledge_loop",
            "agents.meta_learner",
        ),
    ),
    AgentBoundary(
        key="idea_generation",
        title="Idea Generation Agent",
        folder="agents/idea_generation",
        responsibility="Generate, rank, route, and verify cross-paper research ideas before experiments.",
        config_sections=("discovery", "idea", "llm", "paper_orchestra"),
        modules=(
            "agents.insight_agent",
            "agents.insight_ranker",
            "agents.reasoning_agent",
            "agents.abstraction_agent",
            "agents.research_bridge",
            "agents.paradigm_agent",
            "agents.paper_idea_agent",
            "agents.novelty_verifier",
            "agents.evidence_planner",
            "agents.idea_route",
            "agents.discovery_metadata",
            "agents.discovery_supervisor",
        ),
    ),
    AgentBoundary(
        key="experiment_planning",
        title="Experiment Planning Agent",
        folder="agents/experiment_planning",
        responsibility="Turn ideas into benchmark contracts, scaffold experiments, review evidence, and audit claims.",
        config_sections=("experiment", "codex", "gpu", "tracking", "paths"),
        modules=(
            "agents.experiment_forge",
            "agents.experiment_supervisor",
            "agents.experiment_review",
            "agents.benchmark_audit",
            "agents.result_interpreter",
            "agents.evosci_requirements",
        ),
        scripts=(
            "scripts.audit_cggr_shard_contract",
            "scripts.audit_paper_benchmark_artifacts",
            "scripts.materialize_audited_cggr_results",
            "scripts.merge_cggr_method_shards",
            "scripts.prepare_cggr_top_venue_baseline_shard",
            "scripts.triage_cggr_audit_failure",
        ),
    ),
    AgentBoundary(
        key="experiment_execution",
        title="Experiment Execution Agent",
        folder="agents/experiment_execution",
        responsibility="Run validation loops, GPU jobs, remote shards, health checks, and merge watchers.",
        config_sections=("experiment", "gpu", "runtime", "tracking", "paths"),
        modules=(
            "agents.validation_loop",
            "agents.codex_executor",
            "orchestrator.gpu_scheduler",
            "orchestrator.ssh_gpu_backend",
            "orchestrator.benchmark_completion",
            "orchestrator.tracking",
        ),
        scripts=(
            "scripts.run_gpu_scheduler_forever",
            "scripts.stage_and_launch_cggr_top_venue_baseline_shard",
            "scripts.watch_and_merge_cggr_shards",
            "scripts.watch_cggr_live_health",
        ),
    ),
    AgentBoundary(
        key="manuscript_generation",
        title="Manuscript Generation Agent",
        folder="agents/manuscript_generation",
        responsibility="Generate, refine, audit, and bundle manuscripts, figures, references, and submission artifacts.",
        config_sections=("manuscript", "paper_orchestra", "llm", "paths"),
        modules=(
            "agents.manuscript_pipeline",
            "agents.paper_orchestra_pipeline",
            "agents.paper_orchestra_prompts",
            "agents.figure_agent",
            "agents.paperorchestra.full_pipeline",
            "agents.paperorchestra.figure_orchestra",
            "agents.paperorchestra.literature_discovery",
            "agents.paperorchestra.plotting_orchestra",
            "agents.paperorchestra.refinement_loop",
            "agents.paperorchestra.semantic_scholar",
        ),
        scripts=(
            "scripts.paperbanana_wrapper",
            "scripts.repair_manuscript_artifacts",
        ),
    ),
    AgentBoundary(
        key="orchestration",
        title="Orchestration Agent",
        folder="agents/orchestration",
        responsibility="Coordinate end-to-end jobs, background scheduling, workspace layout, web service, and deployment hooks.",
        config_sections=("app", "auto_research", "pipeline", "web", "paths", "database"),
        modules=(
            "agents.workspace_layout",
            "orchestrator.auto_research",
            "orchestrator.discovery_scheduler",
            "orchestrator.manuscript_watchdog",
            "orchestrator.paper_worker",
            "orchestrator.pipeline",
            "web.app",
        ),
        scripts=(
            "scripts.backfill_idea_workspaces",
            "scripts.migrate_sqlite_to_postgres",
            "scripts.run_pipeline_forever",
            "scripts.run_tunnel_forever",
            "scripts.run_web_forever",
        ),
    ),
)

AGENT_BOUNDARY_BY_KEY: dict[str, AgentBoundary] = {boundary.key: boundary for boundary in AGENT_BOUNDARIES}


def iter_agent_boundaries() -> tuple[AgentBoundary, ...]:
    return AGENT_BOUNDARIES


def get_agent_boundary(key: str) -> AgentBoundary:
    return AGENT_BOUNDARY_BY_KEY[key]

