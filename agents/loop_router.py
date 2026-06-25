"""Route closed-loop blockers to the agent/stage that can repair them.

The rest of the pipeline should not have to infer meaning from free-form
failure strings.  This module keeps the mapping deterministic and conservative:
it classifies known blocker phrases into a small set of repair owners, while
leaving unknown issues in a general experiment-design loop.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "loop_router_v1"


_ROUTE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "category": "benchmark_literature_design",
        "stage": "benchmark_literature_design",
        "owner": "Benchmark Design Agent",
        "action": (
            "rerun literature-grounded benchmark selection with official dataset "
            "sources, benchmark-set rationale, and paper evidence"
        ),
        "markers": (
            "benchmark_set_rationale",
            "literature_sources",
            "literature review required",
            "paper or official benchmark sources",
            "dataset count",
            "dataset selection evidence",
            "benchmark design",
            "benchmark family",
            "gsm8k is only allowed",
            "generic gsm8k",
            "all gsm8k",
        ),
    },
    {
        "category": "dataset_materialization",
        "stage": "dataset_materialization",
        "owner": "Dataset Fetch Agent",
        "action": (
            "resolve the official dataset source, pin revisions/files, materialize "
            "the required splits, and write schema/count manifests"
        ),
        "markers": (
            "dataset",
            "hf_dataset",
            "huggingface",
            "hf-mirror",
            "metadata probe",
            "materialize",
            "materialized",
            "download",
            "cache",
            "arrow",
            "split names",
            "official source",
            "not found",
            "does not exist",
            "cannot access",
            "timeout",
            "lock file",
            "schema-checked",
        ),
    },
    {
        "category": "benchmark_harness",
        "stage": "benchmark_harness_required",
        "owner": "Benchmark Harness Code Agent",
        "action": (
            "generate or adapt the custom benchmark runner/evaluator, then run "
            "harness review before GPU execution"
        ),
        "markers": (
            "dedicated benchmark harness",
            "custom harness",
            "benchmark harness required",
            "generated runner cannot execute",
            "generated real-benchmark runner",
            "generated-runner",
            "task_type=",
            "unsupported task",
            "runner emits",
            "harness review",
        ),
    },
    {
        "category": "benchmark_completion",
        "stage": "benchmark_completion_required",
        "owner": "Benchmark Completion Runner",
        "action": (
            "run or repair the full benchmark package so required baselines, seeds, "
            "ablations, metrics, and artifacts are complete"
        ),
        "markers": (
            "benchmark_artifact_manifest",
            "full_benchmark_completed",
            "benchmark summary",
            "benchmark evidence",
            "required baseline",
            "required baselines",
            "required model coverage",
            "per_method",
            "num_seeds",
            "seed(s)",
            "ablation",
            "full benchmark policy",
        ),
    },
    {
        "category": "reference_expansion",
        "stage": "reference_expansion",
        "owner": "Reference Manager",
        "action": "expand and verify bibliography metadata, citations, DOI/arXiv/URL fields, and cited claims",
        "markers": (
            "referenceexpansionerror",
            "reference auditor",
            "bibliography",
            "citation",
            "citations",
            "doi",
            "arxiv",
            "bibtex",
            "missing reference",
            "uncited",
        ),
    },
    {
        "category": "figure_regeneration",
        "stage": "figure_regeneration",
        "owner": "Plotting/Figure Agent",
        "action": "regenerate figures, plot references, captions, and linked result artifacts",
        "markers": (
            "experimentplotreferenceerror",
            "figure",
            "plot",
            "caption",
            "image",
            "missing figure",
            "plotting",
        ),
    },
    {
        "category": "latex_compile",
        "stage": "latex_compile_repair",
        "owner": "LaTeX Compile Agent",
        "action": "repair LaTeX/package/formatting errors and rebuild the manuscript PDF",
        "markers": (
            "latex",
            "pdflatex",
            "undefined control sequence",
            "missing $",
            "float.sty",
            "algorithmicx",
            "algpseudocode",
            "bibtex",
        ),
    },
    {
        "category": "manuscript_quality",
        "stage": "manuscript_quality_repair",
        "owner": "Paper Quality Agent",
        "action": "rerun manuscript quality repair after upstream evidence, references, and figures are ready",
        "markers": (
            "quality gate",
            "paper_quality_report",
            "writing guideline",
            "submission blocker",
            "manuscript",
            "conference-paper",
            "abstract",
            "related work",
        ),
    },
    {
        "category": "evosci_report",
        "stage": "evoscientist_research",
        "owner": "EvoScientist Research Agent",
        "action": "rerun or repair EvoScientist novelty/deep-research inputs and final_report generation",
        "markers": (
            "evoscientist",
            "evosci",
            "final_report",
            "novelty",
            "deep research",
            "verification",
        ),
    },
    {
        "category": "execution_code_repair",
        "stage": "execution_code_repair",
        "owner": "Execution Repair Agent",
        "action": "repair runtime/dependency/proof errors and rerun validation from the failing stage",
        "markers": (
            "traceback",
            "exception",
            "modulenotfounderror",
            "importerror",
            "syntaxerror",
            "runtimeerror",
            "compile failed",
            "test failed",
            "load_failures",
            "code repair",
            "proof repair",
        ),
    },
    {
        "category": "stale_run_invalidation",
        "stage": "stale_run_invalidation",
        "owner": "Run State Auditor",
        "action": "mark stale/legacy invalid runs failed or archived before scheduling replacement work",
        "markers": (
            "stale",
            "legacy",
            "invalid benchmark design",
            "invalid_benchmark_design",
            "old gsm8k manifest",
            "residual process",
        ),
    },
]

_DEFAULT_ROUTE = {
    "category": "experiment_semantic_design",
    "stage": "experiment_semantic_design_repair",
    "owner": "Experiment Design Agent",
    "action": "repair the experiment plan semantics, baselines, metrics, and claim-evidence contract",
    "markers": (),
}

_PRIORITY = {
    "stale_run_invalidation": 0,
    "benchmark_literature_design": 10,
    "dataset_materialization": 20,
    "benchmark_harness": 30,
    "benchmark_completion": 40,
    "experiment_semantic_design": 50,
    "execution_code_repair": 60,
    "evosci_report": 70,
    "reference_expansion": 80,
    "figure_regeneration": 90,
    "latex_compile": 100,
    "manuscript_quality": 110,
}


def _text(value: Any) -> str:
    if isinstance(value, Mapping):
        parts = [
            value.get("standard"),
            value.get("severity"),
            value.get("issue"),
            value.get("summary"),
            value.get("evidence"),
            value.get("message"),
            value.get("error"),
        ]
        return " ".join(str(part or "").strip() for part in parts if str(part or "").strip()).strip()
    return str(value or "").strip()


def _dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _definition_for_text(text: str) -> tuple[dict[str, Any], list[str]]:
    lower = text.lower()
    best: dict[str, Any] | None = None
    best_markers: list[str] = []
    best_priority = 9999
    for definition in _ROUTE_DEFINITIONS:
        markers = [marker for marker in definition["markers"] if marker in lower]
        if not markers:
            continue
        priority = _PRIORITY.get(definition["category"], 999)
        if priority < best_priority:
            best = definition
            best_markers = markers
            best_priority = priority
    return dict(best or _DEFAULT_ROUTE), best_markers


def classify_blocker(blocker: Any, *, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Classify one free-form blocker into a repair route."""

    text = _text(blocker)
    definition, markers = _definition_for_text(text)
    route = {
        "blocker": text,
        "category": definition["category"],
        "stage": definition["stage"],
        "owner": definition["owner"],
        "action": definition["action"],
        "severity": "blocking" if text else "unknown",
        "matched_markers": markers,
    }
    if context:
        source = str(context.get("source") or "").strip()
        if source:
            route["source"] = source
    return route


def route_blockers(blockers: Iterable[Any] | None, *, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Group blockers by owner and expose the next loop action."""

    items = _dedupe(_text(item) for item in (blockers or []))
    routes = [classify_blocker(item, context=context) for item in items]
    routes.sort(key=lambda row: (_PRIORITY.get(row["category"], 999), row["blocker"].lower()))

    owner_map: dict[str, dict[str, Any]] = {}
    for route in routes:
        owner = route["owner"]
        bucket = owner_map.setdefault(
            owner,
            {
                "owner": owner,
                "stage": route["stage"],
                "categories": [],
                "blockers": [],
                "next_actions": [],
            },
        )
        if route["category"] not in bucket["categories"]:
            bucket["categories"].append(route["category"])
        bucket["blockers"].append(route["blocker"])
        if route["action"] not in bucket["next_actions"]:
            bucket["next_actions"].append(route["action"])

    owner_routes = list(owner_map.values())
    primary = routes[0] if routes else None
    next_actions = _dedupe(route["action"] for route in routes)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "actionable" if routes else "clear",
        "blocked": bool(routes),
        "primary_stage": primary["stage"] if primary else "",
        "primary_owner": primary["owner"] if primary else "",
        "primary_action": primary["action"] if primary else "",
        "routes": routes,
        "owner_routes": owner_routes,
        "next_actions": next_actions,
        "summary": (
            f"{primary['owner']} -> {primary['action']}"
            if primary
            else "No loop blockers detected."
        ),
    }


def compact_loop_note(route_report: Mapping[str, Any] | None) -> str:
    """Return a short human-readable owner/action note for DB status fields."""

    if not isinstance(route_report, Mapping) or not route_report.get("blocked"):
        return ""
    owner = str(route_report.get("primary_owner") or "").strip()
    action = str(route_report.get("primary_action") or "").strip()
    stage = str(route_report.get("primary_stage") or "").strip()
    if owner and action:
        return f"Loop owner: {owner}; stage={stage}; next={action}."
    return ""
