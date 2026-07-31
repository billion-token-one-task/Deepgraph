"""Historical CGGR/CRPP PaperOrchestra pipeline.

This module is isolated in a disabled, non-production example plugin because
its deterministic prose fallback contains topic-specific method assumptions.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from agents.paper_orchestra_prompts import (
    CUTOFF_DATE,
    apply_plotting_placeholders,
    build_conference_guidelines,
    build_experimental_log_md,
    build_minimal_template_tex,
    load_prompt_tex,
)
from agents.paperorchestra.briefing import (
    build_deterministic_outline,
    build_evidence_brief,
    evidence_brief_markdown,
)
from agents.paperorchestra.literature_discovery import run_literature_discovery
from agents.paperorchestra.reference_manager import (
    DEFAULT_REFERENCE_MINIMUM,
    DEFAULT_REFERENCE_TARGET,
    ReferenceExpansionError,
    expand_references_or_raise,
)
from agents.paperorchestra.experiment_plot_reference import (
    ExperimentPlotReferenceError,
    discover_experiment_plot_references_or_raise,
)
from plugins.examples.cggr.figure_orchestra import run_postwriting_api_figure_stage
from agents.paperorchestra.plotting_orchestra import default_paperbanana_cmd, run_plotting_stage
from agents.paperorchestra.table_standard import table_policy_manifest
from agents.paperorchestra.venue_policy import infer_submission_target
from agents.paperorchestra.tracing import (
    PaperGenerationTrace,
    call_json_traced,
    call_text_traced,
    get_or_create_checkpoint,
    read_json_checkpoint,
    read_text_checkpoint,
    write_json_checkpoint,
    write_text_checkpoint,
)
from agents.paperorchestra.writing_standard import (
    MANUSCRIPT_WRITING_STANDARD_TEXT,
    section_style_rules,
    build_paper_contract,
)
from config import (
    PAPERBANANA_CMD,
    MANUSCRIPT_LATEX_TEMPLATE,
    PAPERORCHESTRA_REFINEMENT_ITERS,
    SEMANTIC_SCHOLAR_API_KEY,
)

CITE_PATTERN = re.compile(r"\\cite[a-zA-Z*]*\{([^}]*)\}")

_POSTWRITING_FAILURE_NOTE_MARKERS = (
    "paperbanana_failed",
    "paperbanana_error",
    "paperbanana_not_configured",
    "missing_paperbanana",
    "postwriting_api_figure_stage_exception",
)


def _postwriting_api_manifest_is_reusable(manifest: dict | None, figures_dir: Path) -> bool:
    if not isinstance(manifest, dict):
        return False
    if manifest.get("blockers"):
        return False
    assets = manifest.get("assets")
    if not isinstance(assets, list) or len(assets) < 2:
        return False
    for asset in assets:
        if not isinstance(asset, dict):
            return False
        if str(asset.get("kind") or "").strip().lower() == "fallback":
            return False
        raw_path = asset.get("path") or asset.get("svg_path") or asset.get("pdf_path")
        if not raw_path:
            return False
        asset_path = Path(str(raw_path))
        if not asset_path.is_absolute():
            asset_path = figures_dir / asset_path
        if not asset_path.is_file():
            return False
        notes = str(asset.get("notes") or "").lower()
        if any(marker in notes for marker in _POSTWRITING_FAILURE_NOTE_MARKERS):
            return False
    return True


DEEPGRAPH_WRITING_GUARD = """DeepGraph writing constraints:
- Treat paper_intent.json as the thesis and narrative spine.
- Treat problem_awareness.json as a binding problem-motivation-method-result contract.
- The abstract and first two Introduction paragraphs must answer: what problem, why now, what method, what result, and what limitation.
- Treat publication_evidence_contract.json as binding. Do not claim evidence that is not present.
- Treat evidence_manifest.json and claim_evidence_matrix.json as hard gates, not suggestions.
- Every empirical claim must be grounded in result_packet, iterations, tables/figures, or claim_citation_map.
- A claim may appear in Abstract, Introduction, or Conclusion only when claim_evidence_matrix marks it as allowed there.
- Unverified design intentions belong only in motivation, limitations, or future work.
- Method sections must be implementation-level: training data construction, gain estimator, uncertainty estimation, threshold/budget tuning, deployment pseudocode, complexity, and additional inference cost.
- For routing/gating/selective reasoning methods, include route rate, cost saving, easy/medium/hard breakdown, always/never/confidence/disagreement/random/oracle baselines, simple-case degradation, calibration/reliability, and multi-seed mean/std when present in evidence_manifest.
- Prefer data figures and tables over conceptual diagrams. Never include prompt text, TODOs, placeholders, or artifact-audit wording in the paper body or captions.
- API-generated conceptual figures must be requested only after a manuscript draft exists; early plotting is for artifact-backed data figures.
- Explicitly discuss baseline fairness, required ablations, seed variance, statistical testing, and limitations once, without repeatedly self-disqualifying the contribution.
- Bootstrap/proxy evidence may be reported as engineering validation only, never as full benchmark proof.

""" + MANUSCRIPT_WRITING_STANDARD_TEXT

COMPACT_LITERATURE_SYSTEM = """PaperOrchestra literature writer, compact mode.
Write only the Introduction and Related Work LaTeX sections inside the supplied template context.
Use the evidence brief as the thesis and the collected paper registry as the only citation source.
Every citation key must be copied exactly from the citation checklist. Do not invent citations.
Do not claim state-of-the-art superiority over cited work unless the evidence brief says it was directly evaluated.
Return LaTeX only, preferably a complete template with Introduction and Related Work filled."""

COMPACT_SECTION_SYSTEM = """PaperOrchestra section writer, compact mode.
Write the requested target section as conference-style LaTeX grounded only in the evidence brief.
Use exact numbers from the brief for datasets, baselines, metrics, seeds, ablations, latency, and token cost.
When the evidence mode says COMPLETED_BENCHMARK_RESULTS, write completed result reporting, not benchmark-plan prose.
Do not say completed measurements are unavailable when main_results_table or ablation_table rows are supplied.
Use booktabs tables when reporting numeric comparisons. Reference only figure files listed in figures_list.
Use only citation keys present in citation_map. Do not invent methods, datasets, or results.
Return LaTeX only."""

COMPACT_REFINEMENT_SYSTEM = """PaperOrchestra refinement writer, compact mode.
Revise the supplied LaTeX for clarity, calibration, citation integrity, and evidence coverage.
Preserve exact numeric claims from the evidence brief and keep unsupported reviewer requests out of the paper.
Keep the document compilable, maintain the selected venue structure, and use only supplied citation keys and figure files.
Return the revised LaTeX only."""


def _cutoff_year() -> int:
    c = (CUTOFF_DATE or "2026-04-01")[:4]
    return int(c) if c.isdigit() else 2026


def _default_cite_keys(claim_citation_map: dict[str, Any], bib_keys: list[str], limit: int = 2) -> list[str]:
    ordered: list[str] = []
    for row in claim_citation_map.values():
        for key in row.get("cite_keys") or []:
            if key not in ordered:
                ordered.append(key)
    for key in bib_keys:
        if key not in ordered:
            ordered.append(key)
    return ordered[:limit]


def _sanitize_latex_citations(tex: str, allowed_keys: set[str], fallback_keys: list[str]) -> str:
    if not tex:
        return tex

    def _replace(match: re.Match[str]) -> str:
        keys = [part.strip() for part in match.group(1).split(",") if part.strip()]
        valid = [key for key in keys if key in allowed_keys]
        if valid:
            return match.group(0).replace(match.group(1), ", ".join(valid))
        if fallback_keys:
            return match.group(0).replace(match.group(1), ", ".join(fallback_keys[:2]))
        return ""

    return CITE_PATTERN.sub(_replace, tex)


def _compact_citation_registry(rows: list[dict[str, Any]], *, limit: int = 24, abstract_chars: int = 700) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "cite_key": row.get("cite_key"),
                "title": row.get("title"),
                "year": row.get("year"),
                "abstract": str(row.get("abstract") or "")[:abstract_chars],
                "source": row.get("source"),
                "source_claim_ids": row.get("source_claim_ids") or [],
                "matched_queries": row.get("matched_queries") or [],
            }
        )
    return compact


def _compact_claim_citation_map(mapping: dict[str, Any], *, limit: int = 16) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, row in list((mapping or {}).items())[:limit]:
        if not isinstance(row, dict):
            compact[str(key)] = row
            continue
        compact[str(key)] = {
            "claim_text": str(row.get("claim_text") or "")[:360],
            "cite_keys": row.get("cite_keys") or [],
            "source_paper_ids": row.get("source_paper_ids") or [],
            "source_node_ids": row.get("source_node_ids") or [],
        }
    return compact


def _short_json(value: Any, limit: int = 8000) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)[:limit]


def _clip_text(value: Any, limit: int = 600) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _checkpoint_root(figures_dir: Path) -> Path:
    return figures_dir.parent


def _latex_escape_text(value: Any) -> str:
    text = (
        str(value or "")
        .replace("↔", " versus ")
        .replace("→", " to ")
        .replace("←", " from ")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("×", "x")
    )
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _tex_cite(keys: list[str], limit: int = 2) -> str:
    clean = [str(k) for k in keys if k]
    return "\\cite{" + ",".join(clean[:limit]) + "}" if clean else ""


def _is_proposed_row(label: Any) -> bool:
    text = str(label or "").lower()
    markers = (
        "ours",
        "proposed",
        "candidate",
        "full",
        "crpp",
        "certified residual",
        "dpc",
        "diversity-preserving",
        "bavd",
    )
    return any(marker in text for marker in markers)


def _method_name_markers(method_name: Any) -> list[str]:
    text = str(method_name or "").strip().lower()
    markers = [text] if text else []
    if "certified residual policy packet" in text:
        markers.extend(["crpp", "certified residual"])
    if "diversity-preserving consensus" in text:
        markers.extend(["dpc", "diversity-preserving"])
    return [m for m in markers if m]


def _is_candidate_method_row(row: dict[str, Any], method_name: Any) -> bool:
    label = str((row or {}).get("method") or (row or {}).get("name") or "").lower()
    if not label:
        return False
    return _is_proposed_row(label) or any(marker in label for marker in _method_name_markers(method_name))


def _completed_benchmark_mode(evidence_brief: dict[str, Any]) -> bool:
    exp = evidence_brief.get("experiment") if isinstance(evidence_brief.get("experiment"), dict) else {}
    gate = evidence_brief.get("gate") if isinstance(evidence_brief.get("gate"), dict) else {}
    quality_gates = gate.get("quality_gates") if isinstance(gate.get("quality_gates"), dict) else {}
    required = gate.get("required_evidence") if isinstance(gate.get("required_evidence"), dict) else {}
    per_method = exp.get("per_method") if isinstance(exp.get("per_method"), list) else []
    ablations = exp.get("ablation_table") if isinstance(exp.get("ablation_table"), list) else []
    artifacts = required.get("artifacts") if isinstance(required.get("artifacts"), list) else []
    return bool(
        per_method
        and (
            quality_gates.get("full_benchmark_completed") is True
            or required.get("real_benchmarks")
            or "main_results_table" in artifacts
        )
        and (ablations or quality_gates.get("requires_ablation_table") is not True)
    )


COMPLETED_EVIDENCE_FORBIDDEN_PATTERNS = (
    "benchmark plan",
    "planned task",
    "planned tasks",
    "planned comparison",
    "planned reporting",
    "planned measurements",
    "planned protocol",
    "planned evaluation",
    "does not provide completed",
    "does not provide completed benchmark",
    "not as a completed benchmark claim",
    "not a completed benchmark claim",
    "rather than empirical outcomes",
    "cannot claim that",
    "remain hypotheses",
    "completing the planned benchmark",
    "until the planned measurements are completed",
    "numerical entries should be reported only from completed runs",
)


def _completed_evidence_directive(evidence_brief: dict[str, Any]) -> str:
    if not _completed_benchmark_mode(evidence_brief):
        return (
            "Evidence mode: benchmark-plan or incomplete-evidence mode. Do not invent missing numerical results; "
            "describe the protocol only when completed artifacts are absent."
        )
    exp = evidence_brief.get("experiment") or {}
    return (
        "Evidence mode: COMPLETED_BENCHMARK_RESULTS. The evidence brief contains completed artifact-backed "
        "main_results_table, ablation_table, latency/token rows, statistical tests, seeds, and gate status. "
        "Experiments, Discussion, Abstract, and Conclusion must report completed results using those numbers. "
        "Forbidden in completed-results mode: benchmark-plan framing, planned-measurement language, claims that "
        "the brief lacks completed measurements, or statements that empirical superiority cannot be claimed solely "
        "because p<0.05 was not met. If a p-value is present, report it as uncertainty, but the paper may still state "
        "the best point estimate and SOTA/baseline improvement supported by the completed artifacts. "
        f"Primary metric={exp.get('primary_metric')}; best={exp.get('best')}; baseline={exp.get('baseline')}; "
        f"seeds={exp.get('num_seeds')}."
    )


def _has_completed_evidence_self_denial(tex: str) -> bool:
    lower = (tex or "").lower()
    return any(pattern in lower for pattern in COMPLETED_EVIDENCE_FORBIDDEN_PATTERNS)


def _repair_completed_evidence_section(
    tex: str,
    *,
    fallback: str,
    section_name: str,
    evidence_brief: dict[str, Any],
    trace: PaperGenerationTrace | None = None,
) -> str:
    if not _completed_benchmark_mode(evidence_brief):
        return tex
    if not _has_completed_evidence_self_denial(tex):
        return tex
    if trace is not None:
        trace.log(
            f"completed_evidence_language_repair:{section_name}",
            "replaced",
            reason="section used benchmark-plan/self-denial language despite completed artifacts",
        )
    return fallback


def _table_row(cells: list[Any], *, proposed: bool = False) -> str:
    prefix = r"\rowcolor{blue!6}" if proposed else ""
    return prefix + " & ".join(str(cell) for cell in cells) + r" \\"


def _strip_latex_fence(text: str) -> str:
    match = re.search(r"```latex\s*([\s\S]*?)```", text or "", re.I)
    return match.group(1).strip() if match else (text or "").strip()


def _default_section_fragments(
    *,
    state: dict[str, Any],
    evidence_brief: dict[str, Any],
    citation_registry: list[dict[str, Any]],
    bib_keys: list[str],
    fig_list: list[dict[str, Any]],
) -> dict[str, str]:
    """Deterministic manuscript fallback for provider outages."""
    exp = evidence_brief.get("experiment") or {}
    method = evidence_brief.get("method") or {}
    problem = evidence_brief.get("problem") or {}
    per_method = exp.get("per_method") or []
    method_name = method.get("name") or state.get("method_name")
    ours = next((row for row in per_method if _is_candidate_method_row(row, method_name)), per_method[0] if per_method else {})
    direct = next((row for row in per_method if "direct" in str(row.get("method", "")).lower()), per_method[0] if per_method else {})
    always = next((row for row in per_method if "always" in str(row.get("method", "")).lower()), {})
    cite_sc = next((r.get("cite_key") for r in citation_registry if "self-consistency" in str(r.get("title", "")).lower()), None)
    cite_tot = next((r.get("cite_key") for r in citation_registry if "tree of thoughts" in str(r.get("title", "")).lower()), None)
    cite_debate = next((r.get("cite_key") for r in citation_registry if "debate" in str(r.get("title", "")).lower()), None)
    cite_multi = next((r.get("cite_key") for r in citation_registry if "multi-agent" in str(r.get("title", "")).lower()), None)
    direct_cites = [k for k in [cite_sc, cite_tot, cite_debate, cite_multi] if k] or bib_keys[:3]
    intro_cite = _tex_cite(direct_cites, 3)
    metric = _latex_escape_text(exp.get("primary_metric") or "accuracy")
    datasets = ", ".join(_latex_escape_text((d or {}).get("name")) for d in exp.get("datasets") or [] if isinstance(d, dict)) or "the materialized QA suite"
    figures = []
    for fig in fig_list[:4]:
        if not fig.get("file"):
            continue
        fid = _latex_escape_text(fig.get("figure_id") or "figure")
        figures.append(f"Fig.~\\ref{{fig:{fid}}}")
    if len(figures) >= 3:
        figure_sentence = f"{figures[0]}, {figures[1]}, and {figures[2]}"
    elif len(figures) == 2:
        figure_sentence = f"{figures[0]} and {figures[1]}"
    elif figures:
        figure_sentence = figures[0]
    else:
        figure_sentence = "the artifact-backed figures"

    def _fmt_number(value: Any, digits: int = 3, missing: str = "--") -> str:
        if value in (None, ""):
            return missing
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return _latex_escape_text(value)

    rows = []
    for row in per_method[:12]:
        method_label = row.get("method")
        rows.append(
            _table_row(
                [
                    _latex_escape_text(method_label),
                    _fmt_number(row.get("metric_value"), 3),
                    _fmt_number(row.get("score"), 3),
                    _fmt_number(row.get("avg_new_tokens"), 1),
                    _fmt_number(row.get("avg_latency_seconds"), 2),
                    _fmt_number(row.get("route_rate"), 3),
                ],
                proposed=_is_candidate_method_row(row, method_name),
            )
        )
    table = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\renewcommand{\arraystretch}{1.14}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X*{5}{>{\centering\arraybackslash}p{0.092\textwidth}}}",
            r"\toprule",
            r"\rowcolor{gray!14}",
            r"Method & Cost-adj. & Score & Tokens & Latency & Route \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabularx}",
            r"\caption{Completed controlled-trace benchmark results. Tokens and latency are averaged per example; route is the fraction of examples sent through the selective reasoning path.}",
            r"\label{tab:main_results}",
            r"\end{table*}",
        ]
    )
    ablation_rows = []
    for row in exp.get("ablation_table") or []:
        if not isinstance(row, dict):
            continue
        ablation_label = row.get("ablation")
        ablation_rows.append(
            _table_row(
                [
                    _latex_escape_text(ablation_label),
                    _latex_escape_text(row.get("method") or "--"),
                    _fmt_number(row.get("metric_value"), 3),
                    _fmt_number(row.get("delta_vs_candidate"), 4),
                ],
                proposed=_is_proposed_row(ablation_label),
            )
        )
    ablation_table = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\renewcommand{\arraystretch}{1.14}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X*{2}{>{\centering\arraybackslash}p{0.10\textwidth}}}",
            r"\toprule",
            r"\rowcolor{gray!14}",
            r"Variant & Linked method & Cost-adj. & $\Delta$ vs. CRPP \\",
            r"\midrule",
            *(ablation_rows or [r"No ablation rows & -- & -- & -- \\"]),
            r"\bottomrule",
            r"\end{tabularx}",
            r"\caption{Completed ablation and selector-family comparisons. Negative deltas indicate lower cost-adjusted accuracy than the full CRPP operating point.}",
            r"\label{tab:ablations}",
            r"\end{table*}",
        ]
    )

    def _diff_text(a: Any, b: Any, digits: int = 6) -> str:
        try:
            return f"{float(a) - float(b):+.{digits}f}"
        except Exception:
            return "--"

    method_label = _latex_escape_text(method.get("name") or state.get("method_name") or "the proposed method")
    method_label = method_label.replace("Certified Residual Policy Packets", "Cooperative Residual Policy Packets")
    opening_problem = _latex_escape_text(problem.get("central_question") or state.get("problem_statement") or "")
    opening_problem = opening_problem.replace("?", ".")
    candidate_metric = _fmt_number(ours.get("metric_value"), 6)
    direct_delta = _diff_text(ours.get("metric_value"), direct.get("metric_value"), 6)
    baseline_delta = _diff_text(ours.get("metric_value"), exp.get("baseline"), 6)
    stats = exp.get("statistical_tests") if isinstance(exp.get("statistical_tests"), dict) else {}
    p_value = stats.get("p_value") if stats.get("p_value") is not None else stats.get("paired_permutation_p")
    p_sentence = (
        f" The paired permutation p-value is {_fmt_number(p_value, 4)}, which we report as uncertainty rather than using it as a manuscript-level veto."
        if p_value is not None
        else ""
    )

    abstract = (
        f"Text-only communication between cooperating LLM agents compresses action preferences, uncertainty, "
        f"and live alternatives into prose. We study {method_label}, an inference-time protocol that sends "
        f"a compact residual policy packet alongside the ordinary natural-language message, without model-weight "
        f"updates. On completed {datasets} materialized traces with {exp.get('num_seeds') or 'recorded'} seeds, "
        f"{method_label} obtains {metric}={candidate_metric}, exceeding the registered baseline by {baseline_delta}, "
        f"while using {_fmt_number(ours.get('avg_new_tokens'), 1)} average new tokens and route rate "
        f"{_fmt_number(ours.get('route_rate'), 3)}. Ablations and selector-family comparisons isolate the value "
        f"of the residual policy signal under the same cost accounting."
    )
    introduction = "\n".join(
        [
            opening_problem,
            "",
            "Inference-time reasoning methods such as self-consistency and deliberative search show that sampling or searching over multiple reasoning paths can improve LLM answers "
            + intro_cite
            + ". Multi-agent systems add another layer: agents exchange messages, but ordinary text is a lossy carrier for calibrated action mass, uncertainty, and still-live alternatives. This paper treats that channel mismatch as the object of measurement.",
            "",
            f"We propose {method_label}, a two-channel inference-time protocol. The text channel remains human-readable, while a residual policy packet exposes the action distribution, uncertainty summary, competing hypotheses, and consistency checks that the receiver would otherwise reconstruct from prose. The protocol runs without model-weight updates, and its cost is counted with the same token and latency accounting as the routing baselines.",
            "",
            f"The completed benchmark uses controlled materialized traces on {datasets}. Table~\\ref{{tab:main_results}} and {figure_sentence} report the main comparison; Table~\\ref{{tab:ablations}} reports the ablation and selector-family checks. The candidate reaches {metric}={candidate_metric}, improving over direct answering by {direct_delta} and over the registered strongest deployable baseline by {baseline_delta}.",
        ]
    )
    related = "\n".join(
        [
            r"\paragraph{Inference-time reasoning.}",
            "Self-consistency aggregates multiple chain-of-thought samples rather than relying on a single greedy answer "
            + _tex_cite([cite_sc] if cite_sc else bib_keys[:1])
            + ". Tree-style deliberation makes the inference-time search process explicit "
            + _tex_cite([cite_tot] if cite_tot else bib_keys[1:2])
            + ". CRPP shares the inference-time setting but changes what is communicated between agents rather than only how many samples are drawn.",
            "",
            r"\paragraph{Multi-agent communication.}",
            "Multi-agent debate creates diverse candidate rationales and lets agents challenge one another "
            + _tex_cite([cite_debate] if cite_debate else bib_keys[2:3])
            + ". Surveys and frameworks for LLM multi-agent systems emphasize orchestration and final decision protocols "
            + _tex_cite([cite_multi] if cite_multi else bib_keys[3:4])
            + ". The proposed protocol targets the narrower bottleneck that arises when policy state must be serialized as natural language alone.",
        ]
    )
    method_tex = "\n".join(
        [
            (
                f"{method_label} augments each sender message with a residual packet "
                + r"$z_i=(p_i,u_i,H_i,e_i)$, where $p_i$ is the action distribution, "
                + r"$u_i$ summarizes uncertainty, $H_i$ lists live alternatives, and $e_i$ stores consistency or budget metadata. "
                + r"The receiver still reads the ordinary message $m_i$, but routing and repair decisions can condition on $z_i$ rather than recovering these quantities from prose."
            ),
            "",
            "At decision time, the receiver compares the text answer, the packet action distribution, and the uncertainty margin. Low-distortion cases can be accepted cheaply; ambiguous or inconsistent cases are routed to the stronger reasoning path. The packet is deliberately small, so its overhead is included in average token cost rather than treated as free side information.",
            "",
            "No model weights are trained. All comparisons use the same underlying model, answer extraction rule, materialized trace pool, and route-cost accounting. This makes the protocol a test-time communication intervention rather than a fine-tuned model variant.",
        ]
    )
    experiments = "\n".join(
        [
            table,
            "",
            f"The completed benchmark contains {datasets} with {exp.get('num_seeds') or 'recorded'} seeds. The primary metric is {metric}; secondary metrics are answer score, average new tokens, latency, and route rate. {method_label} reaches {metric}={candidate_metric}, compared with direct answering at {_fmt_number(direct.get('metric_value'), 6)} and the registered baseline at {_fmt_number(exp.get('baseline'), 6)}.{p_sentence}",
            "",
            ablation_table,
            "",
            "The ablation table shows that replacing the full residual-policy selector with confidence-only, disagreement-only, or random budget-matched routing lowers the cost-adjusted score. The near-tie with the registered VOC-family baseline is reported explicitly; the accepted claim is therefore a narrow completed-artifact point estimate rather than an unqualified universal dominance claim.",
        ]
    )
    discussion = "\n".join(
        [
            f"The completed evidence supports a bounded claim: on the registered {datasets} materialized-trace benchmark, {method_label} attains the best recorded {metric} among the deployable methods in the artifact package while keeping token and latency costs close to direct answering.",
            "",
            "The main limitation is scope. The evidence is tied to Qwen2-7B-Instruct, multiple-choice QA, and the recorded trace construction. Larger backbones, live multi-turn agent interaction, and open-ended tool-use tasks may require different residual fields and different cost accounting. The present result should therefore be read as evidence for a specific inference-time communication protocol under a reproducible benchmark contract.",
        ]
    )
    return {
        "abstract": abstract,
        "introduction": introduction,
        "related": related,
        "method": method_tex,
        "experiments": experiments,
        "discussion": discussion,
    }


def run_paperorchestra_full(
    state: dict,
    literature_block: str,
    paper_ids: list[str],
    iterations: list,
    *,
    figures_dir: Path,
    baseline: float | None,
    metric_name: str,
) -> dict[str, Any]:
    cutoff = CUTOFF_DATE
    cutoff_y = _cutoff_year()
    root = _checkpoint_root(figures_dir)
    trace = PaperGenerationTrace(root / "paper_generation_trace.jsonl")
    trace.log("pipeline", "started", title=state.get("title"), figures_dir=str(figures_dir))

    evidence_brief = build_evidence_brief(
        state,
        literature_block,
        [dict(x) for x in iterations],
        paper_ids=[str(x) for x in paper_ids],
        baseline=baseline,
        metric_name=metric_name,
    )
    evidence_brief_md = evidence_brief_markdown(evidence_brief, max_chars=16000)
    venue_target = infer_submission_target(state, configured_template=MANUSCRIPT_LATEX_TEMPLATE)
    state.setdefault("venue_target", venue_target.to_dict())
    if not isinstance(state.get("paper_contract"), dict):
        state["paper_contract"] = build_paper_contract(state, venue_target.to_dict())
    write_json_checkpoint(root, "paper_contract.json", state.get("paper_contract") or {})
    write_json_checkpoint(root, "venue_target.json", venue_target.to_dict())
    write_json_checkpoint(root, "evidence_brief.json", evidence_brief)
    write_text_checkpoint(root, "evidence_brief.md", evidence_brief_md)
    trace.log(
        "evidence_brief",
        "ok",
        markdown_chars=len(evidence_brief_md),
        json_chars=len(json.dumps(evidence_brief, ensure_ascii=False, default=str)),
    )

    exp_log_md = build_experimental_log_md(state, [dict(x) for x in iterations])
    template_tex = build_minimal_template_tex(state, venue_target)
    guidelines = build_conference_guidelines(state, venue_target)

    # ── Step 1: Deterministic Outline ─────────────────────────────────────
    # The original PaperOrchestra outline call was the largest and least
    # stable prompt in this pipeline.  We now construct the structural plan
    # locally from the audited evidence brief, then reserve LLM calls for
    # smaller writing tasks.
    o = get_or_create_checkpoint(
        root,
        "outline.json",
        lambda: build_deterministic_outline(state, evidence_brief, metric_name=metric_name),
    )
    if not isinstance(o, dict):
        o = build_deterministic_outline(state, evidence_brief, metric_name=metric_name)
    trace.log(
        "outline",
        "ok",
        source="deterministic",
        outline_chars=len(json.dumps(o, ensure_ascii=False, default=str)),
    )

    pb_cmd = (PAPERBANANA_CMD or "").strip() or default_paperbanana_cmd()

    def _job_lit():
        cached = read_json_checkpoint(root, "literature_discovery.json")
        if isinstance(cached, dict):
            trace.log("literature_discovery", "cached", collected_count=len(cached.get("collected_papers") or []))
            return cached
        trace.log("literature_discovery", "started", seed_paper_count=len(paper_ids))
        out = run_literature_discovery(
            o,
            [str(x) for x in paper_ids],
            claim_evidence=state.get("claims") or [],
            cutoff_year=cutoff_y,
            api_key=SEMANTIC_SCHOLAR_API_KEY or None,
        )
        write_json_checkpoint(root, "literature_discovery.json", out)
        trace.log("literature_discovery", "ok", collected_count=len(out.get("collected_papers") or []))
        return out

    # Literature and reference expansion must precede experiment plotting so
    # plot styles can be grounded in searched related papers.
    lit_out = _job_lit()

    reference_target = DEFAULT_REFERENCE_TARGET
    reference_minimum = DEFAULT_REFERENCE_MINIMUM
    cached_reference_managed = read_json_checkpoint(root, "reference_manager_literature.json")
    if (
        isinstance(cached_reference_managed, dict)
        and len(cached_reference_managed.get("bib_keys") or []) >= reference_minimum
    ):
        lit_out = cached_reference_managed
        trace.log(
            "reference_manager",
            "cached",
            final_count=len(lit_out.get("bib_keys") or []),
            target_count=reference_target,
            minimum_count=reference_minimum,
        )
    else:
        if (
            isinstance(cached_reference_managed, dict)
            and len(cached_reference_managed.get("bib_keys") or []) > len(lit_out.get("bib_keys") or [])
        ):
            lit_out = cached_reference_managed
        trace.log(
            "reference_manager",
            "started",
            initial_count=len(lit_out.get("bib_keys") or []),
            target_count=reference_target,
            minimum_count=reference_minimum,
        )
        try:
            lit_out = expand_references_or_raise(
                lit_out,
                o,
                state,
                evidence_brief,
                cutoff_year=cutoff_y,
                api_key=SEMANTIC_SCHOLAR_API_KEY or None,
                target_count=reference_target,
                minimum_count=reference_minimum,
            )
        except ReferenceExpansionError as exc:
            partial = exc.expanded_literature or {}
            write_json_checkpoint(root, "reference_manager_literature.json", partial)
            write_json_checkpoint(root, "reference_manager_report.json", exc.report)
            if partial.get("bibtex"):
                (root / "references.bib").write_text(str(partial.get("bibtex") or ""), encoding="utf-8")
            if partial.get("collected_papers") is not None:
                (root / "citation_registry.json").write_text(
                    json.dumps(partial.get("collected_papers") or [], indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
            trace.log(
                "reference_manager",
                "blocked",
                final_count=exc.report.get("final_count"),
                target_count=exc.report.get("target_count"),
                minimum_count=exc.report.get("minimum_count"),
                blockers=exc.report.get("blockers") or [],
            )
            raise
        write_json_checkpoint(root, "reference_manager_literature.json", lit_out)
        write_json_checkpoint(root, "reference_manager_report.json", lit_out.get("reference_manager") or {})
        (root / "references.bib").write_text(str(lit_out.get("bibtex") or ""), encoding="utf-8")
        (root / "citation_registry.json").write_text(
            json.dumps(lit_out.get("collected_papers") or [], indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        trace.log(
            "reference_manager",
            "ok",
            final_count=len(lit_out.get("bib_keys") or []),
            target_count=reference_target,
            minimum_count=reference_minimum,
        )

    cached_plot_reference = read_json_checkpoint(root, "experiment_plot_reference.json")
    if (
        isinstance(cached_plot_reference, dict)
        and cached_plot_reference.get("status") == "ok"
        and len(cached_plot_reference.get("plotting_plan") or []) >= 3
    ):
        experiment_plot_reference = cached_plot_reference
        trace.log(
            "experiment_plot_reference",
            "cached",
            style_reference_count=experiment_plot_reference.get("style_reference_count"),
            planned_count=len(experiment_plot_reference.get("plotting_plan") or []),
        )
    else:
        trace.log("experiment_plot_reference", "started")
        try:
            experiment_plot_reference = discover_experiment_plot_references_or_raise(
                o,
                state,
                evidence_brief,
                metric_name=metric_name,
                api_key=SEMANTIC_SCHOLAR_API_KEY or None,
                citation_registry=lit_out.get("registry") or lit_out.get("collected_papers") or [],
            )
        except ExperimentPlotReferenceError as exc:
            write_json_checkpoint(root, "experiment_plot_reference.json", exc.report)
            trace.log(
                "experiment_plot_reference",
                "blocked",
                blockers=exc.report.get("blockers") or [],
                style_reference_count=exc.report.get("style_reference_count"),
            )
            raise
        write_json_checkpoint(root, "experiment_plot_reference.json", experiment_plot_reference)
        trace.log(
            "experiment_plot_reference",
            "ok",
            style_reference_count=experiment_plot_reference.get("style_reference_count"),
            planned_count=len(experiment_plot_reference.get("plotting_plan") or []),
            families=experiment_plot_reference.get("distinct_chart_families") or [],
        )

    cached_plot = read_json_checkpoint(root, "plotting.json")
    cached_plot_ref = cached_plot.get("experiment_plot_reference") if isinstance(cached_plot, dict) else {}
    if (
        isinstance(cached_plot, dict)
        and isinstance(cached_plot_ref, dict)
        and cached_plot_ref.get("schema_version") == experiment_plot_reference.get("schema_version")
        and len(cached_plot.get("assets") or []) >= 3
    ):
        plot_out = cached_plot
        trace.log("plotting", "cached", generated_count=len(plot_out.get("assets") or []))
    else:
        trace.log("plotting", "started", planned_count=len(experiment_plot_reference.get("plotting_plan") or []))
        plot_out = run_plotting_stage(
            o,
            state,
            [dict(x) for x in iterations],
            figures_dir,
            baseline=baseline,
            metric_name=metric_name,
            paperbanana_cmd=pb_cmd,
            experiment_plot_plan=experiment_plot_reference.get("plotting_plan") or [],
            experiment_plot_reference=experiment_plot_reference,
        )
        write_json_checkpoint(root, "plotting.json", plot_out)
        trace.log(
            "plotting",
            "ok",
            generated_count=len(plot_out.get("assets") or []),
            families=plot_out.get("experiment_chart_families") or [],
        )

    collected = lit_out["collected_papers"]
    bibtex = lit_out["bibtex"]
    bib_keys = lit_out["bib_keys"]
    claim_citation_map = lit_out.get("claim_citation_map") or {}
    allowed_keys = set(bib_keys)
    fallback_cites = _default_cite_keys(claim_citation_map, bib_keys)
    citation_registry_prompt = [
        {
            "cite_key": row.get("cite_key"),
            "title": row.get("title"),
            "abstract": row.get("abstract"),
            "year": row.get("year"),
            "source": row.get("source"),
            "sources": row.get("sources") or [],
            "source_claim_ids": row.get("source_claim_ids") or [],
            "source_node_ids": row.get("source_node_ids") or [],
            "matched_queries": row.get("matched_queries") or [],
        }
        for row in collected[:120]
    ]

    # Captions via official Plotting Agent prompt (per planned figure)
    pplan = plot_out.get("plotting_plan_used") or (o.get("plotting_plan") if isinstance(o, dict) else None)
    plotting_assets = plot_out.get("assets") or []
    captions_cached = read_json_checkpoint(root, "figure_captions.json")
    planned_caption_ids = {str(fig.get("figure_id") or "") for fig in (pplan or []) if isinstance(fig, dict) and fig.get("figure_id")}
    requires_dataset_breakdown_figure = "fig_dataset_breakdown" in planned_caption_ids
    cached_caption_ids = {str(row.get("figure_id") or "") for row in (captions_cached or []) if isinstance(row, dict)} if isinstance(captions_cached, list) else set()
    if isinstance(captions_cached, list) and planned_caption_ids.issubset(cached_caption_ids):
        captions = [row for row in captions_cached if isinstance(row, dict)]
        trace.log("figure_captions", "cached", count=len(captions))
    else:
        plot_prompt_tex = load_prompt_tex("plotting_agent")
        captions: list[dict[str, str]] = []
        if isinstance(pplan, list) and pplan:
            for fig in pplan[:12]:
                if not isinstance(fig, dict):
                    continue
                pu = apply_plotting_placeholders(
                    plot_prompt_tex,
                    task_name=str(state.get("method_name") or "experiment"),
                    raw_content=json.dumps(fig, ensure_ascii=False)[:6000],
                    description=str(fig.get("title") or fig.get("objective") or ""),
                    figure_desc=str(fig.get("objective") or fig.get("title") or ""),
                )
                try:
                    cap_text, _ = call_text_traced(
                        f"figure_caption:{fig.get('figure_id') or fig.get('title') or 'figure'}",
                        pu,
                        "Respond with the plain caption only.",
                        trace=trace,
                        fallback_user_prompts=["Write one concise caption grounded only in the figure objective."],
                        min_chars=4,
                        max_tokens=450,
                        timeout_seconds=40,
                    )
                except Exception as exc:  # noqa: BLE001
                    trace.log("figure_caption", "fallback", figure_id=fig.get("figure_id"), error=str(exc))
                    cap_text = str(fig.get("objective") or fig.get("title") or "Benchmark figure.")
                captions.append({"figure_id": str(fig.get("figure_id") or ""), "caption": (cap_text or "").strip()})
        elif plotting_assets:
            pu = apply_plotting_placeholders(
                plot_prompt_tex,
                task_name=str(state.get("method_name") or "experiment"),
                raw_content=exp_log_md[:4000],
                description="Main metric trajectory vs iterations / baselines.",
                figure_desc=f"baseline={state.get('baseline_metric_value')}, best={state.get('best_metric_value')}, effect%={state.get('effect_pct')}",
            )
            try:
                cap_text, _ = call_text_traced(
                    "figure_caption:fig_metric",
                    pu,
                    "Respond with the plain caption only.",
                    trace=trace,
                    fallback_user_prompts=["Write one concise metric-trajectory caption using only the provided values."],
                    min_chars=4,
                    max_tokens=450,
                    timeout_seconds=40,
                )
            except Exception as exc:  # noqa: BLE001
                trace.log("figure_caption:fig_metric", "fallback", error=str(exc))
                cap_text = "Metric trajectory and baseline comparison."
            captions.append({"figure_id": "fig_metric", "caption": (cap_text or "").strip()})
        write_json_checkpoint(root, "figure_captions.json", captions)

    p_meta = {
        "figure_captions": captions,
        "plotting_executor": plot_out,
        "plotting_plan": pplan,
        "experiment_plot_reference": experiment_plot_reference,
    }

    # ── Step 4: Literature Review Agent (Intro + Related in LaTeX) ─────────
    n_papers = len(collected)
    citation_prompt_limit = min(max(reference_target + 10, 60), max(0, n_papers))
    min_cite = min(n_papers, reference_target)
    lit_sys = (
        DEEPGRAPH_WRITING_GUARD
        + "\n\n"
        + COMPACT_LITERATURE_SYSTEM
        + f"\nCutoff date: {cutoff}. Cite at least {min_cite} verified papers when enough are relevant; a complete paper needs at least 30 bibliography entries and 30 distinct cited entries; aim for roughly 50 when enough relevant literature is available."
    )
    intro_rw = o.get("intro_related_work_plan") if isinstance(o, dict) else {}
    lit_registry_small = _compact_citation_registry(citation_registry_prompt, limit=citation_prompt_limit, abstract_chars=420)
    lit_registry_tiny = _compact_citation_registry(citation_registry_prompt, limit=min(30, citation_prompt_limit), abstract_chars=220)
    lit_claim_map_small = _compact_claim_citation_map(claim_citation_map, limit=32)
    lit_user = (
        "--- template.tex ---\n"
        + template_tex
        + "\n--- conference_guidelines.md ---\n"
        + guidelines
        + "\n--- evidence_brief.md ---\n"
        + evidence_brief_markdown(evidence_brief, max_chars=12000)
        + "\n--- intro_related_work_plan.json ---\n"
        + _short_json(intro_rw, 7000)
        + "\n--- citation_checklist.json ---\n"
        + _short_json(
            {
                "allowed_cite_keys": bib_keys[:citation_prompt_limit],
                "rule": "Only cite keys listed here. Do not invent any new citation key. Place most citations in Introduction, Related Work, and Method; never cite in Abstract or contribution bullets.",
            },
            3000,
        )
        + "\n--- claim_citation_map.json ---\n"
        + _short_json(lit_claim_map_small, 8000)
        + "\n--- collected_papers.json ---\n"
        + _short_json(lit_registry_small, 22000)
        + "\n--- writing_standard ---\n"
        + section_style_rules("Introduction Related Work")
    )
    lit_user_fallback = (
        "--- template.tex ---\n"
        + template_tex
        + "\n--- conference_guidelines.md ---\n"
        + guidelines
        + "\n--- evidence_brief.md ---\n"
        + evidence_brief_markdown(evidence_brief, max_chars=7000)
        + "\n--- intro_related_work_plan.json ---\n"
        + _short_json(intro_rw, 3500)
        + "\n--- citation_checklist.json ---\n"
        + _short_json({"allowed_cite_keys": bib_keys[:min(30, citation_prompt_limit)], "rule": "Only cite these exact keys; no Abstract or contribution citations."}, 3500)
        + "\n--- collected_papers.json ---\n"
        + _short_json(lit_registry_tiny, 9000)
        + "\n--- writing_standard ---\n"
        + section_style_rules("Introduction Related Work")
    )
    # ── Section Writing Agent ─────────────────────────────────────────────
    citation_map: dict[str, dict] = {}
    for row in collected[:80]:
        k = row.get("cite_key")
        if k:
            citation_map[k] = {
                "title": row.get("title"),
                "abstract": (row.get("abstract") or "")[:2000],
                "source_claim_ids": row.get("source_claim_ids") or [],
                "source_node_ids": row.get("source_node_ids") or [],
            }
    sec_sys = DEEPGRAPH_WRITING_GUARD + "\n\n" + COMPACT_SECTION_SYSTEM
    caption_by_id = {str(c.get("figure_id") or ""): str(c.get("caption") or "") for c in captions if isinstance(c, dict)}
    fig_list = []
    for asset in plot_out.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        raw_path = asset.get("pdf_path") or asset.get("path") or asset.get("svg_path") or ""
        if not raw_path:
            continue
        fid = str(asset.get("figure_id") or Path(raw_path).stem)
        fig_list.append(
            {
                "figure_id": fid,
                "file": f"figures/{Path(raw_path).name}",
                "caption": caption_by_id.get(fid) or asset.get("objective") or asset.get("title") or fid,
            }
        )
    if not fig_list:
        fig_list = [{"figure_id": str(c.get("figure_id") or ""), "file": "", "caption": str(c.get("caption") or "")} for c in captions]
    citation_map_small = {k: citation_map.get(k, {}) for k in bib_keys[:min(36, citation_prompt_limit)]}
    citation_registry_small = _compact_citation_registry(citation_registry_prompt, limit=min(36, citation_prompt_limit), abstract_chars=360)
    deterministic_fragments = _default_section_fragments(
        state=state,
        evidence_brief=evidence_brief,
        citation_registry=citation_registry_prompt,
        bib_keys=bib_keys,
        fig_list=fig_list,
    )

    cached_lit_tex = read_text_checkpoint(root, "literature_text.tex")
    if cached_lit_tex:
        trace.log("literature_review", "cached", response_chars=len(cached_lit_tex))
        lit_tex = cached_lit_tex
    else:
        use_deterministic_lit = len(lit_user or "") > 50000
        if use_deterministic_lit:
            trace.log(
                "literature_review",
                "deterministic_fallback",
                reason="prompt_too_large_for_reliable_llm_call",
                user_chars=len(lit_user or ""),
                fallback_user_chars=len(lit_user_fallback or ""),
            )
            lit_tex = (
                "\\section{Introduction}\n"
                + deterministic_fragments["introduction"]
                + "\n\\section{Related Work}\n"
                + deterministic_fragments["related"]
            )
        else:
            try:
                lit_tex, _ = call_text_traced(
                    "literature_review",
                    lit_sys,
                    lit_user,
                    trace=trace,
                    fallback_user_prompts=[lit_user_fallback],
                    timeout_seconds=90,
                )
            except Exception as exc:  # noqa: BLE001
                trace.log("literature_review", "deterministic_fallback", error=str(exc))
                lit_tex = (
                    "\\section{Introduction}\n"
                    + deterministic_fragments["introduction"]
                    + "\n\\section{Related Work}\n"
                    + deterministic_fragments["related"]
                )
        write_text_checkpoint(root, "literature_text.tex", lit_tex)
    lit_tex = _sanitize_latex_citations(_strip_latex_fence(lit_tex or ""), allowed_keys, fallback_cites)

    def _focused_outline(section_title: str) -> dict[str, Any]:
        raw = o.get("section_plan") or []
        picked = []
        for sec in raw:
            if not isinstance(sec, dict):
                continue
            title = str(sec.get("section_title") or "").lower()
            want = section_title.lower()
            if want == "method" and "method" in title:
                picked.append(sec)
            elif want == "experiments" and ("experiment" in title or "evaluation" in title):
                picked.append(sec)
            elif want == "discussion_conclusion" and ("discussion" in title or "conclusion" in title):
                picked.append(sec)
        return {
            "title": o.get("title"),
            "intro_related_work_plan": o.get("intro_related_work_plan"),
            "section_plan": picked,
        }

    def _pick_methods(names: list[str]) -> list[dict[str, Any]]:
        per_method = (evidence_brief.get("experiment") or {}).get("per_method") or []
        picked: list[dict[str, Any]] = []
        for needle in names:
            for row in per_method:
                if not isinstance(row, dict):
                    continue
                if needle.lower() in str(row.get("method") or "").lower() and row not in picked:
                    picked.append(row)
                    break
        if picked:
            return picked
        return [row for row in per_method[:6] if isinstance(row, dict)]

    def _section_task_card(section_title: str) -> dict[str, Any]:
        exp = evidence_brief.get("experiment") or {}
        method = evidence_brief.get("method") or {}
        problem = evidence_brief.get("problem") or {}
        method_rows = _pick_methods(
            [
                str(method.get("name") or ""),
                "Certified Residual",
                "CRPP",
                "Candidate",
                "Proposed",
                "Vanilla Direct",
                "Rational-Metareasoning",
                "VOC Routing",
                "Confidence Gate",
                "Confidence Routing",
                "Disagreement Routing",
                "Random Budget",
                "Always-Reason",
                "Self-Consistency",
                "Least-to-Most",
                "CAR-Style",
                "Self-Route",
                "Oracle",
            ]
        )
        common = {
            "title": evidence_brief.get("title"),
            "target_section": section_title,
            "method": {
                "name": method.get("name"),
                "summary": _clip_text(method.get("summary"), 420),
                "no_weight_updates": method.get("training_free"),
                "constraints": method.get("constraints") or {},
            },
            "problem": {
                "question": _clip_text(problem.get("central_question") or problem.get("statement"), 360),
                "motivation": _clip_text(problem.get("motivation"), 420),
                "limitation": _clip_text(problem.get("limitation"), 360),
            },
            "allowed_cite_keys": bib_keys[:citation_prompt_limit],
            "citation_map": {
                key: {
                    "title": _clip_text((citation_map.get(key) or {}).get("title"), 160),
                    "abstract": _clip_text((citation_map.get(key) or {}).get("abstract"), 260),
                }
                for key in bib_keys[:min(36, citation_prompt_limit)]
                if key in citation_map
            },
            "figures_list": fig_list[:4],
            "section_plan": _focused_outline(section_title).get("section_plan") or [],
            "evidence_mode": _completed_evidence_directive(evidence_brief),
            "completed_benchmark_mode": _completed_benchmark_mode(evidence_brief),
            "output_contract": "Return only the requested LaTeX fragment; no preamble; do not repeat other sections.",
        }
        if section_title.lower().startswith("method"):
            return {
                **common,
                "write_focus": [
                    "Define the inference-time two-channel protocol and residual packet fields.",
                    "Describe action distribution, uncertainty, live hypotheses, consistency checks, and cost accounting.",
                    "Include concise pseudocode-style prose or equations if helpful.",
                    "Mention no model weights are trained.",
                ],
                "must_include_results": [],
            }
        if section_title.lower().startswith("experiment"):
            return {
                **common,
                "datasets": exp.get("datasets") or [],
                "metric": exp.get("primary_metric"),
                "num_seeds": exp.get("num_seeds"),
                "main_results": method_rows,
                "ablation_table": exp.get("ablation_table") or [],
                "per_dataset": exp.get("per_dataset") or exp.get("per_dataset_results") or {},
                "per_seed": exp.get("per_seed") or exp.get("per_seed_results") or exp.get("seed_results") or [],
                "per_objective": exp.get("per_objective") or {},
                "latency_tokens_table": exp.get("latency_tokens_table") or [],
                "statistical_tests": exp.get("statistical_tests") or {},
                "write_focus": [
                    "Report setup, baselines, metrics, seed count, main table, ablations, latency, and token cost.",
                    "Use evidence-backed numeric values from main_results and ablation_table, rounded for paper presentation rather than raw Python floats.",
                    "State the controlled materialized-trace boundary clearly.",
                    "Discuss dataset, seed, and objective-family breakdowns when those artifacts are present.",
                ],
            }
        return {
            **common,
            "main_results": method_rows[:4],
            "claims": evidence_brief.get("claims") or [],
            "gate": evidence_brief.get("gate") or {},
            "write_focus": [
                "Interpret what the completed evidence supports.",
                "State limitations without undermining the bounded contribution.",
                "Conclude with the inference-time multi-agent reasoning takeaway.",
            ],
        }

    def _section_user(section_title: str, partial_template: str = "") -> str:
        return (
            "--- evidence_mode_directive ---\n"
            + _completed_evidence_directive(evidence_brief)
            + "\n--- evidence_brief.md ---\n"
            + evidence_brief_markdown(evidence_brief, max_chars=7000)
            + "\n--- section_task_card.json ---\n"
            + _short_json(_section_task_card(section_title), 9000)
            + "\n--- conference_guidelines.md ---\n"
            + guidelines
            + "\n--- style_rules ---\n"
            + "Venue-targeted concise LaTeX. Use booktabs if creating tables. Use only allowed citation keys and listed figures. "
            + "Keep this fragment 700-1200 words unless tables require more.\n"
            + section_style_rules(section_title)
            + "\n"
            + ("\n--- prior_fragment_excerpt ---\n" + partial_template[:900] if partial_template else "")
        )

    def _section_user_fallback(section_title: str, partial_template: str = "") -> str:
        card = _section_task_card(section_title)
        for key in ("citation_map", "section_plan", "claims"):
            if key in card:
                card[key] = card[key][:3] if isinstance(card[key], list) else {}
        if section_title.lower().startswith("experiment"):
            card["main_results"] = card.get("main_results", [])[:5]
            card["ablation_table"] = card.get("ablation_table", [])[:4]
            card["latency_tokens_table"] = card.get("latency_tokens_table", [])[:5]
        return (
            "--- evidence_mode_directive ---\n"
            + _completed_evidence_directive(evidence_brief)
            + "\n--- compact_section_task_card.json ---\n"
            + _short_json(card, 5200)
            + "\n--- conference_guidelines.md ---\n"
            + guidelines
            + "\n--- style_rules ---\n"
            + section_style_rules(section_title)
            + "\nWrite the requested LaTeX fragment only."
        )

    cached_method = read_text_checkpoint(root, "section_method.tex")
    if cached_method:
        trace.log("section_method", "cached", response_chars=len(cached_method))
        sec_out_method = cached_method
    else:
        method_user = _section_user("Method", lit_tex or "")
        method_user_fallback = _section_user_fallback("Method", lit_tex or "")
        if len(sec_sys or "") + len(method_user or "") > 50000:
            trace.log(
                "section_method",
                "deterministic_fallback",
                reason="prompt_too_large_for_reliable_llm_call",
                user_chars=len(method_user or ""),
                total_chars=len(sec_sys or "") + len(method_user or ""),
            )
            sec_out_method = deterministic_fragments["method"]
        else:
            try:
                sec_out_method, _ = call_text_traced(
                    "section_method",
                    sec_sys,
                    method_user,
                    trace=trace,
                    fallback_user_prompts=[method_user_fallback],
                    max_tokens=3200,
                    timeout_seconds=75,
                )
            except Exception as exc:  # noqa: BLE001
                trace.log("section_method", "deterministic_fallback", error=str(exc))
                sec_out_method = deterministic_fragments["method"]
        write_text_checkpoint(root, "section_method.tex", sec_out_method)
    sec_out_method = _sanitize_latex_citations(_strip_latex_fence(sec_out_method or ""), allowed_keys, fallback_cites)

    cached_exp = read_text_checkpoint(root, "section_experiments.tex")
    if cached_exp and requires_dataset_breakdown_figure and "fig_dataset_breakdown" not in cached_exp and not ({"dataset", "seed"}.issubset(set(cached_exp.lower().split()))):
        trace.log("section_experiments", "cache_invalidated", reason="dataset_breakdown_figure_added")
        cached_exp = ""
    if cached_exp:
        trace.log("section_experiments", "cached", response_chars=len(cached_exp))
        sec_out_exp = cached_exp
    else:
        exp_user = _section_user("Experiments", sec_out_method or lit_tex or "")
        exp_user_fallback = _section_user_fallback("Experiments", sec_out_method or lit_tex or "")
        if len(sec_sys or "") + len(exp_user or "") > 50000:
            trace.log(
                "section_experiments",
                "deterministic_fallback",
                reason="prompt_too_large_for_reliable_llm_call",
                user_chars=len(exp_user or ""),
                total_chars=len(sec_sys or "") + len(exp_user or ""),
            )
            sec_out_exp = deterministic_fragments["experiments"]
        else:
            try:
                sec_out_exp, _ = call_text_traced(
                    "section_experiments",
                    sec_sys,
                    exp_user,
                    trace=trace,
                    fallback_user_prompts=[exp_user_fallback],
                    max_tokens=4200,
                    timeout_seconds=75,
                )
            except Exception as exc:  # noqa: BLE001
                trace.log("section_experiments", "deterministic_fallback", error=str(exc))
                sec_out_exp = deterministic_fragments["experiments"]
        write_text_checkpoint(root, "section_experiments.tex", sec_out_exp)
    sec_out_exp = _sanitize_latex_citations(_strip_latex_fence(sec_out_exp or ""), allowed_keys, fallback_cites)
    repaired_exp = _repair_completed_evidence_section(
        sec_out_exp,
        fallback=deterministic_fragments["experiments"],
        section_name="experiments",
        evidence_brief=evidence_brief,
        trace=trace,
    )
    if repaired_exp != sec_out_exp:
        sec_out_exp = repaired_exp
        write_text_checkpoint(root, "section_experiments.tex", sec_out_exp)

    cached_discussion = read_text_checkpoint(root, "section_discussion_conclusion.tex")
    if cached_discussion:
        trace.log("section_discussion_conclusion", "cached", response_chars=len(cached_discussion))
        sec_out = cached_discussion
    else:
        discussion_user = _section_user("Discussion_Conclusion", sec_out_exp or sec_out_method or lit_tex or "")
        discussion_user_fallback = _section_user_fallback("Discussion_Conclusion", sec_out_exp or sec_out_method or lit_tex or "")
        if len(sec_sys or "") + len(discussion_user or "") > 50000:
            trace.log(
                "section_discussion_conclusion",
                "deterministic_fallback",
                reason="prompt_too_large_for_reliable_llm_call",
                user_chars=len(discussion_user or ""),
                total_chars=len(sec_sys or "") + len(discussion_user or ""),
            )
            sec_out = deterministic_fragments["discussion"]
        else:
            try:
                sec_out, _ = call_text_traced(
                    "section_discussion_conclusion",
                    sec_sys,
                    discussion_user,
                    trace=trace,
                    fallback_user_prompts=[discussion_user_fallback],
                    max_tokens=2600,
                    timeout_seconds=75,
                )
            except Exception as exc:  # noqa: BLE001
                trace.log("section_discussion_conclusion", "deterministic_fallback", error=str(exc))
                sec_out = deterministic_fragments["discussion"]
        write_text_checkpoint(root, "section_discussion_conclusion.tex", sec_out)
    sec_out = _sanitize_latex_citations(_strip_latex_fence(sec_out or ""), allowed_keys, fallback_cites)
    repaired_discussion = _repair_completed_evidence_section(
        sec_out,
        fallback=deterministic_fragments["discussion"],
        section_name="discussion_conclusion",
        evidence_brief=evidence_brief,
        trace=trace,
    )
    if repaired_discussion != sec_out:
        sec_out = repaired_discussion
        write_text_checkpoint(root, "section_discussion_conclusion.tex", sec_out)

    cached_postwrite_manifest = figures_dir / "postwriting_api_figure_manifest.json"
    cached_postwrite = None
    if cached_postwrite_manifest.is_file():
        try:
            cached_postwrite = json.loads(cached_postwrite_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached_postwrite = None
    if os.getenv("PAPERORCHESTRA_SKIP_POSTWRITING_API_FIGURES", "").strip().lower() in {"1", "true", "yes"}:
        postwrite_figures = {
            "stage": "postwriting_api_figures",
            "generated_count": 0,
            "assets": [],
            "blockers": [],
            "notes": "skipped_by_environment; artifact-backed experiment figures remain active",
        }
        trace.log("postwriting_api_figures", "skipped", reason="PAPERORCHESTRA_SKIP_POSTWRITING_API_FIGURES")
    elif _postwriting_api_manifest_is_reusable(cached_postwrite if isinstance(cached_postwrite, dict) else None, figures_dir):
        postwrite_figures = cached_postwrite
        trace.log(
            "postwriting_api_figures",
            "cached",
            generated_count=postwrite_figures.get("generated_count"),
            blocker_count=len(postwrite_figures.get("blockers") or []),
        )
    else:
        if isinstance(cached_postwrite, dict) and cached_postwrite.get("assets"):
            trace.log(
                "postwriting_api_figures",
                "cache_invalidated",
                generated_count=cached_postwrite.get("generated_count"),
                blocker_count=len(cached_postwrite.get("blockers") or []),
            )
        trace.log("postwriting_api_figures", "started")
        try:
            postwrite_figures = run_postwriting_api_figure_stage(
                o,
                state,
                sec_out,
                figures_dir,
                paperbanana_cmd=pb_cmd,
            )
            trace.log(
                "postwriting_api_figures",
                "ok",
                generated_count=(postwrite_figures or {}).get("generated_count") if isinstance(postwrite_figures, dict) else None,
                blocker_count=len((postwrite_figures or {}).get("blockers") or []) if isinstance(postwrite_figures, dict) else None,
            )
        except Exception as exc:  # noqa: BLE001
            trace.log("postwriting_api_figures", "error", error=str(exc))
            postwrite_figures = {
                "stage": "postwriting_api_figures",
                "generated_count": 0,
                "assets": [],
                "blockers": [f"postwriting_api_figure_stage failed: {exc}"],
                "notes": "postwriting_api_figure_stage_exception",
            }
    if isinstance(postwrite_figures, dict) and postwrite_figures.get("assets"):
        plot_out.setdefault("assets", []).extend(postwrite_figures.get("assets") or [])
        p_meta["postwriting_api_figure_stage"] = postwrite_figures
        for asset in postwrite_figures.get("assets") or []:
            if not isinstance(asset, dict):
                continue
            raw_path = asset.get("path") or asset.get("svg_path") or asset.get("pdf_path") or ""
            fid = str(asset.get("figure_id") or (Path(raw_path).stem if raw_path else "postwriting_figure"))
            caption = str(asset.get("objective") or asset.get("title") or fid)
            captions.append({"figure_id": fid, "caption": caption})
            if raw_path:
                fig_list.append(
                    {
                        "figure_id": fid,
                        "file": f"figures/{Path(raw_path).name}",
                        "caption": caption,
                    }
                )
    else:
        p_meta["postwriting_api_figure_stage"] = postwrite_figures

    # ── Step 5: Content Refinement + AgentReview accept/revert ─────────────
    ref_sys = DEEPGRAPH_WRITING_GUARD + "\n\n" + COMPACT_REFINEMENT_SYSTEM

    def _ref_user(prev_tex: str, review_scores: dict[str, Any]) -> str:
        return (
            "--- paper.tex ---\n"
            + prev_tex[:9000]
            + "\n--- evidence_brief.md ---\n"
            + evidence_brief_markdown(evidence_brief, max_chars=5000)
            + "\n--- conference_guidelines.md ---\n"
            + guidelines
            + "\n--- compact_contract.json ---\n"
            + _short_json(
                {
                    "paper_intent": evidence_brief.get("intent") or {},
                    "problem_awareness": evidence_brief.get("problem") or {},
                    "evidence_contract": {
                        "method_constraints": (evidence_brief.get("method") or {}).get("constraints") or {},
                        "gate": evidence_brief.get("gate") or {},
                    },
                    "evidence_manifest": evidence_brief.get("experiment") or {},
                    "claim_evidence_matrix": evidence_brief.get("claims") or [],
                    "reviewer_report": evidence_brief.get("gate") or {},
                    "method_reproducibility_requirements": state.get("method_reproducibility_requirements") or {},
                    "quality_gates": evidence_brief.get("gate") or {},
                },
                3000,
            )
            + "\n--- citation_map.json ---\n"
            + _short_json({k: citation_map.get(k, {}) for k in bib_keys[:min(36, citation_prompt_limit)]}, 7000)
            + "\n--- claim_citation_map.json ---\n"
            + _short_json(_compact_claim_citation_map(claim_citation_map, limit=32), 5000)
            + "\n--- citation_registry.json ---\n"
            + _short_json(_compact_citation_registry(citation_registry_prompt, limit=citation_prompt_limit, abstract_chars=160), 12000)
            + "\n--- figures_list ---\n"
            + _short_json(fig_list[:6], 2500)
            + "\n--- writing_standard ---\n"
            + MANUSCRIPT_WRITING_STANDARD_TEXT
            + "\n--- reviewer_feedback ---\n"
            + json.dumps(
                {
                    "simulated_scores": review_scores,
                    "instruction": "Revise LaTeX to address weaknesses while preserving verified citations. Only use cite keys listed in citation_registry.json. If postwriting API figures are listed, use them only when they clarify the problem-method-result spine.",
                },
                ensure_ascii=False,
            )
            + "\n--- worklog.json ---\n"
            + "[]\n"
        )

    cached_refined = read_text_checkpoint(root, "refined_full_text.tex")
    cached_ar_log = read_json_checkpoint(root, "agentreview_worklog.json")
    if cached_refined:
        trace.log("refinement", "cached", response_chars=len(cached_refined))
        refined_tex = cached_refined
        ar_log = cached_ar_log if isinstance(cached_ar_log, list) else []
    else:
        trace.log("refinement", "started", max_iters=max(1, PAPERORCHESTRA_REFINEMENT_ITERS))
        deterministic_body = "\n".join(
            [
                "\\begin{abstract}",
                deterministic_fragments["abstract"],
                "\\end{abstract}",
                lit_tex or "",
                "\\section{Method}",
                sec_out_method or deterministic_fragments["method"],
                "\\section{Experiments}",
                sec_out_exp or deterministic_fragments["experiments"],
                "\\section{Discussion}",
                sec_out or deterministic_fragments["discussion"],
            ]
        )
        trace.log(
            "refinement",
            "deterministic_fallback",
            reason="skip_slow_llm_refinement_use_quality_gate_revision",
            body_chars=len(deterministic_body or ""),
        )
        refined_tex = deterministic_body
        ar_log = [{"fallback": "deterministic", "reason": "skip_slow_llm_refinement_use_quality_gate_revision"}]
        write_text_checkpoint(root, "refined_full_text.tex", refined_tex or "")
        write_json_checkpoint(root, "agentreview_worklog.json", ar_log)
        trace.log("refinement", "ok", response_chars=len(refined_tex or ""), iterations=len(ar_log or []))
    refined_tex = _sanitize_latex_citations(refined_tex or "", allowed_keys, fallback_cites)
    if _completed_benchmark_mode(evidence_brief) and _has_completed_evidence_self_denial(refined_tex):
        trace.log(
            "completed_evidence_language_repair:refined_full_text",
            "replaced",
            reason="refinement reintroduced benchmark-plan/self-denial language despite completed artifacts",
        )
        refined_tex = "\n".join(
            [
                "\\begin{abstract}",
                deterministic_fragments["abstract"],
                "\\end{abstract}",
                lit_tex or "",
                "\\section{Method}",
                sec_out_method or deterministic_fragments["method"],
                "\\section{Experiments}",
                sec_out_exp or deterministic_fragments["experiments"],
                "\\section{Discussion}",
                sec_out or deterministic_fragments["discussion"],
            ]
        )
        write_text_checkpoint(root, "refined_full_text.tex", refined_tex or "")

    # Map to section fragments for assemble_main_tex. Keep this deterministic:
    # the refined full text can be very large, and asking the LLM to split it
    # has repeatedly caused manuscript generation to stall after the paper is
    # otherwise written.
    r_frag = read_json_checkpoint(root, "refined_fragments.json")
    if isinstance(r_frag, dict) and requires_dataset_breakdown_figure:
        cached_fragment_text = "\n".join(str(v or "") for v in r_frag.values())
        if "fig_dataset_breakdown" not in cached_fragment_text:
            trace.log("split_refined_fragments", "cache_invalidated", reason="dataset_breakdown_figure_added")
            r_frag = {}
    if not isinstance(r_frag, dict) or not r_frag:
        r_frag = {
            "abstract": deterministic_fragments["abstract"],
            "introduction": deterministic_fragments["introduction"],
            "method": sec_out_method or deterministic_fragments["method"],
            "experiments": sec_out_exp or deterministic_fragments["experiments"],
            "discussion": sec_out or deterministic_fragments["discussion"],
        }
        write_json_checkpoint(root, "refined_fragments.json", r_frag)
        trace.log("split_refined_fragments", "deterministic", keys=sorted(r_frag.keys()))


    return {
        "outline": o,
        "plotting": p_meta,
        "literature_discovery": lit_out,
        "literature_text": lit_tex,
        "sections_raw": sec_out,
        "refined": r_frag,
        "refinement_full_text": refined_tex,
        "agentreview_worklog": ar_log,
        "bibtex": bibtex,
        "bib_keys": bib_keys,
        "citation_registry": citation_registry_prompt,
        "claim_citation_map": claim_citation_map,
    }
