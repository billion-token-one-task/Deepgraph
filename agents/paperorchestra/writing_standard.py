"""Binding manuscript-writing standards for PaperOrchestra outputs."""

from __future__ import annotations

from agents.manuscript_length_policy import build_length_standard_text
from agents.paper_title_policy import TITLE_NAMING_STANDARD_TEXT
from agents.reference_auditor import REFERENCE_AUDIT_STANDARD_TEXT
from agents.visual_layout_auditor import VISUAL_LAYOUT_STANDARD_TEXT


WRITING_STANDARD_VERSION = "paperorchestra_manuscript_writing_standard_v4_2026_06_10"


ABSTRACT_STANDARD = """Abstract standard:
- Follow this narrative order: background trend -> concrete problem -> limitation of existing methods -> proposed method -> core mechanism -> main results.
- Results must appear only in the final one or two abstract sentences. Do not start with evaluation or numbers.
- The first one or two sentences must enter the concrete problem directly; avoid long generic preambles.
- When the method name first appears, write the full name and abbreviation.
- The first meaningful use of domain abbreviations in the abstract must use full form plus abbreviation, e.g. large language models (LLMs), retrieval-augmented generation (RAG), exact match (EM), question answering (QA).
- Do not over-explain method details in the abstract.
- Do not make unsupported broad generalization claims.
- Do not expose non-core details that weaken presentation, such as tiny sample counts, temporary protocols, or unfinished experiments.
- State results naturally using standard effect language such as percentage points, absolute improvement, relative reduction, or cost reduction.
- If evidence is from a subset, controlled setting, materialized-trace benchmark, or case study, the abstract must state that boundary and must not imply a complete benchmark.
- Avoid excessive colons, semicolons, rhetorical questions, and multi-clause sentences.
- Do not duplicate the abstract, title, or section heading; exactly one abstract environment is allowed.
- Forbidden unless explicitly supported by strong evidence: "This is the first", "Comprehensive experiments show", "Extensive experiments demonstrate", "Universal", "General"."""


INTRODUCTION_STANDARD = """Introduction standard:
- The Introduction must do four jobs: explain why the problem matters, explain why existing methods are insufficient, state what this paper solves, and summarize how the method solves it with supported results.
- Preferred structure: background trend; concrete challenge; two or three core weaknesses of existing methods; proposed method and corresponding mechanism; short result paragraph; contribution summary.
- Do not add standalone Question/Motivation/Answer/Result mini-headings or rhetorical-question paragraphs.
- If using Problem I / II / III framing, each problem must correspond to a real failure mode, the total number should normally be at most three, and every problem must have a matching design in the Method.
- Do not split problems mechanically just to create a numbered structure, and keep problem paragraphs compact.
- After introducing multiple problems, respond to them explicitly with sentences such as "For Problem I, ..." and state the mechanism and why it mitigates the problem.
- The result paragraph should be short, typically two or three sentences, and must not repeat the abstract verbatim."""


CONTRIBUTIONS_STANDARD = """Contribution standard:
- Use three or four contribution bullets.
- Preferred order: identify or define the problem; formulate the task or analyze the failure mode; propose the method; evaluate with completed evidence.
- Each contribution should be a plain sentence beginning with We identify/formulate/propose/construct/evaluate; do not use bold or italic mini labels inside bullets.
- Only claim work that is actually completed by this paper.
- Avoid vague claims, inflated claims, or repackaging ordinary experiments as major contributions.
- Do not use the phrase "training-free"; say "inference-time" or "without model-weight updates" only when that concrete scope is necessary.
- Avoid "We conduct extensive experiments", "We are the first to", and "We comprehensively study" unless the evidence truly supports them."""


RELATED_WORK_STANDARD = """Related Work standard:
- Related Work should position the paper, not dump citations.
- Each subsection should follow: area introduction -> representative method categories -> relationship or difference from this paper.
- Subsection titles should be Title Case noun phrases, one to three words, short enough for two-column layout, and grammatically consistent.
- Organize citations by category. Do not write large undifferentiated citation clusters such as \\cite{a,b,c,d,e,f,g,h}.
- Preferred citation pattern: describe one category and cite one or two papers, then describe another category and cite another one or two papers.
- Each Related Work subsection should end with a gap sentence: what this paper does differently, what it adds, and what it is not trying to solve.
- By default, keep each Related Work subsection to one dense paragraph unless the paper is a survey or has unusually large space."""


METHOD_STANDARD = """Method standard:
- The Method section should make the method understandable and reproducible.
- Use a structure appropriate to the paper, usually three to five subsections such as Problem Formulation, Framework Design, Core Algorithm, Inference/Optimization, and Implementation Details.
- For protocol, benchmark, or analysis papers, prefer a progression such as Problem Settings / Benchmark Matrix -> Core Mechanism -> Reporting or Inference Rule -> Protocol Summary.
- Avoid too many subsections, avoid moving experimental protocol into Method, and avoid baseline evaluation inside Method.
- Every important equation must be followed by an explanation of the input, output, variables, purpose of each term, and which problem the equation addresses.
- Except for theory papers, keep Method display equations to about three or four numbered equation environments; avoid unnumbered $$ display math.
- Prefer intuitive notation and readable piecewise definitions over scattered indicator functions when possible.
- If the method has a clear procedure, include an algorithm block with explicit inputs and outputs, at most 15--20 lines, consistent with the equations.
- Algorithm blocks must use a real LaTeX algorithm environment, preferably algorithm+algpseudocode/algorithmic or algorithm2e; do not fake pseudocode with center/minipage/enumerate/textbf blocks.
- Algorithm blocks must not contain experiment results and must not be an empty generic subsection.
- The Method section must not contain experiment numbers, significance analysis, baseline criticism, TODOs, undefined variables, or assumptions inconsistent with experiments."""


PAPER_CONTRACT_STANDARD = """Paper Contract standard:
- Before writing, create a paper_contract.json that binds paper type, contribution type, evidence scope, supported claims, primary metrics, terminology glossary, target venue/journal, and banned expressions.
- Abstract, Introduction, Experiments, Discussion, and Conclusion must use the same evidence scope: full benchmark, subset, controlled setting, case study, simulation, live evaluation, or cross-model evaluation.
- No claim may appear unless it is supported by a method definition, completed experiment, figure/table, citation, or claim-evidence matrix row.
- Do not present unfinished, placeholder, simulated, smoke-test, or expected results as completed evidence.
- Method names, dataset names, baseline names, metric names, and experiment protocol labels must remain consistent across all sections."""


ABBREVIATION_STANDARD = """Abbreviation and terminology standard:
- In each major section, the first meaningful use of key abbreviations must be full form plus abbreviation.
- This applies independently to Abstract, Introduction, Method, and Experiments.
- Common required expansions include large language model(s) (LLM/LLMs), retrieval-augmented generation (RAG), person re-identification (Re-ID), visible-infrared (VI), exact match (EM), expected calibration error (ECE), and question answering (QA).
- Subsection titles must not contain unexplained bare abbreviations."""


EXPERIMENTS_STANDARD = """Experiments standard:
- Follow Setup -> Main Results -> Ablations -> optional Efficiency/Robustness/Case Study/Failure Analysis.
- Setup must state datasets, metrics, baselines, protocol, implementation/inference settings, repeated runs or seeds, statistical method, and definitions of table metrics.
- Main Results must compare against the strongest baseline, explain why results occur, and state uncertainty or significance when available.
- If the Method has multiple named components or the contract requires ablations, include an ablation subsection/table grounded in completed artifacts.
- If p_value >= 0.05 or the verdict is inconclusive, do not claim SOTA, broad superiority, or validation."""


TABLE_FIGURE_STANDARD = """Tables and figures standard:
- Tables must use booktabs with top/mid/bottom rules, consistent numeric precision, clear captions, and no raw Python floats.
- Figure plans must serve the paper spine. Motivation figures show the existing failure and why the problem matters; method figures show inputs, modules, flow, outputs, and core mechanism.
- Experiment figures should prioritize grouped bars, line charts, heatmaps, radar charts, scatter plots, and Pareto/frontier curves from real artifacts.
- Avoid duplicate figures, decorative conceptual diagrams, 3D charts, excessive gradients, unreadable legends, and large blank areas."""


DISCUSSION_LIMITATIONS_STANDARD = """Discussion, Limitations, and Conclusion standard:
- Discussion interprets what the completed evidence supports, when the method is useful, and the tradeoff against strong baselines. It must not introduce new results, methods, or unsupported claims.
- Limitations should be honest but not self-destructive: state method limitations and evidence-scope limitations without overturning the main supported claim.
- Conclusion is short: problem recap -> method recap -> core empirical conclusion -> significance. It must not introduce new tasks, new data, new claims, new terms, or new citations."""


POSTPROCESS_STANDARD = """Required automatic post-processing checks:
- Claim-evidence consistency for Abstract, Introduction, Contributions, Discussion, and Conclusion.
- Scope consistency across Abstract, Introduction, Experiments, Discussion, and Conclusion.
- Numeric consistency across abstract numbers, intro numbers, tables, captions, figures, and conclusion.
- LaTeX integrity for tables, labels, figure paths, percent signs, duplicate labels, and unclosed environments.
- Venue-target consistency: selected venue or journal controls template, page-limit language, anonymity, bibliography style, and watchdog checks.
- Length-auditor and reference-auditor findings are binding. High-severity findings must return the manuscript to revision rather than being ignored."""


MANUSCRIPT_WRITING_STANDARD_TEXT = "\n\n".join(
    [
        f"Binding manuscript writing standard: {WRITING_STANDARD_VERSION}.",
        TITLE_NAMING_STANDARD_TEXT,
        build_length_standard_text(),
        REFERENCE_AUDIT_STANDARD_TEXT,
        VISUAL_LAYOUT_STANDARD_TEXT,
        ABSTRACT_STANDARD,
        INTRODUCTION_STANDARD,
        CONTRIBUTIONS_STANDARD,
        RELATED_WORK_STANDARD,
        METHOD_STANDARD,
        PAPER_CONTRACT_STANDARD,
        ABBREVIATION_STANDARD,
        EXPERIMENTS_STANDARD,
        TABLE_FIGURE_STANDARD,
        DISCUSSION_LIMITATIONS_STANDARD,
        POSTPROCESS_STANDARD,
    ]
)


def section_style_rules(section_title: str) -> str:
    """Return section-specific style rules for compact section-writing prompts."""
    title = (section_title or "").lower()
    if "intro" in title or "related" in title:
        return INTRODUCTION_STANDARD + "\n\n" + CONTRIBUTIONS_STANDARD + "\n\n" + RELATED_WORK_STANDARD + "\n\n" + ABBREVIATION_STANDARD
    if "method" in title:
        return METHOD_STANDARD + "\n\n" + ABBREVIATION_STANDARD
    if "experiment" in title or "result" in title:
        return EXPERIMENTS_STANDARD + "\n\n" + TABLE_FIGURE_STANDARD + "\n\n" + ABBREVIATION_STANDARD
    if "discussion" in title or "conclusion" in title:
        return DISCUSSION_LIMITATIONS_STANDARD
    return MANUSCRIPT_WRITING_STANDARD_TEXT


def build_paper_contract(state: dict, venue_target: dict | None = None) -> dict:
    """Construct a compact binding paper contract from manuscript state."""
    paper_intent = state.get("paper_intent") if isinstance(state.get("paper_intent"), dict) else {}
    publication_contract = (
        state.get("publication_evidence_contract")
        if isinstance(state.get("publication_evidence_contract"), dict)
        else {}
    )
    result_packet = state.get("result_packet") if isinstance(state.get("result_packet"), dict) else {}
    benchmark_summary = result_packet.get("benchmark_summary") if isinstance(result_packet.get("benchmark_summary"), dict) else {}
    method_name = state.get("method_name") or benchmark_summary.get("candidate_method") or state.get("title")
    datasets = benchmark_summary.get("datasets") or []
    baselines = list((benchmark_summary.get("per_method") or {}).keys()) if isinstance(benchmark_summary.get("per_method"), dict) else []
    metric = benchmark_summary.get("primary_metric") or benchmark_summary.get("metric_name") or state.get("baseline_metric_name")
    evidence_scope = (
        publication_contract.get("evidence_tier")
        or result_packet.get("evidence_tier")
        or ("full benchmark" if result_packet.get("full_benchmark_completed") else "controlled setting")
    )
    glossary = {
        "method_name": method_name,
        "datasets": [row.get("name") if isinstance(row, dict) else str(row) for row in datasets],
        "baselines": baselines,
        "primary_metric": metric,
    }
    return {
        "schema_version": "deepgraph_paper_contract_v1",
        "paper_type": paper_intent.get("paper_type") or publication_contract.get("paper_type") or "method paper",
        "contribution_type": paper_intent.get("contribution_type") or publication_contract.get("contribution_type") or "new method",
        "target": venue_target or {},
        "evidence_scope": evidence_scope,
        "supported_claims": state.get("claim_evidence_matrix") or state.get("claims") or [],
        "primary_metric": metric,
        "glossary": glossary,
        "banned_expressions": [
            "training-free",
            "This is the first",
            "Comprehensive experiments show",
            "Extensive experiments demonstrate",
            "Universal",
            "General",
            "state-of-the-art",
            "SOTA",
        ],
        "consistency_requirements": [
            "Do not expand evidence scope in abstract or conclusion.",
            "Do not write unfinished or placeholder numbers as completed results.",
            "Use the same names for methods, datasets, baselines, metrics, and protocol across sections.",
        ],
    }
