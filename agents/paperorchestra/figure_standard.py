"""Binding figure policy for PaperOrchestra.

This module is the single source of truth for figure generation policy:
quantitative experiment figures are native matplotlib artifacts backed by
benchmark data, while motivation/overview/mechanism diagrams are mandatory
post-writing Gemini/PaperBanana assets.
"""

from __future__ import annotations

from typing import Any


FIGURE_STANDARD_VERSION = "paperorchestra_figure_standard_v15_2026_06_11"

EXPERIMENT_FIGURE_RULES = {
    "renderer": "python_matplotlib_only",
    "evidence_first": True,
    "forbidden_renderers": ["gemini", "paperbanana", "svg_hand_drawn", "llm_image_generation"],
    "font_family": "Times New Roman or serif fallback",
    "dpi": 300,
    "background": "white",
    "fixed_method_palette": True,
    "no_placeholder_experiment_figures": True,
    "no_mixed_unrelated_chart_types": True,
    "no_single_panel_experiment_default": True,
    "line_plots_require_fill_between": True,
    "line_band_alpha": [0.18, 0.25],
    "chart_diversity_contract": {
        "enabled": True,
        "rule": "A paper's experiment figure pack must diversify by visual family, not merely by 2D versus 3D variants.",
        "minimum_distinct_visual_families_when_multiple_experiment_figures": 3,
        "visual_family_examples": {
            "bar_family": ["2D bar", "grouped bar", "stacked bar", "3D bar"],
            "line_family": ["line plot", "rank line", "trend line", "calibration curve"],
            "matrix_family": ["heatmap", "confusion matrix", "method-by-condition matrix"],
            "distribution_family": ["scatter", "t-SNE", "UMAP", "violin", "box"],
            "radar_family": ["radar", "spider chart"],
        },
        "examples": [
            "main results grouped bars + heatmap",
            "main results grouped bars + line plot with uncertainty bands",
            "main results grouped bars + radar analysis",
            "heatmap + rank line panels for backend-aware studies",
        ],
        "forbidden": [
            "2D bar main results plus 3D bar ablation or sensitivity as the only additional experiment figure",
            "three experiment figures that are all in the bar family",
            "all experiment figures rendered as single-panel plots",
            "reusing the same panel structure for main results, ablation, and sensitivity",
        ],
    },
    "style_reference_dir": "{project_root}/实验图例子",
    "literature_style_reference_required": True,
    "literature_style_reference_agent": "agents.paperorchestra.experiment_plot_reference",
    "minimum_experiment_figures": 3,
    "style_reference_contract": [
        "Before rendering experiment figures, run real literature search for the paper's field and record searched style references in experiment_plot_reference.json.",
        "The experiment plot pack must contain at least three artifact-backed figures and at least three distinct chart families; motivation/overview diagrams do not count.",
        "Choose field-specific figure types from searched related papers when the required artifacts exist, e.g. person Re-ID commonly uses t-SNE/UMAP embeddings, CMC/mAP curves, and retrieval/ranking examples, while LLM reasoning papers often need accuracy-cost/frontier, ablation, and calibration or routing-profile plots.",
        "Do not fabricate field-specific plots: t-SNE, CMC, ranking examples, confusion matrices, or qualitative panels require real embedding, ranking, prediction, or per-class artifacts.",
        "Use compact conference-style multi-panel layouts when the evidence naturally has multiple metrics, conditions, datasets, or hyperparameters.",
        "Experiment figures should be multi-panel by default. Single-panel experiment figures are not allowed for ordinary result, ablation, sensitivity, or robustness plots.",
        "For single-column paper layouts, prefer 1x4 experiment figures when there are four datasets, metrics, conditions, difficulty buckets, or ablation facets.",
        "For double-column papers, prefer wide figure* 1x4 experiment figures for dense comparisons; double-column single-width figures should still prefer 1x2 or 2x1 over single panels.",
        "Use grouped bar layouts whenever comparisons span datasets, scenarios, metrics, or ablation variants: x-axis groups are scenarios/datasets, and bars within each group are methods/variants.",
        "Use a clean paper-plot style: white background, light gray grid lines, low-saturation pastel fills such as soft orange, soft red, soft purple, and soft blue, black or dark-gray bar edges, hatch patterns for every grouped-bar method, numeric value labels above each bar when space permits, shared legends, and concise panel titles.",
        "Grouped-bar hatches must remain distinguishable in grayscale print, using patterns such as diagonal '/', horizontal '--', cross 'xx', and backslash '\\'.",
        "Place grouped-bar legends below the plot in one horizontal row when possible; use matching color+hatch legend swatches and a subtle light-gray legend frame.",
        "Grouped bar figures should resemble a carefully formatted conference plot rather than a decorative infographic: no tinted panel backgrounds, no pictorial icons, no large empty margins.",
        "Do not reserve a large bottom gutter for legends. For wide 1x4 figures, use a compact shared legend directly below the axes so the caption sits close to the plotted area.",
        "Save experiment figures with tight bounding boxes and minimal bottom whitespace; LaTeX captions should sit directly under the figure, not after a blank legend band.",
        "Do not use quality-cost scatter plots as the sole main result; they are allowed as a separate experiment figure when the paper studies routing, inference-time compute, efficiency, latency, token cost, or deployment tradeoffs and the plotted coordinates come from verified artifacts.",
        "When a table is more informative than a plot, keep the table numeric and compact; move interpretation to prose.",
    ],
    "latex_spacing": {
        "caption_font": "normalsize",
        "caption_label_font": "bf",
        "abovecaptionskip": "4pt",
        "belowcaptionskip": "2pt",
        "post_graphics_vspace_for_plots": "-0.2em",
        "rule": "Experiment captions must be readable in PDF review size and should sit directly below the plotted content.",
    },
    "single_column_latex": {"environment": "figure", "width": "\\linewidth"},
    "double_column_latex": {"environment": "figure*", "width": "\\textwidth"},
}

CONCEPT_FIGURE_RULES = {
    "enabled": True,
    "required": True,
    "renderer": "gemini_native_paperbanana_postwriting_only",
    "scope": ["motivation", "overview", "mechanism"],
    "native_fallback_allowed": False,
    "block_if_missing": True,
    "style": {
        "overall": "structured high-information PPT-built academic schematic with flat hand-drawn visual language; not a rendered illustration, not a poster, not a mascot scene, and not a text-card dashboard",
        "style_reference_dir": "{project_root}/动机图和框架图例子",
        "style_reference_contract": [
            "Study the local reference examples before generating motivation/overview figures.",
            "Borrow their useful traits: tidy aligned regions, dashed/rounded containers, local zoom-ins, concrete task examples, mini matrices/tables, formula callouts, arrows that connect evidence rather than stages, rounded hand-written or marker-like labels, and flat PowerPoint-like geometry.",
            "Do not copy content, characters, layouts, or text verbatim; adapt only the visual language to the current paper.",
            "Prefer structured academic schematics with small flat icon accents, not full-scene posters. Icons must be selected from the paper's domain entities; characters, robots, avatars, cells, documents, graphs, databases, patients, queries, or other icons may appear only when they are semantically relevant and must not carry the main content alone.",
            "Answer slips, confidence tags, token chips, group piles, margin score, retain/discard marks, and selected output should be arranged as a paper schematic, with local illustrations embedded inside modules.",
            "The figure should feel like a top-conference method/motivation schematic assembled in PowerPoint with dense technical semantics, not like an AI-made rendered illustration, poster, mascot scene, icon board, dashboard, isolated pictogram collage, or generic infographic.",
            "Use flat 2D shapes, thick clean outlines, subtle hatch/dashed boxes inside containers, crisp alignment, and little-to-no shading. Use a pure white canvas by default; only tiny local modules may use very pale tints. Avoid warm full-canvas backgrounds, yellow/cream wash, vignette, gradient, grid paper, graph paper, notebook lines, painterly lighting, volumetric objects, glossy highlights, cast shadows, scenic backgrounds, and rendered 3D depth.",
            "Do not put LaTeX-style captions, Figure/Fig. numbering, standalone titles, explanatory paragraphs, line numbers, or panel numbers such as 1./2./3. inside the generated image; the LaTeX caption carries all figure numbering and long explanation.",
            "Do not force a three-column motivation comparison. Prefer a compact tension map, a central mechanism schematic, or one worked-example composition with at most two small callouts.",
            "Choose icons adaptively from the paper domain. For multi-agent papers, agent/avatar/robot icons should be visible and semantically central as trace sources; for other domains, use the corresponding domain objects instead. Do not replace domain entities with unrelated office metaphors such as envelopes, inbox trays, or generic storage bins.",
        ],
        "strict_no_flowchart": {
            "enabled": True,
            "forbidden": [
                "sequential pipeline",
                "left-to-right process flow",
                "step-by-step boxes connected by arrows",
                "input-to-output module chain",
                "diamond gate branching workflow",
                "empty flowchart",
                "weak-information flowchart",
                "large whitespace with a few boxes and arrows",
                "swimlane workflow",
                "business process chart",
                "algorithm flowchart",
                "module chain",
                "boxes labeled as stages connected by arrows",
                "isolated icon collage",
                "decision board",
                "dashboard full of gauges and cards",
                "visible step numbers, numbered panels, or numbered circle badges",
                "card pile composition",
                "full-scene cartoon poster",
                "rendered cartoon illustration",
                "AI illustration style",
                "painterly or glossy objects",
                "heavy shadows or scenic lighting",
                "graph paper or notebook grid background",
                "warm yellow or cream full-canvas background",
                "background gradient, vignette, or paper wash",
                "unrelated office metaphors such as envelopes and trays as the main result",
                "mascot-dominated illustration",
                "large scenic laboratory illustration",
                "single-room story illustration",
                "three-column motivation comparison",
                "internal Figure/Fig. labels or caption text",
                "long explanatory paragraphs inside the image",
                "cute character poster with sparse technical content",
                "large furniture or environmental scenery",
            ],
            "allowed_alternatives": [
                "structured academic schematic with cartoon accents",
                "method block diagram with local illustrated callouts",
                "motivation comparison with concrete failure callouts",
                "icon-rich academic cartoon metaphor",
                "task-specific compact panel with concrete visual objects",
                "spatial relationship map",
                "central mechanism with surrounding semantic regions",
                "before/after conceptual juxtaposition without process arrows",
                "tension map with opposing regions",
                "non-sequential mechanism landscape",
                "mechanism metaphor grounded in actual method terms",
                "worked-example schematic with small tables, formulas, score tags, and local zoom-ins",
            ],
        },
        "no_empty_flowchart": {
            "enabled": True,
            "reject_if": [
                "large whitespace without effective information",
                "only input -> module -> output structure",
                "module interiors contain only titles",
                "arrows carry nearly all logic",
                "overview is ordinary input -> gate -> output",
                "motivation is current-practice -> limitation -> motivation boxes",
                "no strong visual center",
                "generic labels such as Module, Decision, Process, Output dominate",
                "looks like a temporary PPT sketch",
                "all content is stacked text cards with no icons or concrete visual metaphors",
                "isolated cards, gauges, arrows, and icons floating on white space",
                "large title plus sparse icon groups",
                "overview is a dashboard or decision board rather than an illustrated method scene",
                "full canvas is a poster scene with characters and furniture",
                "figure looks like a generated illustration instead of PPT-built schematic",
                "objects have glossy shading, cast shadows, or 3D depth",
                "background is graph paper, grid paper, lined notebook paper, or worksheet paper",
                "background is globally tinted yellow/cream or uses a vignette/gradient wash",
                "the paper's core domain entities are missing or replaced by unrelated generic icons",
                "characters or mascots are larger than the method objects",
                "figure lacks structural modules, local zoom-ins, or explicit mechanism relations",
                "cute illustration style replaces the technical diagram",
                "font looks like serif manuscript typography instead of rounded schematic lettering",
                "no concrete numeric cues, equations, scores, or trace examples appear",
            ],
            "positive_standard": [
                "concise but not empty",
                "low saturation but clear focal area",
                "few arrows but explicit mechanism",
                "short text but paper-specific semantics",
                "whitespace but not loose spacing",
                "mechanism structure rather than generic pipeline",
                "icon-like objects encode the method's actual domain entities and decisions",
                "small flat domain-specific icons support, but do not replace, the mechanism schematic",
                "method objects such as answer slips, group piles, margin evidence, score tags, budget tokens, retain/discard marks, and selected output carry the semantics",
                "local examples, formulas, and score/cost tags make the figure recognizable without relying on the caption",
                "small flat icons annotate operations but never replace the mechanism objects",
            ],
        },
        "minimum_information_density": {
            "effective_area_coverage": "main content should occupy at least 70 percent of the useful canvas",
            "semantic_elements_per_main_region": "each main region should contain at least 3-5 concrete semantic elements: example objects, numeric tags, state marks, mini tables/matrices, local zoom-ins, or formula snippets",
            "worked_example_required": "concept figures should contain a worked miniature example whenever the method has discrete objects such as answers, agents, costs, modalities, samples, or layers",
            "formula_or_rule_callout_required": "overview figures must include at least one compact rule/formula/score callout when the method has a named decision rule",
            "overview_requires_at_least_three": [
                "input structure such as agents, answers, confidence, cost, samples",
                "intermediate representation such as groups, candidate sets, scores, routing states",
                "core mechanism such as margin gate, confidence-weighted scoring, variance decomposition",
                "branch basis such as stable/unstable, majority/dissent, known/unknown",
                "information selection such as retained, discarded, compressed, reweighted",
                "output semantics showing how the final output is determined",
            ],
            "motivation_requires": [
                "existing strategy",
                "failure mechanism",
                "failure consequence",
                "paper-specific motivation",
            ],
        },
        "palette": {
            "max_semantic_colors": 4,
            "background": "pure white canvas preferred; at most very pale local region tints inside modules; no full-canvas cream/yellow wash, no gradient, no vignette, no grid, graph-paper, notebook, ruled, or worksheet lines",
            "primary_flow": "blue-gray or pale blue",
            "contrast_or_candidate": "pale orange or light brown",
            "problem_or_risk": "pale red only for limited emphasis",
            "improvement_or_output": "pale green",
            "supporting_elements": "neutral gray",
            "avoid": "high-saturation color blocks, rainbow palettes, heavy red-blue-orange-green combinations",
        },
        "arrows": {
            "principle": "arrows cannot carry all logic; use only necessary connectors and put mechanism inside regions",
            "main_flow": "no empty box-arrow flowcharts; arrows may indicate method relations only when modules contain concrete semantics",
            "secondary_flow": "omit unless absolutely necessary",
            "avoid": "pipeline arrows, crossing arrows, many curved arrows, mixed arrow styles, arrows touching labels or module borders",
        },
        "layout": {
            "principle": "clear regions, strong alignment, enough whitespace, readable when scaled to paper width; layouts should be tidy and aligned but not use a visible grid background",
            "motivation": "high-density structured comparison grounded in the paper's own domain entities. For multi-agent papers this means agent/avatar trace sources, majority voting, lost useful dissent, keep-all token waste, and conditional retention with concrete answer bubbles, counters, score/cost tags, and small callout icons; for other domains, substitute the correct domain objects. Use tidy comparison regions or a single-row triptych, not three empty boxes, card stacks, icon-only groups, or a poster scene",
            "overview": "mechanism-rich schematic grounded in the paper's own domain entities. For multi-agent papers this means agent/avatar traces, method-specific grouping, margin evidence, dissent score, budget constraint, retained/discarded traces, and selected output; for other domains, substitute the correct domain objects. Include at least one worked example table or local zoom-in; use a central mechanism map or structured evidence map, not a numbered four-column flow",
            "avoid": "deep nested containers, overcrowded modules, repeated titles, weak output area, empty step-by-step chains, visible step numbers, numbered circle badges, large blank canvas around tiny modules, isolated icon boards, decision dashboards, full-scene cartoon posters, mascot-dominated illustrations, cinematic rooms, decorative backgrounds, graph-paper backgrounds, furniture, envelopes/trays as central metaphors, or large environmental scenes",
        },
        "text": {
            "principle": "short phrases, not paragraphs",
            "module_titles": "1-4 words",
            "max_lines_per_small_box": 2,
            "font": "rounded hand-written or marker-like academic sans lettering, similar to a neat PPT annotation font; do not use Times New Roman, manuscript serif, or formal book typography inside concept figures",
            "text_style": "labels should look like tidy schematic annotations, with high contrast and consistent size; use handwritten-style emphasis sparingly for warnings or key concepts; avoid formal serif fonts",
            "avoid": "long sentences, dense acronyms, formula derivations, caption-like text inside image, serif manuscript font, oversized title banners",
        },
        "icons_shapes": {
            "allowed_shapes": ["rounded rectangles", "small cards", "circles or ellipses", "diamonds for gates", "light stage containers"],
            "encouraged_icons": [
                "domain-specific entity icons selected from the paper context",
                "agent faces or tiny avatars only for agentic or multi-agent papers",
                "speech bubbles for candidate answers",
                "coins or chips for token cost",
                "thermometer/gauge/meter for budget or confidence",
                "scale icon for scoring or trade-off",
                "magnifier for evidence inspection",
                "check/cross badges for retained versus discarded candidates",
                "subtle warning mark for failure cases",
            ],
            "icon_rule": "Use consistent, cute-but-academic icons only when they carry semantics; icons must come from the paper's domain entities and explain the mechanism rather than decorate empty boxes.",
            "scene_rule": "Choose flat icons adaptively: agent/avatar/robot icons only when the paper is about agents, graph/database/query/patient/cell/sample/document icons when those are the actual entities. The main composition remains a structured technical schematic with concrete evidence objects.",
            "avoid": "3D icons, strong gradients, heavy shadows, glossy rendering, decorative icon piles, generic stock icons unrelated to the method, isolated dashboard widgets, floating card collages, full-scene laboratory posters, mascot-dominated scenes, characters larger than the mechanism, furniture-heavy scenes, envelopes/trays as the main semantic objects",
        },
    },
    "role_contract": {
        "motivation": {
            "purpose": "show why existing strategy is insufficient and why this paper is needed",
            "structure": "mechanism-specific structured comparison between majority collapse, lost dissent, keep-all waste, and conditional retention, with visible domain-specific trace sources and compact flat callouts only where helpful",
            "must_show": ["majority voting", "lost useful dissent", "keep-all token waste", "conditional retention need"],
            "avoid": "empty current-practice/limitation/motivation boxes, full method pipeline, left-to-right stages, arrows between stages, too many method modules, more than 2-3 core problems, text-only card stacks, isolated icon collages, large all-caps headings, full-scene posters, mascot-dominated scenes",
        },
        "overview": {
            "purpose": "make the method intuitive before Method details by showing entities, decisions, and evidence as a structured schematic with local visual callouts",
            "structure": "mechanism-rich method schematic with domain-specific trace sources, candidate answers, confidence/cost cues, grouped support, margin evidence, dissent scoring, retained/discarded traces, selected output, and flat icon accents",
            "main_region_count": [3, 6],
            "must_show_at_least_three": ["input structure", "intermediate groups or candidate sets", "core margin/scoring mechanism", "information retained/discarded/reweighted", "output semantics"],
            "avoid": "empty input-gate-output diagrams, generic module/process labels, all implementation details, loss/training minutiae, repeated motivation contrast, mandatory left-to-right flow, text-only card stacks, decision boards, dashboard layouts, isolated icon collages, full-scene posters, mascot-dominated scenes",
        },
    },
    "preflight_checklist": [
        "Is this motivation or overview?",
        "Can the core message be stated in one sentence?",
        "Does the figure show this paper's specific problem or mechanism?",
        "Does it use concrete icons or visual metaphors rather than only text cards?",
        "Does it stay a structured academic schematic rather than becoming a poster scene?",
        "Are cartoon characters/icons small accents rather than the main subject?",
        "If the title is removed, can a reader still recognize this paper's method?",
        "Does it have a core mechanism region rather than only arrows?",
        "Does every main region contain concrete semantic elements?",
        "Is effective content area at least about 70 percent of the useful canvas?",
        "Is overview more than input -> gate -> output?",
        "Does motivation show a failure mechanism rather than abstract labels?",
        "Does the output show how it is produced by the mechanism?",
        "Does it look like a paper figure rather than a PPT sketch?",
        "Does it avoid being a plain input-output diagram?",
        "Are semantic colors limited to 3-4?",
        "Are arrow count and styles controlled?",
        "Are modules aligned with enough whitespace?",
        "Is text short and readable at paper width?",
        "Does the output region clearly close the diagram?",
    ],
}

PANEL_LAYOUT_RULES = {
    "single_column_paper": ["1x4", "1x3", "1x2", "2x1"],
    "double_column_single_column_figure": ["1x2", "2x1"],
    "double_column_double_column_figure": ["1x3", "1x4"],
    "principle": "Experiment figures should not be isolated single panels. Prefer 1x4 for single-column paper layouts and wide figure* analyses; use 1x2/2x1 only for narrow single-column slots.",
}

BLOCKLISTED_INTERNAL_FIGURE_TOKENS = (
    "trajectory",
    "keep",
    "discard",
    "search dynamics",
    "seed variance",
)


def default_plot_plan(metric_name: str) -> list[dict[str, Any]]:
    return [
        {
            "figure_id": "fig_main_results",
            "plot_type": "plot",
            "role": "experiment_figure_pack",
            "chart_type": "main_results_bar",
            "title": "Main results",
            "objective": f"Report verified {metric_name}, token cost, latency, and routing rate across methods with seed uncertainty.",
            "data_source": "benchmark_summary.json",
            "aspect_ratio": "4:1",
            "layout": "1x4",
            "placement": "double_column",
            "standard_version": FIGURE_STANDARD_VERSION,
        }
    ]


def backend_plot_pack(metric_name: str) -> list[dict[str, Any]]:
    metric_title = str(metric_name or "accuracy")
    return [
        {
            "figure_id": "fig_backend_grouped_bars",
            "plot_type": "plot",
            "role": "experiment_figure_pack",
            "chart_type": "backend_grouped_bars",
            "title": "Backend accuracy bars",
            "objective": f"Compare {metric_title} across inference backends and methods with seed uncertainty.",
            "data_source": "benchmark_summary.json",
            "aspect_ratio": "4:3",
            "layout": "single",
            "standard_version": FIGURE_STANDARD_VERSION,
        },
        {
            "figure_id": "fig_backend_heatmap_single",
            "plot_type": "plot",
            "role": "experiment_figure_pack",
            "chart_type": "backend_heatmap_single",
            "title": "Backend accuracy heatmap",
            "objective": f"Show the method-by-backend {metric_title} matrix with seed standard deviation.",
            "data_source": "benchmark_summary.json",
            "aspect_ratio": "1:1",
            "layout": "single",
            "standard_version": FIGURE_STANDARD_VERSION,
        },
        {
            "figure_id": "fig_backend_rank_lines_1x4",
            "plot_type": "plot",
            "role": "experiment_figure_pack",
            "chart_type": "backend_rank_lines_1x4",
            "title": "Backend rank stability",
            "objective": "Track method ranks across inference backends on four datasets.",
            "data_source": "benchmark_summary.json",
            "aspect_ratio": "4:1",
            "layout": "1x4",
            "placement": "double_column",
            "standard_version": FIGURE_STANDARD_VERSION,
        },
    ]


def is_blocklisted_internal_figure(fig: dict[str, Any]) -> bool:
    text = " ".join(
        str(fig.get(key) or "")
        for key in ("figure_id", "title", "objective", "chart_type", "data_source")
    ).lower()
    return any(token in text for token in BLOCKLISTED_INTERNAL_FIGURE_TOKENS)


def experiment_figure_policy_manifest() -> dict[str, Any]:
    return {
        "standard_version": FIGURE_STANDARD_VERSION,
        "experiment_figure_rules": EXPERIMENT_FIGURE_RULES,
        "concept_figure_rules": CONCEPT_FIGURE_RULES,
        "panel_layout_rules": PANEL_LAYOUT_RULES,
    }
