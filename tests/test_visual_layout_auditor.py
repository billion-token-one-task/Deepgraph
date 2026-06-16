from agents.visual_layout_auditor import audit_visual_layout
from agents.paper_orchestra_pipeline import pick_main_tex, _build_manuscript_revision_feedback


def _required_assets():
    return [
        {
            "figure_id": "fig_motivation_symbolic",
            "kind": "diagram",
            "stage": "postwriting_api_figures",
            "path": "figures/fig_motivation_symbolic.png",
            "notes": "paperbanana_ok",
            "aspect_ratio": "4:3",
        },
        {
            "figure_id": "fig_overview_symbolic",
            "kind": "diagram",
            "stage": "postwriting_api_figures",
            "path": "figures/fig_overview_symbolic.png",
            "notes": "paperbanana_ok",
            "aspect_ratio": "4:3",
        },
    ]


def test_visual_auditor_requires_postwriting_motivation_and_overview_figures():
    tex = r"""
\title{Paper}
\begin{document}
\maketitle
\begin{abstract}
Abstract text.
\end{abstract}
\section{Introduction}
Intro text with enough substance for layout.
"""
    audit = audit_visual_layout(main_tex=tex, figure_assets=[], page_count=1)
    assert audit["status"] == "fail"
    assert any(issue["standard"] == "Visual layout auditor / required concept figures" for issue in audit["issues"])


def test_visual_auditor_blocks_topmatter_figure_and_duplicate_caption():
    tex = r"""
\title{Paper}
\begin{document}
\begin{figure}[t]
\centering
\includegraphics{figures/fig_motivation_symbolic.png}
\caption{Motivation.}
\label{fig:fig_motivation_symbolic}
\end{figure}
Figure 1: Duplicate caption outside the figure environment.
\maketitle
\begin{abstract}
Abstract text.
\end{abstract}
\section{Introduction}
Intro text.
"""
    audit = audit_visual_layout(main_tex=tex, figure_assets=_required_assets(), page_count=1)
    standards = {issue["standard"] for issue in audit["issues"]}
    assert audit["status"] == "fail"
    assert "Visual layout auditor / top matter" in standards
    assert "Visual layout auditor / duplicate captions" in standards


def test_pick_main_tex_moves_topmatter_figure_after_intro_and_strips_raw_caption():
    full = r"""\documentclass{article}
\title{Paper}
\begin{document}
\begin{figure}[t]
\centering
\includegraphics{figures/fig_motivation_symbolic.png}
\caption{Motivation.}
\label{fig:fig_motivation_symbolic}
\end{figure}
Figure 1: Duplicate caption outside the figure environment.
\maketitle
\begin{abstract}
Abstract text.
\end{abstract}
\section{Introduction}
This paragraph explains the problem before any concept figure appears.

More introduction text.
\section{Method}
Method text.
\end{document}
"""
    tex = pick_main_tex(
        {"refinement_full_text": full, "plotting": {"postwriting_api_figure_stage": {"assets": _required_assets()}}},
        {"title": "Paper", "venue_target": {"key": "iclr2026"}},
        "conference",
    )
    assert tex.find(r"\maketitle") < tex.find(r"\section{Introduction}") < tex.find(r"\begin{figure}")
    assert "Figure 1: Duplicate caption" not in tex



def test_manuscript_revision_feedback_sends_layout_issues_back_to_writer():
    quality_report = {
        "writing_guideline_audit": {"decision": "manuscript_blocked"},
        "issues": [
            {
                "severity": "high",
                "standard": "Visual layout auditor / top matter",
                "issue": "A figure appears before maketitle/title/authors.",
                "fix": "Move the figure after Introduction prose.",
            },
            {
                "severity": "high",
                "standard": "Visual layout auditor / duplicate captions",
                "issue": "Standalone Figure caption text appears outside the figure environment.",
                "fix": "Keep only the LaTeX caption.",
            },
        ],
    }
    feedback = _build_manuscript_revision_feedback(quality_report, attempt=1)
    assert feedback["authorable_issue_count"] == 2
    assert feedback["stage_blocker_count"] == 0


def test_manuscript_revision_feedback_keeps_missing_paperbanana_figures_as_stage_blockers():
    tex = r"""
\title{Paper}
\begin{document}
\maketitle
\begin{abstract}Abstract text.\end{abstract}
\section{Introduction}
Intro text.
\end{document}
"""
    audit = audit_visual_layout(main_tex=tex, figure_assets=[], page_count=1)
    quality_report = {
        "writing_guideline_audit": {"decision": "manuscript_blocked"},
        "issues": audit["issues"],
    }
    feedback = _build_manuscript_revision_feedback(quality_report, attempt=1)
    assert feedback["authorable_issue_count"] >= 1
    assert feedback["stage_blocker_count"] >= 2
    fixes = " ".join(issue.get("fix", "") for issue in audit["issues"])
    assert "optional concept figures" not in fixes.lower()
    assert "mandatory" in fixes.lower()



def _valid_experiment_assets():
    base = []
    for idx, (fid, family, chart_type) in enumerate([
        ("fig_main_results", "bar_family", "main_results_bar"),
        ("fig_ablation_results", "bar_family", "ablation_bar"),
        ("fig_hyperparameter_sweep", "line_family", "hyperparameter_sweep"),
    ]):
        base.append(
            {
                "figure_id": fid,
                "kind": "plot",
                "path": f"figures/{fid}.pdf",
                "pdf_path": f"figures/{fid}.pdf",
                "chart_family": family,
                "chart_type": chart_type,
                "layout": "1x4" if idx == 0 else "1x3",
                "placement": "double_column",
                "style_reference_keys": ["S2_related"],
                "style_reference_titles": ["Related field paper"],
                "style_reference_sources": ["local_user_examples", "related_literature_search"],
                "local_style_reference_dir": "实验图例子",
            }
        )
    return base


def test_visual_auditor_allows_missing_hyperparameter_when_main_and_ablation_present():
    assets = _required_assets() + _valid_experiment_assets()[:2]
    tex = r"""
\begin{document}
\maketitle
\begin{abstract}Abstract text.\end{abstract}
\section{Introduction}
This is enough introduction prose to avoid top-matter concerns before figures are considered by the layout auditor.
"""
    audit = audit_visual_layout(main_tex=tex, figure_assets=assets, page_count=8)
    assert audit["status"] != "fail"
    assert not any(
        issue["severity"] == "high" and "hyperparameter" in (issue.get("evidence", "") + issue.get("fix", "")).lower()
        for issue in audit["issues"]
    )


def test_visual_auditor_blocks_single_panel_experiment_figures():
    assets = _required_assets() + _valid_experiment_assets()
    assets[-1]["layout"] = "single"
    assets[-1]["aspect_ratio"] = "4:3"
    tex = r"""
\begin{document}
\maketitle
\begin{abstract}Abstract text.\end{abstract}
\section{Introduction}
This is enough introduction prose to avoid top-matter concerns before figures are considered by the layout auditor.
"""
    audit = audit_visual_layout(main_tex=tex, figure_assets=assets, page_count=8)
    assert audit["status"] == "fail"
    assert any(issue["standard"] == "Visual layout auditor / experiment panel layout" for issue in audit["issues"])


def test_visual_auditor_blocks_single_column_experiment_placement():
    assets = _required_assets() + _valid_experiment_assets()
    tex = r"""
\begin{document}
\maketitle
\begin{abstract}Abstract text.\end{abstract}
\section{Introduction}
This is enough introduction prose to avoid top-matter concerns before figures are considered by the layout auditor.
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig_ablation_results.pdf}
\caption{Ablation results.}
\end{figure}
"""
    audit = audit_visual_layout(main_tex=tex, figure_assets=assets, page_count=8)
    assert audit["status"] == "fail"
    assert any(issue["standard"] == "Visual layout auditor / experiment figure placement" for issue in audit["issues"])


def test_visual_auditor_blocks_narrow_centered_tables():
    assets = _required_assets() + _valid_experiment_assets()
    tex = r"""
\begin{document}
\maketitle
\begin{abstract}Abstract text.\end{abstract}
\section{Introduction}
Intro text.
\begin{table}[t]
\centering
\caption{Narrow table.}
\begin{tabular}{lcc}
\toprule
Method & A & B \\
\midrule
Ours & 1 & 2 \\
\bottomrule
\end{tabular}
\end{table}
"""
    audit = audit_visual_layout(main_tex=tex, figure_assets=assets, page_count=8)
    assert audit["status"] == "fail"
    assert any(issue["standard"] == "Visual layout auditor / table width" for issue in audit["issues"])


def test_visual_auditor_blocks_legacy_quality_cost_scatter_pack():
    assets = _required_assets() + _valid_experiment_assets()
    assets[3] = {
        **assets[3],
        "figure_id": "fig_quality_cost_tradeoff",
        "chart_type": "quality_cost_tradeoff",
        "chart_family": "distribution_family",
        "path": "figures/fig_quality_cost_tradeoff.pdf",
    }
    tex = r"""
\begin{document}
\maketitle
\begin{abstract}Abstract text.\end{abstract}
\section{Introduction}
This is enough introduction prose to avoid top-matter concerns before figures are considered by the layout auditor.
"""
    audit = audit_visual_layout(main_tex=tex, figure_assets=assets, page_count=8)
    assert audit["status"] == "fail"
    assert any(issue["standard"] == "Visual layout auditor / experiment scatter policy" for issue in audit["issues"])
    assert any(issue["standard"] == "Visual layout auditor / experiment figure required roles" for issue in audit["issues"])


def test_visual_auditor_blocks_patterned_experiment_bars():
    assets = _required_assets() + _valid_experiment_assets()
    assets[-1]["uses_hatch"] = True
    tex = r"""
\begin{document}
\maketitle
\begin{abstract}Abstract text.\end{abstract}
\section{Introduction}
This is enough introduction prose to avoid top-matter concerns before figures are considered by the layout auditor.
"""
    audit = audit_visual_layout(main_tex=tex, figure_assets=assets, page_count=8)
    assert audit["status"] == "fail"
    assert any(issue["standard"] == "Visual layout auditor / experiment bar texture" for issue in audit["issues"])


def test_visual_auditor_blocks_wide_concept_aspect_ratio():
    assets = _required_assets() + _valid_experiment_assets()
    assets[0]["aspect_ratio"] = "16:9"
    tex = r"""
\begin{document}
\maketitle
\begin{abstract}Abstract text.\end{abstract}
\section{Introduction}
This is enough introduction prose to avoid top-matter concerns before figures are considered by the layout auditor.
"""
    audit = audit_visual_layout(main_tex=tex, figure_assets=assets, page_count=8)
    assert audit["status"] == "fail"
    assert any(issue["standard"] == "Visual layout auditor / concept aspect ratio" for issue in audit["issues"])


def test_visual_auditor_blocks_cramped_tabularx_method_column():
    assets = _required_assets() + _valid_experiment_assets()
    tex = r"""
\begin{document}
\maketitle
\begin{abstract}Abstract text.\end{abstract}
\section{Introduction}
Intro text.
\begin{table*}[t]
\centering
\caption{Cramped table.}
\begin{tabularx}{\textwidth}{l*{5}{>{\centering\arraybackslash}X}}
\toprule
Method & A & B & C & D & E \\
\midrule
Very Long Method Name & 1 & 2 & 3 & 4 & 5 \\
\bottomrule
\end{tabularx}
\end{table*}
"""
    audit = audit_visual_layout(main_tex=tex, figure_assets=assets, page_count=8)
    assert audit["status"] == "fail"
    assert any(issue["standard"] == "Visual layout auditor / table column allocation" for issue in audit["issues"])



def test_visual_auditor_blocks_table_overflow_beyond_text_block():
    assets = _required_assets() + _valid_experiment_assets()
    tex = r"""
\begin{document}
\maketitle
\begin{abstract}Abstract text.\end{abstract}
\section{Introduction}
Intro text.
\begin{table*}[t]
\centering
\caption{Overflowing table.}
\resizebox{1.15\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Method & A & B & C & D & E \\
\midrule
Ours & 1 & 2 & 3 & 4 & 5 \\
\bottomrule
\end{tabular}}
\end{table*}
"""
    audit = audit_visual_layout(main_tex=tex, figure_assets=assets, page_count=8)
    assert audit["status"] == "fail"
    assert any(issue["standard"] == "Visual layout auditor / table overflow" for issue in audit["issues"])


def test_visual_auditor_blocks_overcompressed_table_spacing():
    assets = _required_assets() + _valid_experiment_assets()
    tex = r"""
\begin{document}
\maketitle
\begin{abstract}Abstract text.\end{abstract}
\section{Introduction}
Intro text.
\begin{table*}[t]
\centering
\small
\renewcommand{\arraystretch}{0.85}
\setlength{\tabcolsep}{1.5pt}
\caption{Compressed table.}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X*{5}{>{\centering\arraybackslash}p{0.095\textwidth}}}
\toprule
\rowcolor{gray!14}
Method & A & B & C & D & E \\
\midrule
M1 & 1 & 2 & 3 & 4 & 5 \\
M2 & 1 & 2 & 3 & 4 & 5 \\
M3 & 1 & 2 & 3 & 4 & 5 \\
M4 & 1 & 2 & 3 & 4 & 5 \\
\bottomrule
\end{tabularx}
\end{table*}
"""
    audit = audit_visual_layout(main_tex=tex, figure_assets=assets, page_count=8)
    assert audit["status"] == "fail"
    assert any(issue["standard"] == "Visual layout auditor / table row spacing" for issue in audit["issues"])
    assert any(issue["standard"] == "Visual layout auditor / table column spacing" for issue in audit["issues"])


def test_visual_auditor_requests_table_style_polish_for_plain_dense_table():
    assets = _required_assets() + _valid_experiment_assets()
    tex = r"""
\begin{document}
\maketitle
\begin{abstract}Abstract text.\end{abstract}
\section{Introduction}
Intro text.
\begin{table*}[t]
\centering
\small
\renewcommand{\arraystretch}{1.08}
\setlength{\tabcolsep}{4pt}
\caption{Plain dense table.}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X*{4}{>{\centering\arraybackslash}p{0.11\textwidth}}}
\toprule
Method & A & B & C & D \\
\midrule
Baseline & 1 & 2 & 3 & 4 \\
Variant & 1 & 2 & 3 & 4 \\
Other & 1 & 2 & 3 & 4 \\
Ours & 2 & 3 & 4 & 5 \\
\bottomrule
\end{tabularx}
\end{table*}
"""
    audit = audit_visual_layout(main_tex=tex, figure_assets=assets, page_count=8)
    assert audit["status"] == "needs_revision"
    assert any(issue["standard"] == "Visual layout auditor / table style polish" for issue in audit["issues"])
    assert any(issue["standard"] == "Visual layout auditor / table result emphasis" for issue in audit["issues"])
