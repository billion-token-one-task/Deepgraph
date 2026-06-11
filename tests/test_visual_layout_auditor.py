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
        },
        {
            "figure_id": "fig_overview_symbolic",
            "kind": "diagram",
            "stage": "postwriting_api_figures",
            "path": "figures/fig_overview_symbolic.png",
            "notes": "paperbanana_ok",
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
    assert feedback["authorable_issue_count"] == 0
    assert feedback["stage_blocker_count"] >= 2
    fixes = " ".join(issue.get("fix", "") for issue in audit["issues"])
    assert "optional concept figures" not in fixes.lower()
    assert "mandatory" in fixes.lower()
