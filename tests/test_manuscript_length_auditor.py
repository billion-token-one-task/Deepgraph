from agents.manuscript_length_auditor import audit_manuscript_length


def _words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


def test_short_iclr_manuscript_is_blocked_by_length_auditor():
    tex = r"""
\begin{abstract}
Short abstract.
\end{abstract}
\section{Introduction}
Short intro.
\section{Related Work}
Short related work.
\section{Method}
Short method.
\section{Experiments}
Short experiments.
\section{Discussion}
Short discussion.
"""
    audit = audit_manuscript_length(
        main_tex=tex,
        page_count=3,
        venue_target={"family": "iclr"},
    )
    assert audit["status"] == "fail"
    assert any(issue["standard"] == "Length auditor / complete-paper page floor" for issue in audit["issues"])
    assert any(issue["standard"] == "Length auditor / main-body word floor" for issue in audit["issues"])


def test_complete_section_word_profile_passes_length_auditor():
    tex = rf"""
\begin{{abstract}}
{_words(180)}
\end{{abstract}}
\section{{Introduction}}
{_words(900)}
\section{{Related Work}}
{_words(900)}
\section{{Method}}
{_words(1400)}
\section{{Experiments}}
{_words(1800)}
\section{{Discussion}}
{_words(500)}
"""
    audit = audit_manuscript_length(
        main_tex=tex,
        page_count=8,
        venue_target={"family": "iclr"},
    )
    assert audit["status"] == "pass"
    assert audit["section_words"]["experiments_results"] >= 1800
