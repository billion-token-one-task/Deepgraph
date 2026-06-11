from agents.reference_auditor import audit_references, parse_bib_entries


def _bib_entry(i: int) -> str:
    return (
        f"@inproceedings{{ref{i},\n"
        f"  title={{Verified Paper {i}}},\n"
        f"  author={{Author {i} and Coauthor {i}}},\n"
        f"  booktitle={{Proceedings of a Conference}},\n"
        f"  year={{2024}},\n"
        f"  doi={{10.0000/example.{i}}}\n"
        f"}}\n"
    )


def test_reference_auditor_blocks_sparse_and_misplaced_citations():
    bib = "\n".join(_bib_entry(i) for i in range(3))
    tex = r"""
\begin{abstract}
This abstract wrongly cites prior work \cite{ref0}.
\end{abstract}
\section{Introduction}
We propose the method.
\paragraph{Contributions}
We evaluate it \cite{ref1}.
\section{Related Work}
Prior work exists \cite{ref0, ref1, ref2}.
\section{Method}
The method builds on a known idea.
"""
    audit = audit_references(main_tex=tex, bibtex=bib)
    assert audit["status"] == "fail"
    assert audit["bibliography_entry_count"] == 3
    assert any(issue["standard"] == "Reference auditor / bibliography size" for issue in audit["issues"])
    assert any(issue["standard"] == "Reference auditor / citation placement" for issue in audit["issues"])
    assert any(issue["standard"] == "Reference auditor / contribution placement" for issue in audit["issues"])


def test_reference_auditor_accepts_fifty_verified_distributed_citations():
    bib = "\n".join(_bib_entry(i) for i in range(50))
    def cite_chunks(start, end, size=5):
        chunks = []
        for offset in range(start, end, size):
            chunks.append("\\cite{" + ", ".join(f"ref{i}" for i in range(offset, min(end, offset + size))) + "}")
        return " ".join(chunks)

    intro = cite_chunks(0, 8)
    related_a = cite_chunks(8, 23)
    related_b = cite_chunks(23, 43)
    method = cite_chunks(43, 50)
    tex = rf"""
\begin{{abstract}}
This abstract has no citations.
\end{{abstract}}
\section{{Introduction}}
Prior work motivates the problem {intro}.
\section{{Related Work}}
One category studies the first line of work {related_a}. Another category studies the second line {related_b}.
\section{{Method}}
The method follows relevant algorithmic conventions {method}.
"""
    audit = audit_references(main_tex=tex, bibtex=bib)
    assert audit["status"] == "pass"
    assert audit["bibliography_entry_count"] == 50
    assert audit["unique_cited_count"] == 50
    assert parse_bib_entries(bib)["ref0"]["fields"]["title"] == "Verified Paper 0"
