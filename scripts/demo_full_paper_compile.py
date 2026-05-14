#!/usr/bin/env python3
"""End-to-end: realistic full paper → 6 venue adapters → real PDF compile.

What this exercises:
  • Venue router rule-based selection (CV/NLP/ML/Theory).
  • Each adapter's ``normalize_source`` produces a venue-specific preamble.
  • Each adapter's ``copy_files`` stages the right ``.sty`` / ``.bst``.
  • Tectonic compiles every venue to a real PDF.
  • Format linter runs the 7 contract checks on the same source.

Outputs land under ``/tmp/full_paper_demo/<venue>/paper.pdf`` so reviewers
can visually inspect column-layout, header, bibstyle differences.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("DEEPGRAPH_DATABASE_URL", "")
os.environ["DEEPGRAPH_DB_PATH"] = "/tmp/full_paper_demo.db"

from agents import venue_router  # noqa: E402
from agents.format_linter import lint_manuscript  # noqa: E402
from agents.manuscript_templates import get_adapter, list_adapters  # noqa: E402


PAPER_BODY = r"""\documentclass{article}
\title{Linear Attention for Long-Context Language Modelling}
\author{Anonymous}
\begin{document}
\maketitle

\begin{abstract}
We introduce a linear-time attention mechanism that retains the modelling
capacity of full softmax attention while reducing complexity from
$O(n^2)$ to $O(n)$ for sequence length $n$. On three long-context
benchmarks our method matches or exceeds strong dense baselines while
running $4{\times}$ faster on $16$K-token inputs. A controlled ablation
shows the gain stems from a kernel feature map rather than auxiliary
losses.
\end{abstract}

\section{Introduction}
Transformers achieve state-of-the-art results across natural language
tasks, but the quadratic cost of self-attention makes them brittle on
long inputs~\cite{vaswani2017attention,tay2022efficient}. We revisit
linear attention~\cite{katharopoulos2020transformers} and show that a
simple kernel feature map suffices to recover the quality gap without
auxiliary losses or extra parameters.

\section{Related Work}
Efficient attention has been studied along three axes: sparsity,
factorisation, and kernel approximation. We focus on the third axis
and clarify when first-order kernels suffice. In contrast to recent
state-space approaches we keep the standard attention block intact,
which simplifies deployment.

\section{Method}
Let $Q, K, V \in \mathbb{R}^{n \times d}$. Standard attention computes
\begin{equation}
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\Bigl(\frac{Q K^\top}{\sqrt{d}}\Bigr) V.
\label{eq:softmax}
\end{equation}
Our linear variant replaces softmax with a positive feature map $\phi$:
\begin{equation}
\mathrm{LinAttn}(Q,K,V) = \frac{\phi(Q) \bigl(\phi(K)^\top V\bigr)}
                              {\phi(Q) \bigl(\phi(K)^\top \mathbf{1}\bigr)}.
\label{eq:linear}
\end{equation}
Equation~\ref{eq:linear} avoids materialising the $n \times n$ attention
matrix; the cost is $O(n d^2)$, linear in sequence length.

\section{Experiments}
\textbf{Setup.} We pretrain a $350$M-parameter decoder on a $30$B-token
corpus and evaluate on PG19, ProofPile, and LongBench. All baselines
share identical optimiser, schedule, and tokenizer.

\begin{table}[h]
\centering
\caption{Perplexity (lower is better) on three long-context corpora.}
\label{tab:main}
\begin{tabular}{lccc}
\toprule
Method        & PG19 & ProofPile & LongBench \\
\midrule
Softmax       & 12.4 & 3.8 & 18.1 \\
Sparse        & 12.7 & 4.0 & 18.6 \\
\textbf{Ours} & \textbf{11.9} & \textbf{3.6} & \textbf{17.4} \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Result.} As shown in Table~\ref{tab:main}, our method beats both
baselines on every corpus. The wallclock measurement (Figure deferred to
appendix) confirms a $4{\times}$ speedup at $16$K tokens.

\section{Conclusion}
Linear attention with a positive feature map recovers full softmax
quality on long-context modelling while remaining $O(n)$ in time and
memory. We release training code and checkpoints upon acceptance.

\bibliography{refs}
\end{document}
"""

REFS_BIB = """\
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
@article{tay2022efficient,
  title={Efficient Transformers: A Survey},
  author={Tay, Yi and Dehghani, Mostafa and Bahri, Dara and Metzler, Donald},
  journal={ACM Computing Surveys},
  volume={55},
  number={6},
  pages={1--28},
  year={2022}
}
@inproceedings{katharopoulos2020transformers,
  title={Transformers are {RNN}s: Fast autoregressive transformers with linear attention},
  author={Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and Fleuret, Fran{\\c{c}}ois},
  booktitle={International Conference on Machine Learning},
  year={2020}
}
"""


THIRD_PARTY = {
    "iclr2026":    REPO / "third_party" / "iclr2026" / "iclr2026",
    "cvpr2024":    REPO / "third_party" / "cvpr2024",
    "icml2024":    REPO / "third_party" / "icml2024",
    "neurips2024": REPO / "third_party" / "neurips2024",
    "acl_arr":     REPO / "third_party" / "acl_arr",
    "arxiv_plain": None,
}


def stage_assets(venue_id: str, dst: Path) -> list[str]:
    src = THIRD_PARTY[venue_id]
    copied: list[str] = []
    if not src:
        return copied
    for ext in ("*.sty", "*.bst", "*.tex", "*.cls"):
        for f in src.rglob(ext):
            if "__MACOSX" in str(f):
                continue
            target = dst / f.name
            shutil.copy(f, target)
            copied.append(f.name)
    return copied


def compile_with_tectonic(workdir: Path) -> dict:
    proc = subprocess.run(
        ["tectonic", "--keep-logs", "paper.tex"],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        timeout=180,
    )
    pdf = workdir / "paper.pdf"
    return {
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr.strip().splitlines()[-4:],
        "pdf_exists": pdf.exists(),
        "pdf_bytes": pdf.stat().st_size if pdf.exists() else 0,
    }


def pdf_pages(pdf: Path) -> int | None:
    if not pdf.exists():
        return None
    try:
        out = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        for ln in out.splitlines():
            if ln.startswith("Pages:"):
                return int(ln.split(":", 1)[1].strip())
    except Exception:
        return None
    return None


def column_check(pdf: Path) -> str:
    """Heuristic: in two-column PDFs, ``Multi-`` hyphenates within the first 12 lines."""
    if not pdf.exists():
        return "n/a"
    try:
        out = subprocess.check_output(
            ["pdftotext", "-layout", str(pdf), "-"], text=True
        )
    except Exception:
        return "n/a"
    first = "\n".join(out.splitlines()[:12])
    if "Multi-\nparagraph" in first or "Multi-" in first.split("\n")[-3:]:
        return "two-column-ish"
    # Two-column body lines are typically shorter (≈55 chars vs ≈90).
    body_lines = [
        ln for ln in out.splitlines() if 20 < len(ln.strip()) < 200
    ][:30]
    if not body_lines:
        return "?"
    avg = sum(len(ln) for ln in body_lines) / len(body_lines)
    return f"avg_line={avg:.0f}ch ({'two_col' if avg < 70 else 'single_col'})"


def main() -> int:
    out_root = Path("/tmp/full_paper_demo")
    shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir()

    print("=" * 78)
    print("Venue router routes 4 fixtures → expected venue")
    print("=" * 78)
    fixtures = [
        ("CV",     {"title": "Diffusion-based image detection", "claim_type": "empirical",
                    "domain": "vision", "has_real_data": True, "tier": 1,
                    "page_count_estimate": 8}),
        ("NLP",    {"title": "Cross-lingual transfer for NER", "claim_type": "empirical",
                    "domain": "nlp", "has_real_data": True, "tier": 2,
                    "page_count_estimate": 9}),
        ("ML",     {"title": "Long-context linear attention", "claim_type": "empirical",
                    "domain": "ml", "has_real_data": True, "tier": 1,
                    "page_count_estimate": 9}),
        ("Theory", {"title": "Proof of convergence",  "claim_type": "theory",
                    "domain": "theory", "has_real_data": False, "tier": 2,
                    "page_count_estimate": 14}),
    ]
    cfg = venue_router.load_venue_config()
    for tag, state in fixtures:
        result = venue_router.evaluate_venues(state, cfg)
        if result["selected"]:
            picked = result["selected"]["venue"].template_id
            score = result["selected"]["breakdown"]["score"]
            print(f"  {tag:7s} → {picked:14s}  score={score:.2f}")

    summary: list[dict] = []
    print()
    print("=" * 78)
    print("For each venue: normalize source → stage assets → tectonic compile")
    print("=" * 78)
    # Render plan: every venue gets the submission-mode build (default), and
    # the four venues that expose a submission/camera-ready toggle each get a
    # second ``camera-ready`` build (``submission_mode=False``) so reviewers
    # can eyeball the line-numbers-vs-final difference side-by-side.
    DUAL_MODE_VENUES = ("iclr2026", "neurips2024", "acl_arr", "cvpr2024")
    build_plan: list[tuple[str, str, dict]] = []
    for venue_id in sorted(list_adapters()):
        build_plan.append((venue_id, venue_id, {}))
        if venue_id in DUAL_MODE_VENUES:
            build_plan.append((venue_id, f"{venue_id}_camera_ready",
                               {"submission_mode": False}))

    for venue_id, build_id, normalize_kwargs in build_plan:
        ad = get_adapter(venue_id)
        venue_dir = out_root / build_id
        venue_dir.mkdir()
        # write refs.bib first so adapter sees real file
        (venue_dir / "refs.bib").write_text(REFS_BIB)
        stage_assets(venue_id, venue_dir)
        # ICLR also wants its preamble math_commands.tex preserved
        # ICLR + all 3 stub adapters (NeurIPS/ACL/CVPR) accept submission_mode;
        # arxiv_plain ignores unknown kwargs via its signature so this is safe.
        try:
            tex = ad.normalize_source(PAPER_BODY, **normalize_kwargs)
        except TypeError:
            tex = ad.normalize_source(PAPER_BODY)
        (venue_dir / "paper.tex").write_text(tex)
        # ICLR adapter writes the bundle via copy_files too; mirror that side.
        try:
            ad.copy_files(venue_dir)
        except Exception:
            pass
        compile_res = compile_with_tectonic(venue_dir)
        lint = lint_manuscript(source=tex, adapter=ad, page_count=8)
        summary.append({
            "venue": build_id,
            "column_layout": ad.column_layout,
            "bibstyle": ad.bibstyle_name,
            "max_pages": ad.max_pages,
            "compile_returncode": compile_res["returncode"],
            "pdf_bytes": compile_res["pdf_bytes"],
            "pdf_pages": pdf_pages(venue_dir / "paper.pdf"),
            "rendered_columns": column_check(venue_dir / "paper.pdf"),
            "lint_pass": lint["pass"],
            "lint_summary": lint["summary"],
        })

    print()
    print(f"{'venue':14s} {'col':14s} {'bib':24s} {'pages':>5s} {'pdfB':>7s} {'render':28s} lint")
    print("-" * 110)
    for row in summary:
        print(
            f"{row['venue']:14s} {row['column_layout']:14s} "
            f"{row['bibstyle']:24s} "
            f"{str(row['pdf_pages']):>5s} "
            f"{row['pdf_bytes']:>7d} "
            f"{row['rendered_columns']:28s} "
            f"{'PASS' if row['lint_pass'] else 'FAIL'}"
        )

    bundle_path = out_root / "demo_summary.json"
    bundle_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"[ok] summary written to {bundle_path}")
    print(f"[ok] PDFs under {out_root}/<venue>/paper.pdf")
    return 0 if all(r["compile_returncode"] == 0 for r in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
