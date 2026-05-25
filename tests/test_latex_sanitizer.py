"""Tests for ``sanitize_main_tex_for_compile`` / ``repair_main_tex_structure``."""

from __future__ import annotations

import unittest

from agents.manuscript_submission_enrichment import (
    repair_main_tex_structure,
    sanitize_main_tex_for_compile,
    strip_llm_wrapper_markup,
)


class StripWrapperTests(unittest.TestCase):
    def test_drops_think_block(self) -> None:
        src = "<think>internal reasoning</think>\n\\documentclass{article}"
        self.assertEqual(strip_llm_wrapper_markup(src), r"\documentclass{article}")

    def test_drops_unclosed_think_tag(self) -> None:
        src = "<think>truncated\n\\documentclass{article}"
        out = strip_llm_wrapper_markup(src)
        self.assertTrue(out.startswith(r"\documentclass"))

    def test_strips_leading_markdown_chatter(self) -> None:
        src = "Some chatter then later \\documentclass{article}\n..."
        out = strip_llm_wrapper_markup(src)
        self.assertTrue(out.startswith(r"\documentclass"))


class RepairStructureTests(unittest.TestCase):
    def test_splices_inner_document(self) -> None:
        src = (
            r"\documentclass{article}\usepackage{iclr2026_conference}"
            r"\begin{document}\maketitle\begin{abstract}\end{abstract}"
            "<think>chatter</think>\n"
            r"\documentclass{article}\usepackage{graphicx}"
            r"\begin{document}\section{Intro}Body text.\end{document}"
        )
        repaired, notes = repair_main_tex_structure(src)
        self.assertTrue(notes.get("spliced_inner_document"))
        self.assertEqual(repaired.count(r"\documentclass"), 1)
        self.assertIn("Body text.", repaired)

    def test_dedupes_abstract(self) -> None:
        src = (
            r"\documentclass{article}\begin{document}"
            r"\begin{abstract}\begin{abstract}Hello.\end{abstract}\end{abstract}"
            r"\end{document}"
        )
        repaired, _ = repair_main_tex_structure(src)
        self.assertEqual(repaired.count(r"\begin{abstract}"), 1)
        self.assertEqual(repaired.count(r"\end{abstract}"), 1)

    def test_dedupes_cleveref(self) -> None:
        src = (
            r"\documentclass{article}\usepackage{cleveref}"
            r"\usepackage[capitalize]{cleveref}\begin{document}body\end{document}"
        )
        repaired, notes = repair_main_tex_structure(src)
        self.assertEqual(repaired.count(r"\usepackage"), 1)
        self.assertIn("[capitalize]", repaired)
        self.assertEqual(notes.get("deduped_cleveref"), 1)

    def test_relocates_content_after_end_document(self) -> None:
        src = (
            r"\documentclass{article}\begin{document}\label{mainbody:end}"
            r"\end{document}"
            "\n\\begin{table}Stranded.\\end{table}\n"
        )
        repaired, notes = repair_main_tex_structure(src)
        self.assertTrue(notes.get("relocated_post_end_document"))
        end_idx = repaired.find(r"\end{document}")
        self.assertIn("Stranded.", repaired[:end_idx])

    def test_citeA_rewritten_to_cite(self) -> None:
        src = r"\documentclass{article}\begin{document}Foo \citeA{key123}.\end{document}"
        repaired, notes = repair_main_tex_structure(src)
        self.assertIn(r"\cite{key123}", repaired)
        self.assertNotIn(r"\citeA", repaired)
        self.assertEqual(notes.get("rewrote_citation_aliases"), 1)


class SanitizeTopLevelTests(unittest.TestCase):
    def test_unicode_replaced(self) -> None:
        src = "\\documentclass{article}\\begin{document}x \u2264 y\\end{document}"
        out, _ = sanitize_main_tex_for_compile(src)
        self.assertIn(r"$\leq$", out)


if __name__ == "__main__":
    unittest.main()
