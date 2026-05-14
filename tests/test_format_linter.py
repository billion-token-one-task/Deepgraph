"""Tests for ``agents.format_linter``.

Issue #11/#14 (D3). Coverage:

1. Happy path: normalised ICLR source passes every check.
2. Missing ``\\documentclass`` → error severity, pass=False.
3. Mismatched bibstyle → error.
4. Page count over budget → warning (small) and error (large overage).
5. Column-layout violation (two_column venue with ``\\textwidth`` figure) → warning.
6. Figure-grid density violation (5-panel row on single_column) → warning.
7. ``persist_lint_run`` + ``get_lint_run`` round-trip.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FormatLinterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="dg_format_lint_")
        os.environ["DEEPGRAPH_DATABASE_URL"] = ""
        os.environ["DEEPGRAPH_DB_PATH"] = str(Path(cls._tmpdir) / "lint.db")
        from db import database as db
        # Explicit reset so prior tests that override db.DB_PATH (and clean up
        # their tempdir) don't leave us pointing at a deleted file.
        for attr in ("sqlite_conn", "pg_conn", "conn"):
            if hasattr(db._local, attr):
                try:
                    getattr(db._local, attr).close()
                except Exception:
                    pass
                delattr(db._local, attr)
        db.DB_PATH = Path(os.environ["DEEPGRAPH_DB_PATH"])
        db.init_db()
        cls._db = db

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------
    def test_normalised_iclr_source_passes_all_checks(self):
        from agents.manuscript_templates import get_adapter
        from agents.format_linter import lint_manuscript
        adapter = get_adapter("iclr2026")
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Hello world." "\n"
            r"\bibliography{refs}" "\n"
            r"\end{document}" "\n"
        )
        normalised = adapter.normalize_source(body)
        result = lint_manuscript(normalised, adapter, page_count=8)
        self.assertTrue(result["pass"], result)
        self.assertEqual(result["template_id"], "iclr2026")
        self.assertEqual(result["column_layout"], "single_column")
        self.assertEqual(len(result["checks"]), 7)
        self.assertEqual(result["summary"]["error_count"], 0)

    # ------------------------------------------------------------------
    # Individual check failures
    # ------------------------------------------------------------------
    def test_missing_documentclass_is_an_error(self):
        from agents.manuscript_templates import get_adapter
        from agents.format_linter import lint_manuscript
        adapter = get_adapter("iclr2026")
        source = r"\begin{document}body\end{document}"
        result = lint_manuscript(source, adapter)
        doc_check = next(c for c in result["checks"] if c["name"] == "documentclass_present")
        self.assertFalse(doc_check["passed"])
        self.assertEqual(doc_check["severity"], "error")
        self.assertFalse(result["pass"])

    def test_bibstyle_mismatch_is_an_error(self):
        from agents.manuscript_templates import get_adapter
        from agents.format_linter import lint_manuscript
        adapter = get_adapter("iclr2026")
        source = (
            r"\documentclass{article}\usepackage{graphicx}\usepackage{amsmath}"
            r"\usepackage{hyperref}\begin{document}body"
            r"\bibliographystyle{plain}\bibliography{refs}\end{document}"
        )
        result = lint_manuscript(source, adapter)
        bib_check = next(c for c in result["checks"] if c["name"] == "bibstyle_matches_venue")
        self.assertFalse(bib_check["passed"])
        self.assertEqual(bib_check["details"]["found"], "plain")
        self.assertEqual(bib_check["details"]["expected"], "iclr2026_conference")

    def test_page_count_overage_warning_vs_error(self):
        from agents.manuscript_templates import get_adapter
        from agents.format_linter import lint_manuscript
        adapter = get_adapter("iclr2026")  # max_pages=9
        source = (
            r"\documentclass{article}\usepackage{graphicx}\usepackage{amsmath}"
            r"\usepackage{hyperref}\begin{document}body\end{document}"
        )
        # 10 pages → over by 1 → warning (not error)
        warn = lint_manuscript(source, adapter, page_count=10)
        warn_check = next(c for c in warn["checks"] if c["name"] == "page_count_within_budget")
        self.assertEqual(warn_check["severity"], "warning")
        self.assertFalse(warn_check["passed"])
        # 15 pages → over by 6 → error
        err = lint_manuscript(source, adapter, page_count=15)
        err_check = next(c for c in err["checks"] if c["name"] == "page_count_within_budget")
        self.assertEqual(err_check["severity"], "error")
        self.assertFalse(err["pass"])

    # ------------------------------------------------------------------
    # Column-layout checks (D1 user feedback consumed here)
    # ------------------------------------------------------------------
    def test_two_column_venue_with_textwidth_figure_warns(self):
        from agents.manuscript_templates import get_adapter
        from agents.format_linter import lint_manuscript
        adapter = get_adapter("acl_arr")  # two_column
        source = (
            r"\documentclass{article}\usepackage{graphicx}\usepackage{amsmath}"
            r"\usepackage{hyperref}"
            r"\begin{document}"
            r"\begin{figure}[t]\includegraphics[width=\textwidth]{plot.pdf}\end{figure}"
            r"\bibliographystyle{acl_natbib}\bibliography{refs}\end{document}"
        )
        result = lint_manuscript(source, adapter)
        col_check = next(c for c in result["checks"] if c["name"] == "column_layout_consistency")
        self.assertFalse(col_check["passed"])
        self.assertEqual(col_check["severity"], "warning")
        self.assertEqual(col_check["details"]["layout"], "two_column")
        self.assertTrue(len(col_check["details"]["violations"]) >= 1)

    def test_single_column_grid_density_five_panels_warns(self):
        from agents.manuscript_templates import get_adapter
        from agents.format_linter import lint_manuscript
        adapter = get_adapter("iclr2026")  # single_column, cap=4
        # Build a 5-panel row using \\ separator (LaTeX double backslash).
        row = (
            r"\begin{subfigure}{a}\end{subfigure}"
            r"\begin{subfigure}{b}\end{subfigure}"
            r"\begin{subfigure}{c}\end{subfigure}"
            r"\begin{subfigure}{d}\end{subfigure}"
            r"\begin{subfigure}{e}\end{subfigure}"
        )
        source = (
            r"\documentclass{article}\usepackage{graphicx}\usepackage{amsmath}"
            r"\usepackage{hyperref}"
            r"\begin{document}"
            r"\begin{figure}[t]" + row + r"\end{figure}"
            r"\bibliographystyle{iclr2026_conference}\bibliography{refs}\end{document}"
        )
        result = lint_manuscript(source, adapter)
        grid_check = next(c for c in result["checks"] if c["name"] == "figure_grid_density")
        self.assertFalse(grid_check["passed"])
        self.assertEqual(grid_check["details"]["cap_per_row"], 4)
        self.assertEqual(grid_check["details"]["overflow"][0]["panel_count"], 5)

    def test_two_column_grid_density_three_panels_warns(self):
        from agents.manuscript_templates import get_adapter
        from agents.format_linter import lint_manuscript
        adapter = get_adapter("cvpr2024")  # two_column, cap=2
        row = (
            r"\begin{subfigure}{a}\end{subfigure}"
            r"\begin{subfigure}{b}\end{subfigure}"
            r"\begin{subfigure}{c}\end{subfigure}"
        )
        source = (
            r"\documentclass{article}\usepackage{graphicx}\usepackage{amsmath}"
            r"\usepackage{hyperref}"
            r"\begin{document}"
            r"\begin{figure}[t]" + row + r"\end{figure}"
            r"\bibliographystyle{ieee_fullname}\bibliography{refs}\end{document}"
        )
        result = lint_manuscript(source, adapter)
        grid_check = next(c for c in result["checks"] if c["name"] == "figure_grid_density")
        self.assertFalse(grid_check["passed"])
        self.assertEqual(grid_check["details"]["cap_per_row"], 2)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def test_persist_and_read_back_lint_run(self):
        from agents.manuscript_templates import get_adapter
        from agents.format_linter import lint_manuscript, persist_lint_run, get_lint_run
        adapter = get_adapter("iclr2026")
        source = (
            r"\documentclass{article}\usepackage{graphicx}\usepackage{amsmath}"
            r"\usepackage{hyperref}\begin{document}body"
            r"\bibliographystyle{iclr2026_conference}\bibliography{refs}\end{document}"
        )
        result = lint_manuscript(source, adapter, page_count=8)
        run_id = persist_lint_run(selection_id=42, adapter=adapter, lint_result=result)
        self.assertIsInstance(run_id, int)
        self.assertGreater(run_id, 0)
        readback = get_lint_run(run_id)
        self.assertIsNotNone(readback)
        self.assertEqual(readback["template_id"], "iclr2026")
        self.assertEqual(readback["selection_id"], 42)
        self.assertTrue(readback["pass"])
        self.assertEqual(len(readback["checks"]), 7)
        self.assertEqual(readback["rule_set"], "format_linter_v1")


if __name__ == "__main__":
    unittest.main()
