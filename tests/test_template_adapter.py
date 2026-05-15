"""Tests for TemplateAdapter base + ICLR2026 / arXiv plain adapters.

Issue #11/#12 D1. Coverage:
1. ICLR adapter.copy_files() materialises every shipped sty/bst/tex.
2. arXiv adapter.copy_files() is a no-op (no shipped venue assets).
3. inject_preamble is idempotent: f(f(x)) == f(x).
4. normalize_source handles bibstyle replacement on both branches.
5. The legacy ``normalize_latex_source(force_iclr2026=...)`` shim returns
   byte-identical output to the adapter call (regression guard for the
   ``bundle_format=='conference'`` byte-level diff acceptance criterion).
"""

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TemplateAdapterTests(unittest.TestCase):
    def setUp(self):
        from agents.manuscript_templates import get_adapter
        self.iclr = get_adapter("iclr2026")
        self.arx = get_adapter("arxiv_plain")
        self.tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "dg_test_template_adapter"
        self.tmp.mkdir(parents=True, exist_ok=True)
        # Wipe between runs so copy_files results are unambiguous
        for p in self.tmp.glob("*"):
            try:
                p.unlink()
            except IsADirectoryError:
                pass

    def test_iclr_copy_files_includes_all_assets(self):
        copied = self.iclr.copy_files(self.tmp)
        # ICLR template ships at least the .sty + .bst + math_commands.tex
        expected_must = {
            "iclr2026_conference.sty",
            "iclr2026_conference.bst",
            "math_commands.tex",
        }
        self.assertTrue(
            expected_must.issubset(set(copied)),
            f"missing required ICLR assets: copied={copied}",
        )
        for name in copied:
            self.assertTrue((self.tmp / name).exists(), f"{name} not materialised")

    def test_arxiv_copy_files_is_noop(self):
        copied = self.arx.copy_files(self.tmp)
        self.assertEqual(copied, [], "arxiv_plain must not copy venue assets")

    def test_inject_preamble_is_idempotent(self):
        raw = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Hello" "\n"
            r"\end{document}" "\n"
        )
        once = self.iclr.inject_preamble(raw)
        twice = self.iclr.inject_preamble(once)
        self.assertEqual(once, twice, "inject_preamble must be idempotent")
        self.assertIn("iclr2026_conference", once)
        self.assertIn("math_commands.tex", once)

    def test_normalize_source_handles_bibstyle_on_both_branches(self):
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Some content." "\n"
            r"\bibliography{refs}" "\n"
            r"\end{document}" "\n"
        )
        iclr_out = self.iclr.normalize_source(body)
        self.assertIn(r"\bibliographystyle{iclr2026_conference}", iclr_out)
        arx_out = self.arx.normalize_source(body)
        self.assertIn(r"\bibliographystyle{plain}", arx_out)
        self.assertNotIn("iclr2026_conference", arx_out)

    def test_submission_mode_toggle_changes_iclr_preamble(self):
        """ICLR adapter must support a camera-ready render path.

        ``submission_mode=False`` MUST emit the ``\\iclrfinalcopy`` macro
        toggle (the official switch the ICLR sty exposes — it does NOT take
        a ``[final]`` package option) and replace the anonymous author block
        with a real-author placeholder so reviewers can eyeball the final
        paper layout.
        """
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Body." "\n"
            r"\end{document}" "\n"
        )
        sub = self.iclr.inject_preamble(body, submission_mode=True)
        final = self.iclr.inject_preamble(body, submission_mode=False)
        # Submission build: no final-copy toggle, anonymous authors.
        self.assertNotIn(r"\iclrfinalcopy", sub)
        self.assertIn("Anonymous authors", sub)
        # Camera-ready build: \iclrfinalcopy present, no anonymous author.
        self.assertIn(r"\iclrfinalcopy", final)
        self.assertNotIn("Anonymous authors", final)
        # Default (no kwarg) must equal submission_mode=True for back-compat.
        self.assertEqual(sub, self.iclr.inject_preamble(body))
        # normalize_source threads the kwarg through.
        self.assertIn(
            r"\iclrfinalcopy",
            self.iclr.normalize_source(body, submission_mode=False),
        )

    def test_legacy_shim_byte_equivalent(self):
        """Pre-D1 normalize_latex_source signature still produces same bytes."""
        from agents.paper_orchestra_pipeline import normalize_latex_source
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Sample doc." "\n"
            r"\bibliography{refs}" "\n"
            r"\end{document}" "\n"
        )
        # Legacy True branch must equal direct ICLR adapter call
        self.assertEqual(
            normalize_latex_source(body, force_iclr2026=True),
            self.iclr.normalize_source(body),
        )
        # Legacy False branch must equal direct arXiv adapter call
        self.assertEqual(
            normalize_latex_source(body, force_iclr2026=False),
            self.arx.normalize_source(body),
        )

    def test_normalize_latex_source_template_id_routes_to_adapter(self):
        """The new ``template_id`` kwarg dispatches through the adapter registry.

        Closes the legacy ``assemble_main_tex`` / ``pick_main_tex`` hard-coded
        ICLR boundary documented in PR #10. Each registered venue must be
        reachable via ``normalize_latex_source(template_id=...)``.
        """
        from agents.paper_orchestra_pipeline import normalize_latex_source
        from agents.manuscript_templates import get_adapter
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Body." "\n"
            r"\bibliography{refs}" "\n"
            r"\end{document}" "\n"
        )
        for template_id in ("iclr2026", "neurips2024", "icml2024", "acl_arr", "cvpr2024", "arxiv_plain"):
            with self.subTest(template_id=template_id):
                out = normalize_latex_source(body, template_id=template_id)
                self.assertEqual(out, get_adapter(template_id).normalize_source(body))

    def test_normalize_latex_source_template_id_overrides_force_flag(self):
        """``template_id`` takes precedence over the legacy ``force_iclr2026`` flag."""
        from agents.paper_orchestra_pipeline import normalize_latex_source
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Body." "\n"
            r"\bibliography{refs}" "\n"
            r"\end{document}" "\n"
        )
        # force_iclr2026=True but template_id=arxiv_plain → arxiv wins
        out = normalize_latex_source(body, force_iclr2026=True, template_id="arxiv_plain")
        self.assertNotIn("iclr2026_conference", out)
        self.assertIn(r"\bibliographystyle{plain}", out)

    def test_pick_main_tex_routes_through_adapter(self):
        """``pick_main_tex(template_id=...)`` produces venue-specific output.

        Verifies the legacy bundle-loop hard-coding (issue #11 known boundary)
        is gone: a non-ICLR ``template_id`` yields a non-ICLR preamble even
        when ``bundle_format == "conference"``.
        """
        from agents.paper_orchestra_pipeline import pick_main_tex
        state = {
            "title": "T",
            "baseline_metric_name": "acc",
            "baseline_metric_value": 0.1,
            "problem_statement": "P",
            "method_summary": "M",
        }
        orchestrated: dict = {}
        # Default (no template_id) → conference bundle still gets ICLR (back-compat)
        default_out = pick_main_tex(orchestrated, state, "conference")
        self.assertIn("iclr2026_conference", default_out)
        # Explicit neurips2024 → NeurIPS preamble (sty basename `neurips_2024`),
        # no ICLR sty.
        nips_out = pick_main_tex(orchestrated, state, "conference", template_id="neurips2024")
        self.assertIn(r"\usepackage{neurips_2024}", nips_out)
        self.assertIn(r"\bibliographystyle{unsrtnat}", nips_out)
        self.assertNotIn("iclr2026_conference", nips_out)
        # Explicit acl_arr (two-column venue) → twocolumn option + acl sty
        acl_out = pick_main_tex(orchestrated, state, "conference", template_id="acl_arr")
        self.assertIn("twocolumn", acl_out)
        self.assertIn(r"\usepackage", acl_out)
        self.assertNotIn("iclr2026_conference", acl_out)


if __name__ == "__main__":
    unittest.main()
