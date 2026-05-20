"""Tests for the four D2 (#13) top-tier venue adapters.

Coverage matrix (≥ 2 cases per adapter as required by the issue):

| Adapter        | Tests                                                     |
| -------------- | --------------------------------------------------------- |
| NeurIPS 2024   | copy_files materialises assets; column_layout single      |
|                | inject_preamble idempotent + injects \\usepackage{neurips}|
| ICML 2024      | bibstyle icml2024 applied; max_pages=8                    |
|                | inject_preamble idempotent (sty drives \\twocolumn)       |
| ACL ARR        | column_layout two_column; bibstyle acl_natbib applied     |
|                | inject_preamble idempotent                                |
| CVPR 2024      | column_layout two_column; max_pages=8; bibstyle           |
|                | copy_files materialises real cvpr.sty + ieeenat_fullname  |

Plus router fixtures asserting CV → cvpr2024, NLP → acl_arr, ML/RL → one
of {neurips2024, iclr2026, icml2024} (rule_set still picks deterministic).
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


class TopVenueAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmproot = Path(tempfile.mkdtemp(prefix="dg_d2_adapters_"))

    def _scratch(self, name: str) -> Path:
        d = self._tmproot / name
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*"):
            try:
                p.unlink()
            except IsADirectoryError:
                pass
        return d

    # ------------------------------------------------------------------
    # NeurIPS 2024
    # ------------------------------------------------------------------
    def test_neurips_copy_files_materialises_stub(self):
        from agents.manuscript_templates import get_adapter
        adapter = get_adapter("neurips2024")
        out = self._scratch("nrps_copy")
        copied = adapter.copy_files(out)
        self.assertIn("neurips_2024.sty", copied)
        self.assertIn("README.md", copied)
        self.assertTrue((out / "neurips_2024.sty").exists())
        self.assertEqual(adapter.column_layout, "single_column")

    def test_neurips_submission_mode_toggles_final_option(self):
        """``submission_mode=False`` MUST emit the ``[final]`` package option
        which the upstream ``neurips_2024.sty`` uses to switch from the
        line-numbered double-blind review layout to the camera-ready layout.
        """
        from agents.manuscript_templates import get_adapter
        adapter = get_adapter("neurips2024")
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Body" "\n"
            r"\end{document}" "\n"
        )
        sub = adapter.inject_preamble(body, submission_mode=True)
        final = adapter.inject_preamble(body, submission_mode=False)
        self.assertIn(r"\usepackage{neurips_2024}", sub)
        self.assertNotIn(r"\usepackage[final]{neurips_2024}", sub)
        self.assertIn(r"\usepackage[final]{neurips_2024}", final)
        # Default (no kwarg) stays byte-equivalent to submission_mode=True.
        self.assertEqual(sub, adapter.inject_preamble(body))
        # normalize_source threads the kwarg through.
        self.assertIn(
            r"\usepackage[final]{neurips_2024}",
            adapter.normalize_source(body, submission_mode=False),
        )

    def test_neurips_inject_preamble_idempotent(self):
        from agents.manuscript_templates import get_adapter
        adapter = get_adapter("neurips2024")
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Hello" "\n"
            r"\end{document}" "\n"
        )
        once = adapter.inject_preamble(body)
        twice = adapter.inject_preamble(once)
        self.assertEqual(once, twice, "inject_preamble must be idempotent")
        self.assertIn(r"\usepackage{neurips_2024}", once)

    # ------------------------------------------------------------------
    # ICML 2024
    # ------------------------------------------------------------------
    def test_icml_normalize_applies_icml2024_bibstyle(self):
        from agents.manuscript_templates import get_adapter
        adapter = get_adapter("icml2024")
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Body." "\n"
            r"\bibliography{refs}" "\n"
            r"\end{document}" "\n"
        )
        out = adapter.normalize_source(body)
        self.assertIn(r"\bibliographystyle{icml2024}", out)
        self.assertEqual(adapter.max_pages, 8)

    def test_icml_inject_preamble_idempotent(self):
        from agents.manuscript_templates import get_adapter
        adapter = get_adapter("icml2024")
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Body" "\n"
            r"\end{document}" "\n"
        )
        once = adapter.inject_preamble(body)
        twice = adapter.inject_preamble(once)
        self.assertEqual(once, twice)
        self.assertIn(r"\usepackage{icml2024}", once)
        # icml2024.sty itself issues \twocolumn, so the adapter declares
        # two_column to keep FormatLinter (D3) aligned with reality.
        self.assertEqual(adapter.column_layout, "two_column")

    # ------------------------------------------------------------------
    # ACL ARR
    # ------------------------------------------------------------------
    def test_acl_arr_normalize_applies_acl_natbib(self):
        from agents.manuscript_templates import get_adapter
        adapter = get_adapter("acl_arr")
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Sentence." "\n"
            r"\bibliography{refs}" "\n"
            r"\end{document}" "\n"
        )
        out = adapter.normalize_source(body)
        self.assertIn(r"\bibliographystyle{acl_natbib}", out)
        self.assertEqual(adapter.column_layout, "two_column")

    def test_acl_arr_submission_mode_toggles_review_option(self):
        """ACL .sty defaults to camera-ready. ``submission_mode=True`` must
        emit ``[review]`` to enter the line-numbered double-blind build.
        ``submission_mode=False`` falls back to the bare default (final).
        """
        from agents.manuscript_templates import get_adapter
        adapter = get_adapter("acl_arr")
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"X" "\n"
            r"\end{document}" "\n"
        )
        sub = adapter.inject_preamble(body, submission_mode=True)
        final = adapter.inject_preamble(body, submission_mode=False)
        self.assertIn(r"\usepackage[review]{acl}", sub)
        self.assertNotIn(r"\usepackage[review]{acl}", final)
        self.assertIn(r"\usepackage{acl}", final)
        # Default (no kwarg) stays byte-equivalent to submission_mode=True.
        self.assertEqual(sub, adapter.inject_preamble(body))

    def test_acl_arr_inject_preamble_idempotent(self):
        from agents.manuscript_templates import get_adapter
        adapter = get_adapter("acl_arr")
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Text" "\n"
            r"\end{document}" "\n"
        )
        once = adapter.inject_preamble(body)
        twice = adapter.inject_preamble(once)
        self.assertEqual(once, twice)
        # ACL defaults to ``submission_mode=True`` → ``[review]`` option,
        # which switches the upstream .sty into the line-numbered double-blind
        # review build the routing pipeline actually wants to submit.
        self.assertIn(r"\usepackage[review]{acl}", once)

    # ------------------------------------------------------------------
    # CVPR 2024
    # ------------------------------------------------------------------
    def test_cvpr_copy_files_materialises_stub(self):
        from agents.manuscript_templates import get_adapter
        adapter = get_adapter("cvpr2024")
        out = self._scratch("cvpr_copy")
        copied = adapter.copy_files(out)
        self.assertIn("cvpr.sty", copied)
        self.assertIn("README.md", copied)
        self.assertEqual(adapter.column_layout, "two_column")
        self.assertEqual(adapter.max_pages, 8)

    def test_cvpr_submission_mode_toggles_review_option(self):
        """CVPR .sty defaults to camera-ready. ``submission_mode=True`` must
        emit ``[review]`` to enter the line-numbered double-blind build.
        """
        from agents.manuscript_templates import get_adapter
        adapter = get_adapter("cvpr2024")
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"X" "\n"
            r"\end{document}" "\n"
        )
        sub = adapter.inject_preamble(body, submission_mode=True)
        final = adapter.inject_preamble(body, submission_mode=False)
        self.assertIn(r"\usepackage[review]{cvpr}", sub)
        self.assertNotIn(r"\usepackage[review]{cvpr}", final)
        self.assertIn(r"\usepackage{cvpr}", final)

    def test_cvpr_normalize_applies_ieeenat_fullname_bibstyle(self):
        from agents.manuscript_templates import get_adapter
        adapter = get_adapter("cvpr2024")
        body = (
            r"\documentclass{article}" "\n"
            r"\begin{document}" "\n"
            r"Vision body." "\n"
            r"\bibliography{refs}" "\n"
            r"\end{document}" "\n"
        )
        out = adapter.normalize_source(body)
        self.assertIn(r"\bibliographystyle{ieeenat_fullname}", out)

    # ------------------------------------------------------------------
    # Router integration: domain-specific states route to the right venue.
    # ------------------------------------------------------------------
    def test_router_routes_vision_state_to_cvpr(self):
        from agents.venue_router import evaluate_venues, load_venue_config
        state = {
            "title": "Real-time image segmentation with diffusion priors",
            "claim_type": "empirical",
            "domain": "vision",
            "has_real_data": True,
            "tier": 1,
            "page_count_estimate": 8,
        }
        result = evaluate_venues(state, load_venue_config())
        self.assertIsNotNone(result["selected"])
        self.assertEqual(result["selected"]["venue"].template_id, "cvpr2024")

    def test_router_routes_nlp_state_to_acl_arr(self):
        from agents.venue_router import evaluate_venues, load_venue_config
        state = {
            "title": "A new tokenization scheme for low-resource language models",
            "claim_type": "empirical",
            "domain": "nlp",
            "has_real_data": True,
            "tier": 1,
            "page_count_estimate": 9,
        }
        result = evaluate_venues(state, load_venue_config())
        self.assertIsNotNone(result["selected"])
        self.assertEqual(result["selected"]["venue"].template_id, "acl_arr")

    def test_router_distinguishes_three_domains(self):
        """CV/NLP/ML states must each route to a different venue."""
        from agents.venue_router import evaluate_venues, load_venue_config
        venues = load_venue_config()
        cv = evaluate_venues(
            {
                "title": "Diffusion-based image detection",
                "claim_type": "empirical",
                "domain": "vision",
                "has_real_data": True,
                "tier": 1,
                "page_count_estimate": 8,
            },
            venues,
        )
        nlp = evaluate_venues(
            {
                "title": "Dialogue language model evaluation",
                "claim_type": "empirical",
                "domain": "nlp",
                "has_real_data": True,
                "tier": 1,
                "page_count_estimate": 9,
            },
            venues,
        )
        ml = evaluate_venues(
            {
                "title": "Self-supervised representation learning with deep learning",
                "claim_type": "empirical",
                "domain": "ml",
                "has_real_data": True,
                "tier": 1,
                "page_count_estimate": 9,
            },
            venues,
        )
        chosen = {
            "cv": cv["selected"]["venue"].template_id,
            "nlp": nlp["selected"]["venue"].template_id,
            "ml": ml["selected"]["venue"].template_id,
        }
        # CV must be CVPR, NLP must be ACL-ARR. ML state must NOT be either.
        self.assertEqual(chosen["cv"], "cvpr2024")
        self.assertEqual(chosen["nlp"], "acl_arr")
        self.assertNotIn(chosen["ml"], {"cvpr2024", "acl_arr"})

    def test_all_six_adapters_registered(self):
        from agents.manuscript_templates import list_adapters
        ids = set(list_adapters())
        self.assertEqual(
            ids,
            {"iclr2026", "arxiv_plain", "neurips2024", "icml2024", "acl_arr", "cvpr2024"},
        )


if __name__ == "__main__":
    unittest.main()
