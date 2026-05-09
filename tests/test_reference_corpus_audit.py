import tempfile
import textwrap
import unittest
from pathlib import Path

from agents.reference_corpus_audit import audit_against_reference_corpus, generated_manuscript_profile


class ReferenceCorpusAuditTests(unittest.TestCase):
    def test_generated_profile_detects_problem_spine_and_counts(self):
        tex = textwrap.dedent(
            r"""
            \section{Introduction}
            Problem and motivation. Our method addresses the result gap \citep{a,b}.
            \section{Method}
            \begin{equation}x=1\end{equation}
            \section{Experiments}
            \begin{table}x\end{table}
            """
        )

        profile = generated_manuscript_profile(
            main_tex=tex,
            page_count=8,
            figure_count=3,
            bibliography_entry_count=12,
        )

        self.assertEqual(profile["section_count"], 3)
        self.assertEqual(profile["citation_command_count"], 1)
        self.assertTrue(profile["has_problem_motivation_spine"])
        self.assertEqual(profile["table_count"], 1)
        self.assertEqual(profile["equation_count"], 1)

    def test_audit_blocks_missing_expected_section_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = audit_against_reference_corpus(
                main_tex=r"\section{Background} Some text.",
                page_count=2,
                figure_count=0,
                bibliography_entry_count=0,
                corpus_dir=Path(tmp) / "missing",
            )

        self.assertFalse(audit["pass"])
        issues = " ".join(item["issue"] for item in audit["issues"])
        self.assertIn("Reference PDF corpus", issues)
        self.assertIn("missing expected section", issues)
        self.assertIn("Problem-motivation-method-result", issues)


if __name__ == "__main__":
    unittest.main()
