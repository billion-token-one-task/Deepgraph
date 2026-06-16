import unittest

from agents.tex_code_agent import repair_latex_source


class TexCodeAgentTests(unittest.TestCase):
    def test_missing_adjustbox_package_is_removed(self):
        source = r"""\documentclass{article}
\usepackage{graphicx}
\usepackage{adjustbox}
\begin{document}
Body.
\end{document}
"""
        log = "! LaTeX Error: File `adjustbox.sty' not found."

        repaired, report = repair_latex_source(source, compile_log=log)

        self.assertNotIn(r"\usepackage{adjustbox}", repaired)
        self.assertTrue(report["changed"])
        self.assertIn(
            {"kind": "package_removed", "package": "adjustbox", "reason": "missing_optional_package"},
            report["actions"],
        )

    def test_existing_algorithm_float_prevents_duplicate_fallback(self):
        source = r"""\documentclass{article}
\usepackage{float}
\newfloat{algorithm}{tbp}{loa}
\newenvironment{algorithmic}[1][]{\begin{list}{}{\leftmargin=1em}}{\end{list}}
\title{T}
% TeX Code Agent fallback for environments normally provided by algorithm/algpseudocode.
\usepackage{float}
\newenvironment{algorithm}[1][]{
}
\newenvironment{algorithmic}[1][]{\begin{list}{}{\leftmargin=1.5em}}{\end{list}}
\begin{document}
\begin{algorithm}\begin{algorithmic}\State ok\end{algorithmic}\end{algorithm}
\end{document}
"""
        repaired, report = repair_latex_source(source, compile_log="")

        self.assertEqual(repaired.count(r"\newfloat{algorithm}"), 1)
        self.assertNotIn("TeX Code Agent fallback", repaired)
        self.assertIn({"kind": "duplicate_algorithm_fallback_removed"}, report["actions"])


if __name__ == "__main__":
    unittest.main()
