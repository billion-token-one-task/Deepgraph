import io
import tarfile
import unittest
from unittest import mock

from ingestion import pdf_parser


class ArxivSourceParserTests(unittest.TestCase):
    def test_tex_files_are_extracted_from_arxiv_tarball(self):
        buf = io.BytesIO()
        tex = b"""\\documentclass{article}
\\title{A Source First Paper}
\\begin{document}
\\section{Results}
Method A reaches 91.2\\% accuracy on Dataset X.
\\caption{Main benchmark table}
\\end{document}
"""
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            info = tarfile.TarInfo("paper/main.tex")
            info.size = len(tex)
            tf.addfile(info, io.BytesIO(tex))

        files = pdf_parser._tex_files_from_source_blob(buf.getvalue())
        self.assertEqual(len(files), 1)
        plain = pdf_parser._latex_to_plaintext(files[0][1])
        self.assertIn("A Source First Paper", plain)
        self.assertIn("Results", plain)
        self.assertIn("91.2", plain)

    def test_source_auto_uses_tex_before_pdf(self):
        with (
            mock.patch.object(pdf_parser, "PDF_TEXT_BACKEND", "source_auto"),
            mock.patch.object(pdf_parser, "extract_text_arxiv_source", return_value="x" * 1000),
            mock.patch.object(pdf_parser, "download_pdf") as download_pdf,
        ):
            main, appendix = pdf_parser.get_paper_text_parts("2606.00001", abstract="fallback")

        self.assertEqual(len(main), 1000)
        self.assertEqual(appendix, "")
        download_pdf.assert_not_called()


if __name__ == "__main__":
    unittest.main()
