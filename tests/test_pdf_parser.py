"""Tests for main-body/appendix splitting in PDF text extraction."""
import unittest

from ingestion.pdf_parser import MAX_MAIN_TEXT_CHARS, split_main_and_appendix_text


class TestPdfParserSplit(unittest.TestCase):
    def test_splits_on_appendix_heading(self):
        text = "Introduction\nCore method details.\nAppendix\nAblation tables and implementation details."
        main_text, appendix_text = split_main_and_appendix_text(text)
        self.assertIn("Core method details.", main_text)
        self.assertTrue(appendix_text.startswith("Appendix"))
        self.assertIn("Ablation tables", appendix_text)

    def test_overflow_moves_to_appendix_when_no_heading(self):
        text = ("A" * (MAX_MAIN_TEXT_CHARS + 50)).strip()
        main_text, appendix_text = split_main_and_appendix_text(text)
        self.assertEqual(len(main_text), MAX_MAIN_TEXT_CHARS)
        self.assertEqual(len(appendix_text), 50)


if __name__ == "__main__":
    unittest.main()
