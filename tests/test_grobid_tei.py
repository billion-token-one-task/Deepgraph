"""Tests for GROBID TEI → plaintext."""
import unittest

from ingestion.grobid_tei import tei_xml_to_plaintext

SAMPLE_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader/>
  <text>
    <body>
      <div>
        <head n="1">Introduction</head>
        <p>We study neural networks for classification.</p>
        <p>Our model reaches 95.2% accuracy on ImageNet.</p>
      </div>
    </body>
  </text>
</TEI>
"""


class TestGrobidTei(unittest.TestCase):
    def test_extracts_paragraphs(self):
        text = tei_xml_to_plaintext(SAMPLE_TEI)
        self.assertIn("Introduction", text)
        self.assertIn("neural networks", text)
        self.assertIn("95.2%", text)

    def test_empty_on_garbage(self):
        self.assertEqual(tei_xml_to_plaintext("not xml"), "")


if __name__ == "__main__":
    unittest.main()
