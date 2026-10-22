"""Tests for cite-and-verify grounding helpers."""
import unittest

from agents.claim_grounding import (
    apply_claim_grounding,
    locate_quote_span,
)


class TestLocateQuote(unittest.TestCase):
    def test_verbatim(self):
        full = "We introduce YOLOv9 which reaches 55.6 mAP on COCO val2017."
        q = "55.6 mAP on COCO val2017"
        span = locate_quote_span(full, q)
        self.assertIsNotNone(span)
        self.assertEqual(full[span[0] : span[1]], q)

    def test_flexible_whitespace(self):
        full = "Our model achieves\n55.6%\nmAP on the validation set."
        q = "Our model achieves 55.6% mAP on the validation set."
        span = locate_quote_span(full, q)
        self.assertIsNotNone(span)

    def test_too_short_quote(self):
        full = "Hello world"
        self.assertIsNone(locate_quote_span(full, "short"))


class TestApplyClaim(unittest.TestCase):
    def test_applies_fields(self):
        full = "Experiments show BLEU of 41.2 on the test set for our model."
        claim = {
            "claim_text": "BLEU 41.2",
            "source_quote": "BLEU of 41.2 on the test set",
        }
        apply_claim_grounding(claim, full)
        self.assertEqual(claim["grounding_status"], "verified")
        self.assertEqual(claim["grounding_score"], 1.0)
        self.assertIsNotNone(claim["char_start"])

    def test_finds_quote_in_appendix(self):
        claim = {
            "claim_text": "Appendix ablation improves accuracy to 94.1",
            "source_quote": "accuracy reaches 94.1 on CIFAR-10",
        }
        apply_claim_grounding(
            claim,
            "Main body text without the exact ablation number.",
            appendix_text="Appendix A\nDetailed ablations show accuracy reaches 94.1 on CIFAR-10.",
        )
        self.assertEqual(claim["grounding_status"], "verified")
        self.assertEqual(claim["grounding_source_field"], "appendix_text")
        self.assertIsNotNone(claim["char_start"])


if __name__ == "__main__":
    unittest.main()
