"""Tests for PaperBanana image retry helpers."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_wrapper():
    path = Path(__file__).resolve().parents[1] / "scripts" / "paperbanana_wrapper.py"
    spec = importlib.util.spec_from_file_location("paperbanana_wrapper", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PaperBananaImageRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_wrapper()

    def test_retriable_http_codes(self) -> None:
        self.assertTrue(self.mod._is_retriable_image_error("", http_code=403))
        self.assertTrue(self.mod._is_retriable_image_error("", http_code=429))
        self.assertFalse(self.mod._is_retriable_image_error("", http_code=400))

    def test_retriable_balance_message(self) -> None:
        msg = '{"error":"insufficient balance"}'
        self.assertTrue(self.mod._is_retriable_image_error(msg))

    def test_retry_sleep_grows(self) -> None:
        first = self.mod._image_retry_sleep_seconds(1)
        second = self.mod._image_retry_sleep_seconds(2)
        self.assertGreater(second, first)


if __name__ == "__main__":
    unittest.main()
