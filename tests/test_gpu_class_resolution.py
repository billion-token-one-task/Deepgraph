"""The GPU class a grant asks for must come from hardware that exists.

The requested class decides which backend a grant may draw on. The default
named "NVIDIA A100-PCIE-40GB" long after that fleet stopped answering, and a
lane pinned to "Tesla T4" sent three runs to Colab and recorded three invalid
outcomes while a registered, verified A10G sat idle. A GPU fleet changes; a
constant in an argument parser does not.
"""

import unittest
from unittest import mock

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "auto_advance_under_test",
    Path(__file__).resolve().parent.parent / "scripts" / "auto_advance.py")


class _FakeDb:
    def __init__(self, rows=None, raises=False):
        self.rows = rows or []
        self.raises = raises

    def fetchall(self, sql, params=()):
        if self.raises:
            raise RuntimeError("registry unavailable")
        return self.rows


class GpuClassResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(cls.module)
        except Exception as exc:                     # needs the full runtime env
            raise unittest.SkipTest("auto_advance is not importable here: %s" % exc)

    def _resolve(self, requested, rows=None, raises=False):
        with mock.patch.object(self.module, "db", _FakeDb(rows, raises)):
            return self.module.resolve_gpu_class(requested)

    def test_an_explicit_class_is_never_overridden(self):
        rows = [{"gpu_model": "NVIDIA A10G", "total_mem_gb": 22.0}]
        self.assertEqual(self._resolve("NVIDIA H100", rows), "NVIDIA H100")

    def test_it_takes_the_registered_card(self):
        rows = [{"gpu_model": "NVIDIA A10G", "total_mem_gb": 22.0}]
        self.assertEqual(self._resolve(None, rows), "NVIDIA A10G")

    def test_the_widest_card_wins_when_several_are_registered(self):
        rows = [{"gpu_model": "NVIDIA L40S", "total_mem_gb": 46.0},
                {"gpu_model": "NVIDIA A10G", "total_mem_gb": 22.0}]
        self.assertEqual(self._resolve(None, rows), "NVIDIA L40S")

    def test_an_empty_registry_asks_for_nothing_rather_than_inventing_hardware(self):
        self.assertEqual(self._resolve(None, []), "none")

    def test_a_registry_error_does_not_invent_hardware_either(self):
        self.assertEqual(self._resolve(None, raises=True), "none")

    def test_blank_model_names_are_skipped(self):
        rows = [{"gpu_model": "   ", "total_mem_gb": 80.0},
                {"gpu_model": "NVIDIA A10G", "total_mem_gb": 22.0}]
        self.assertEqual(self._resolve(None, rows), "NVIDIA A10G")


if __name__ == "__main__":
    unittest.main()
