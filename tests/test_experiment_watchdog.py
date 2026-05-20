"""Tests for experiment watchdog helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from agents.metric_parser import build_benchmark_summary_from_predictions


class ExperimentWatchdogTests(unittest.TestCase):
    def test_build_benchmark_summary_from_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            pred = results / "raw_predictions.jsonl"
            rows = [
                {"method": "CPG", "dataset": "GSM8K", "primary_score": 1.0, "seed": 0},
                {"method": "CPG", "dataset": "GSM8K", "primary_score": 0.0, "seed": 0},
                {"method": "baseline", "dataset": "GSM8K", "primary_score": 0.5, "seed": 0},
            ]
            pred.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            summary = build_benchmark_summary_from_predictions(
                results,
                candidate_method="CPG",
                min_lines=1,
            )
            self.assertTrue(summary.get("partial_from_predictions"))
            self.assertIn("CPG", summary.get("per_method", {}))
            self.assertAlmostEqual(summary["per_method"]["CPG"]["primary_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
