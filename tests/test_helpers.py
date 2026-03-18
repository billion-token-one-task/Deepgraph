import unittest

from agents.domain_summary_agent import fallback_domain_summary
from db.taxonomy import _metric_prefers_lower


class MetricPreferenceTests(unittest.TestCase):
    def test_lower_is_better_metrics(self):
        self.assertTrue(_metric_prefers_lower("WER"))
        self.assertTrue(_metric_prefers_lower("gFID"))
        self.assertTrue(_metric_prefers_lower("relative absolute error"))

    def test_higher_is_better_metrics(self):
        self.assertFalse(_metric_prefers_lower("accuracy"))
        self.assertFalse(_metric_prefers_lower("F1"))
        self.assertFalse(_metric_prefers_lower("Recall"))


class FallbackSummaryTests(unittest.TestCase):
    def test_fallback_uses_limitations_as_gaps(self):
        node = {"id": "ml.dl.cv", "name": "Computer Vision", "description": "Visual understanding with deep learning"}
        snapshot = {
            "children": [],
            "paper_count": 5,
            "result_count": 20,
            "work_types": [{"work_type": "benchmark", "count": 2}],
            "methods": [{"name": "Model-A", "paper_count": 3}],
            "datasets": [{"name": "Dataset-A", "paper_count": 4}],
            "limitations": ["Models break under lighting changes."],
            "open_questions": ["How well do they generalize outdoors?"],
        }

        summary = fallback_domain_summary(node, snapshot)

        self.assertIn("overview", summary)
        self.assertEqual(summary["what_people_are_building"][0]["label"], "Benchmark")
        self.assertEqual(summary["current_gaps"][0]["description"], "Models break under lighting changes.")
        self.assertIn("How well do they generalize outdoors?", summary["starter_questions"])


if __name__ == "__main__":
    unittest.main()
