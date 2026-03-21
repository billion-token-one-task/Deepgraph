import unittest

from agents.domain_summary_agent import fallback_domain_summary
from db.taxonomy import _intersection_strength, _metric_prefers_lower, cluster_papers_from_signals


class MetricPreferenceTests(unittest.TestCase):
    def test_lower_is_better_metrics(self):
        self.assertTrue(_metric_prefers_lower("WER"))
        self.assertTrue(_metric_prefers_lower("gFID"))
        self.assertTrue(_metric_prefers_lower("relative absolute error"))

    def test_higher_is_better_metrics(self):
        self.assertFalse(_metric_prefers_lower("accuracy"))
        self.assertFalse(_metric_prefers_lower("F1"))
        self.assertFalse(_metric_prefers_lower("Recall"))

    def test_intersection_strength_weights_papers_more_than_entities(self):
        low = _intersection_strength(0, 4)
        high = _intersection_strength(2, 1)
        self.assertGreater(high, low)


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


class PaperClusterTests(unittest.TestCase):
    def test_cluster_papers_from_shared_entities(self):
        papers = [
            {"id": "p1", "title": "Paper 1"},
            {"id": "p2", "title": "Paper 2"},
            {"id": "p3", "title": "Paper 3"},
            {"id": "p4", "title": "Paper 4"},
        ]
        paper_entities = {
            "p1": {"e1", "e2"},
            "p2": {"e1", "e2", "e3"},
            "p3": {"e8"},
            "p4": {"e8", "e9"},
        }
        work_types = {"p1": "model", "p2": "model", "p3": "benchmark", "p4": "benchmark"}
        entity_names = {
            "p1": {"e1": "Transformer", "e2": "ImageNet"},
            "p2": {"e1": "Transformer", "e2": "ImageNet", "e3": "FID"},
            "p3": {"e8": "Animal Re-ID"},
            "p4": {"e8": "Animal Re-ID", "e9": "Cross-Domain Re-ID"},
        }

        clusters = cluster_papers_from_signals(
            papers,
            paper_entities,
            work_types,
            entity_names,
            min_shared_entities=1,
            min_papers_to_cluster=4,
        )

        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0]["paper_count"], 2)


if __name__ == "__main__":
    unittest.main()
