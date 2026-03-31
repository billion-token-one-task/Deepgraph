import unittest
from unittest.mock import patch

from web.app import app


class WebAppTriageTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_includes_triage_tab(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"data-tab=\"triage\"", response.data)
        self.assertIn(b"Triage Queue", response.data)

    def test_taxonomy_node_includes_triage_queue(self):
        with patch("web.app.tax.get_node", return_value={"id": "ml.cv", "name": "Computer Vision"}), \
             patch("web.app.tax.get_children", return_value=[]), \
             patch("web.app.tax.get_breadcrumb", return_value=[]), \
             patch("web.app.tax.get_node_papers", return_value=[]), \
             patch("web.app.tax.get_node_paper_clusters", return_value=[]), \
             patch("web.app.tax.get_subfield_intersection_matrix", return_value={}), \
             patch("web.app.tax.get_method_dataset_matrix", return_value={"methods": [], "datasets": [], "metrics": [], "cells": {}}), \
             patch("web.app.tax.get_node_gaps", return_value=[]), \
             patch("web.app.opp.get_node_opportunities", return_value=[]), \
             patch("web.app.opp.get_opportunity_triage", return_value=[{"opportunity_id": 1, "priority_score": 4.8, "priority_band": "high"}]), \
             patch("web.app.tax.get_node_summary", return_value={}), \
             patch("web.app.graph.get_node_graph_summary", return_value={}):
            response = self.client.get("/api/taxonomy/ml.cv")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["triage_queue"][0]["priority_band"], "high")



if __name__ == "__main__":
    unittest.main()
