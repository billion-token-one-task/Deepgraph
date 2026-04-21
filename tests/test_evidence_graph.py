import unittest

from db.evidence_graph import (
    build_structured_graph_payload_from_records,
    canonicalize_entity_name,
    get_merge_candidate_context,
    list_merge_candidates_with_context,
    make_entity_id,
    merge_graph_payloads,
    normalize_entity_name,
    normalize_entity_type,
    normalize_predicate,
    score_entity_merge_candidate,
    summarize_graph_rows,
)


class EvidenceGraphNormalizationTests(unittest.TestCase):
    def test_normalize_entity_name(self):
        self.assertEqual(normalize_entity_name("Protein Folding"), "protein_folding")
        self.assertEqual(normalize_entity_name("  WMT14 En-De  "), "wmt14_en_de")

    def test_make_entity_id(self):
        self.assertEqual(make_entity_id("method", "Transformer XL"), "method:transformer_xl")
        self.assertEqual(make_entity_id("unknown-type", "Graph"), "concept:graph")

    def test_canonicalize_method_name(self):
        self.assertEqual(canonicalize_entity_name("method", "SemTok (Ours)"), "SemTok")
        self.assertEqual(canonicalize_entity_name("artifact", "GPT-4o*"), "GPT-4o")

    def test_normalize_predicate(self):
        self.assertEqual(normalize_predicate("evaluated_on"), "evaluated_on")
        self.assertEqual(normalize_predicate("unsupported relation"), "related_to")

    def test_score_entity_merge_candidate_matches_ours_suffix(self):
        entity_a = {"canonical_name": "SemTok", "entity_type": "method", "aliases": '[]'}
        entity_b = {"canonical_name": "SemTok (Ours)", "entity_type": "method", "aliases": '[]'}
        score, rationale = score_entity_merge_candidate(entity_a, entity_b)
        self.assertGreaterEqual(score, 0.99)
        self.assertIn("match", rationale.lower())

    def test_score_entity_merge_candidate_rejects_different_names(self):
        entity_a = {"canonical_name": "Transformer", "entity_type": "method", "aliases": '[]'}
        entity_b = {"canonical_name": "ResNet", "entity_type": "method", "aliases": '[]'}
        self.assertIsNone(score_entity_merge_candidate(entity_a, entity_b))


class EvidenceGraphSummaryTests(unittest.TestCase):
    def test_summarize_graph_rows(self):
        entity_rows = [
            {"canonical_name": "Transformer", "entity_type": "method", "paper_id": "p1", "mention_count": 2},
            {"canonical_name": "Transformer", "entity_type": "method", "paper_id": "p2", "mention_count": 1},
            {"canonical_name": "WMT14 En-De", "entity_type": "dataset", "paper_id": "p1", "mention_count": 1},
        ]
        relation_rows = [
            {"subject_name": "Transformer", "predicate": "evaluated_on", "object_name": "WMT14 En-De", "paper_id": "p1", "relation_count": 1},
            {"subject_name": "Transformer", "predicate": "evaluated_on", "object_name": "WMT14 En-De", "paper_id": "p2", "relation_count": 1},
        ]

        summary = summarize_graph_rows(entity_rows, relation_rows)

        self.assertEqual(summary["entity_count"], 2)
        self.assertEqual(summary["relation_count"], 1)
        self.assertEqual(summary["top_entities"][0]["name"], "Transformer")
        self.assertEqual(summary["top_entities"][0]["paper_count"], 2)
        self.assertEqual(summary["top_relations"][0]["predicate"], "evaluated_on")
        self.assertEqual(summary["top_relations"][0]["paper_count"], 2)

    def test_structured_graph_payload_builds_relations(self):
        payload = build_structured_graph_payload_from_records(
            methods=[{
                "name": "SemTok (Ours)",
                "description": "Tokenizer",
                "builds_on": ["VQ-VAE"],
                "first_paper_id": "p1",
            }],
            results=[{
                "method_name": "SemTok (Ours)",
                "dataset_name": "ImageNet-1K",
                "metric_name": "FID",
                "evidence_location": "Table 1",
            }],
            claims=[{
                "claim_text": "SemTok is evaluated on ImageNet-1K with FID.",
                "claim_type": "performance",
                "method_name": "SemTok (Ours)",
                "dataset_name": "ImageNet-1K",
                "metric_name": "FID",
                "evidence_location": "Table 1",
            }],
            insight={"work_type": "model", "key_findings": [], "limitations": [], "open_questions": []},
        )

        entity_names = {item["name"] for item in payload["entities"]}
        relation_keys = {(item["subject"], item["predicate"], item["object"]) for item in payload["relations"]}

        self.assertIn("SemTok", entity_names)
        self.assertIn("ImageNet-1K", entity_names)
        self.assertIn(("SemTok", "evaluated_on", "ImageNet-1K"), relation_keys)
        self.assertIn(("SemTok", "builds_on", "VQ-VAE"), relation_keys)

    def test_merge_graph_payloads_deduplicates_entities(self):
        merged = merge_graph_payloads(
            {"entities": [{"name": "SemTok (Ours)", "entity_type": "method", "aliases": [], "confidence": 0.9}], "relations": []},
            {"entities": [{"name": "SemTok", "entity_type": "method", "aliases": ["SemTok (Ours)"], "confidence": 0.8}], "relations": []},
        )
        self.assertEqual(len(merged["entities"]), 1)
        self.assertEqual(merged["entities"][0]["name"], "SemTok")

    def test_merge_candidate_context_helpers_exist(self):
        self.assertIsInstance(list_merge_candidates_with_context(limit=0), list)
        self.assertIsNone(get_merge_candidate_context(-1))


if __name__ == "__main__":
    unittest.main()
