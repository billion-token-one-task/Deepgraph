import json
import unittest
from unittest.mock import patch

from agents.knowledge_loop import cascade_from_claim


class FakeKnowledgeDb:
    def __init__(self):
        self.claim = {
            "id": 1,
            "run_id": 10,
            "deep_insight_id": 20,
            "verdict": "confirmed",
            "effect_size": 0.2,
            "cascaded": 0,
        }
        self.insight = {
            "id": 20,
            "title": "Unknown node insight",
            "source_node_ids": json.dumps(["missing.node"]),
            "supporting_papers": "[]",
        }
        self.inserted_opportunities = []

    def fetchone(self, sql, params=()):
        if "FROM experimental_claims WHERE id" in sql:
            return self.claim
        if "FROM deep_insights WHERE id" in sql:
            return self.insight
        if "FROM taxonomy_nodes WHERE id" in sql:
            return None
        return None

    def fetchall(self, sql, params=()):
        if "FROM deep_insights" in sql:
            return []
        if "FROM node_opportunities" in sql:
            return []
        return []

    def execute(self, sql, params=()):
        stripped = sql.strip()
        if stripped.startswith("INSERT INTO node_opportunities"):
            self.inserted_opportunities.append(params)
        elif stripped.startswith("UPDATE experimental_claims"):
            self.claim["cascaded"] = 1
        return None

    def commit(self):
        return None


class KnowledgeLoopTests(unittest.TestCase):
    def test_cascade_skips_unknown_source_nodes(self):
        fake_db = FakeKnowledgeDb()

        with patch("agents.knowledge_loop.db", fake_db):
            cascade_from_claim(1)

        self.assertEqual(fake_db.inserted_opportunities, [])
        self.assertEqual(fake_db.claim["cascaded"], 1)


if __name__ == "__main__":
    unittest.main()
