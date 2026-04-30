import tempfile
import unittest
from pathlib import Path

from agents import research_bridge
from db import database


class ResearchBridgeDeepInsightsTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test.db"
        self.old_db_path = database.DB_PATH
        self.old_database_url = database.DATABASE_URL
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(database._local, attr):
                try:
                    getattr(database._local, attr).close()
                except Exception:
                    pass
                setattr(database._local, attr, None)
        database.DATABASE_URL = ""
        database.DB_PATH = self.db_path
        database.init_db()

        database.execute("INSERT INTO taxonomy_nodes (id, name, depth) VALUES ('ml.test', 'Test Node', 1)")
        database.execute(
            """
            INSERT INTO deep_insights
            (id, tier, status, title, problem_statement, proposed_method, experimental_plan,
             evidence_summary, mechanism_type, signal_mix, supporting_papers, source_node_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                7,
                2,
                "candidate",
                "Bridgeable Deep Insight",
                "Hypothesis about a more stable training recipe.",
                '{"name": "StableRecipe", "definition": "Use adaptive confidence gating."}',
                '{"datasets": ["ToySet"], "metrics": {"primary": "accuracy"}}',
                "Evidence summary from deep_insights.",
                "mechanism_mismatch",
                '["mechanism_mismatch", "plateau"]',
                '["p1"]',
                '["ml.test"]',
            ),
        )
        database.execute(
            "INSERT INTO papers (id, title, abstract, status) VALUES ('p1', 'Paper 1', 'Abstract', 'reasoned')"
        )
        database.execute("INSERT INTO paper_taxonomy (paper_id, node_id) VALUES ('p1', 'ml.test')")
        database.execute(
            """
            INSERT INTO claims (paper_id, claim_text, metric_name, metric_value)
            VALUES ('p1', 'Improves accuracy on ToySet.', 'accuracy', 0.91)
            """
        )
        database.execute(
            """
            INSERT INTO results (paper_id, method_name, dataset_name, metric_name, metric_value)
            VALUES ('p1', 'StableRecipe', 'ToySet', 'accuracy', 0.91)
            """
        )
        database.commit()

    def tearDown(self):
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(database._local, attr):
                try:
                    getattr(database._local, attr).close()
                except Exception:
                    pass
                setattr(database._local, attr, None)
        database.DATABASE_URL = self.old_database_url
        database.DB_PATH = self.old_db_path
        self.tmpdir.cleanup()

    def test_gather_context_reads_deep_insights_primary_record(self):
        ctx = research_bridge.gather_context(7)

        self.assertEqual(ctx["insight"]["title"], "Bridgeable Deep Insight")
        self.assertEqual(ctx["insight"]["mechanism_type"], "mechanism_mismatch")
        self.assertEqual(ctx["insight"]["focus_node_ids"], ["ml.test"])
        self.assertEqual(len(ctx["papers"]), 1)
        self.assertEqual(ctx["papers"][0]["id"], "p1")
        self.assertEqual(len(ctx["results_sample"]), 1)

    def test_format_proposal_handles_deep_insight_fields(self):
        ctx = research_bridge.gather_context(7)

        proposal = research_bridge.format_proposal(ctx)

        self.assertIn("Bridgeable Deep Insight", proposal)
        self.assertIn("StableRecipe", proposal)
        self.assertIn("mechanism mismatch", proposal.lower())
        self.assertIn("ToySet", proposal)


if __name__ == "__main__":
    unittest.main()
