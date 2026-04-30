import sqlite3
import unittest
from pathlib import Path


class DatabaseSchemaTests(unittest.TestCase):
    def test_legacy_insights_table_is_created(self):
        schema = Path("db/schema.sql").read_text(encoding="utf-8")
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)

        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(insights)").fetchall()
        }

        self.assertIn("id", cols)
        self.assertIn("node_id", cols)
        self.assertIn("insight_type", cols)
        self.assertIn("title", cols)
        self.assertIn("hypothesis", cols)
        self.assertIn("experiment", cols)
        self.assertIn("novelty_score", cols)
        self.assertIn("feasibility_score", cols)
        self.assertIn("paradigm_score", cols)
        self.assertIn("rank_rationale", cols)

    def test_patterns_table_supports_abstraction_pipeline(self):
        schema = Path("db/schema.sql").read_text(encoding="utf-8")
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)

        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(patterns)").fetchall()
        }

        self.assertIn("node_id", cols)
        self.assertIn("abstraction_level", cols)
        self.assertIn("source_claims", cols)


if __name__ == "__main__":
    unittest.main()
