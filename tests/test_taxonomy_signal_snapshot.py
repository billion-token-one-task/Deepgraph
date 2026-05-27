"""Taxonomy signal snapshot normalization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from db import database as db
from db import taxonomy as tax


class TaxonomySignalSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.old_db_path = db.DB_PATH
        self.old_database_url = db.DATABASE_URL
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(db._local, attr):
                try:
                    getattr(db._local, attr).close()
                except Exception:
                    pass
                setattr(db._local, attr, None)
        db.DATABASE_URL = ""
        db.DB_PATH = Path(self.tmpdir.name) / "test.db"
        db.init_db()

    def tearDown(self):
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(db._local, attr):
                try:
                    getattr(db._local, attr).close()
                except Exception:
                    pass
                setattr(db._local, attr, None)
        db.DATABASE_URL = self.old_database_url
        db.DB_PATH = self.old_db_path
        self.tmpdir.cleanup()

    def test_flatten_text_items_handles_nested_open_questions(self):
        nested = ["Outer question", ["Nested inner question"]]
        flat = tax._flatten_text_items(nested)
        self.assertEqual(flat, ["Outer question", "Nested inner question"])

    def test_get_node_signal_snapshot_accepts_nested_open_questions(self):
        db.execute("INSERT INTO taxonomy_nodes (id, name, depth) VALUES ('ml.test', 'Test', 1)")
        db.execute("INSERT INTO papers (id, title, status) VALUES ('p1', 'Paper', 'reasoned')")
        db.execute("INSERT INTO paper_taxonomy (paper_id, node_id) VALUES ('p1', 'ml.test')")
        db.execute(
            """
            INSERT INTO paper_insights (paper_id, limitations, open_questions)
            VALUES (?, ?, ?)
            """,
            (
                "p1",
                json.dumps(["Limitation text"]),
                json.dumps([["Nested open question"]]),
            ),
        )
        db.commit()

        snapshot = tax.get_node_signal_snapshot("ml.test")
        self.assertIn("Nested open question", snapshot["open_questions"])


if __name__ == "__main__":
    unittest.main()
