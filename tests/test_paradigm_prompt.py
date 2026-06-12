"""_build_structure_prompt must handle DB rows from both backends.

Postgres returns datetime objects in signal rows where SQLite returns
strings; prompt building must not crash on either."""

from __future__ import annotations

import os
import unittest
from datetime import datetime

os.environ["DEEPGRAPH_DATABASE_URL"] = ""  # force SQLite tmpdir; never touch a real DB from the environment


class StructurePromptTest(unittest.TestCase):
    def test_rows_with_datetime_values(self):
        from agents.paradigm_agent import _build_structure_prompt

        signals = {
            "entity_overlaps": [],
            "pattern_matches": [],
            "contradiction_clusters": [],
            "taxonomy_map": [],
            "hidden_variable_bridges": [
                {
                    "entity": "gaussian splatting",
                    "node_a": "cv.3d",
                    "node_b": "cv.sfm",
                    "created_at": datetime(2026, 6, 12, 12, 0, 0),
                }
            ],
        }
        prompt = _build_structure_prompt(signals)
        self.assertIn("gaussian splatting", prompt)
        self.assertIn("2026-06-12", prompt)


if __name__ == "__main__":
    unittest.main()
