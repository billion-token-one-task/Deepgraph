"""The normal pytest entry point must not inherit a database URL."""

from __future__ import annotations

import os
import unittest


class TestDatabaseIsolationContract(unittest.TestCase):
    def test_candidate_test_entry_forces_empty_database_url(self):
        self.assertEqual(os.environ.get("DEEPGRAPH_DATABASE_URL", "").strip(), "")

