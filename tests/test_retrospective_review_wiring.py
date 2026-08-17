"""The retrospective decision row must name the audit row it rests on.

scientific_decision_records.evidence_audit_record_id is NOT NULL. The
retrospective writer originally discarded the id returned by its own
evidence_audit_records insert, so every apply_review() on PostgreSQL failed
with NotNullViolation and the legacy path had never once completed.
"""

import inspect
import re
import unittest

from meta_harness import retrospective_review as rr


class RetrospectiveWiringTests(unittest.TestCase):
    def setUp(self):
        self.source = inspect.getsource(rr.apply_review)

    def test_audit_insert_returns_its_id(self):
        self.assertIn("insert_returning_id", self.source)
        audit = re.search(
            r"INSERT INTO evidence_audit_records.*?\"\"\"", self.source, re.S
        )
        self.assertIsNotNone(audit, "audit insert not found")
        self.assertIn("RETURNING id", audit.group(0))

    def test_decision_row_carries_the_audit_id(self):
        decision = re.search(
            r"INSERT INTO scientific_decision_records.*?\"\"\"", self.source, re.S
        )
        self.assertIsNotNone(decision, "decision insert not found")
        self.assertIn("evidence_audit_record_id", decision.group(0))
        self.assertIn("audit_record_id", self.source)

    def test_placeholders_match_the_decision_column_count(self):
        decision = re.search(
            r"INSERT INTO scientific_decision_records\s*\((?P<cols>.*?)\)\s*"
            r"VALUES \((?P<vals>.*?)\)",
            self.source,
            re.S,
        )
        self.assertIsNotNone(decision)
        columns = [c.strip() for c in decision.group("cols").split(",") if c.strip()]
        placeholders = [v.strip() for v in decision.group("vals").split(",") if v.strip()]
        self.assertEqual(len(columns), len(placeholders))

    def test_verdict_stays_capped_at_inconclusive(self):
        with self.assertRaises(rr.RetrospectiveReviewError):
            rr.apply_review(run_id=1, approval={}, verdict="supported")


if __name__ == "__main__":
    unittest.main()
