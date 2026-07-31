"""Read-only scientific authority policy tests."""

from __future__ import annotations

import unittest
from unittest import mock

from meta_harness.scientific_authority import positive_decision_authorized


class ScientificAuthorityTests(unittest.TestCase):
    def test_legacy_confirmed_state_without_decision_record_is_rejected(self):
        with mock.patch(
            "meta_harness.scientific_authority.db.fetchone",
            side_effect=[
                {
                    "agenda_id": 11,
                    "scientific_evidence_state": "scientifically_decided",
                },
                None,
            ],
        ):
            self.assertFalse(
                positive_decision_authorized(agenda_id=11, run_id=31)
            )

    def test_scoped_supported_decision_record_is_authoritative(self):
        with mock.patch(
            "meta_harness.scientific_authority.db.fetchone",
            side_effect=[
                {
                    "agenda_id": 11,
                    "scientific_evidence_state": "scientifically_decided",
                },
                {
                    "verdict": "supported",
                    "verdict_hash": "a" * 64,
                    "evidence_audit_record_id": 41,
                },
            ],
        ):
            self.assertTrue(
                positive_decision_authorized(agenda_id=11, run_id=31)
            )

    def test_cross_agenda_run_is_rejected_before_decision_lookup(self):
        fetch = mock.Mock(
            return_value={
                "agenda_id": 12,
                "scientific_evidence_state": "scientifically_decided",
            }
        )
        with mock.patch(
            "meta_harness.scientific_authority.db.fetchone",
            fetch,
        ):
            self.assertFalse(
                positive_decision_authorized(agenda_id=11, run_id=31)
            )
        self.assertEqual(fetch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
