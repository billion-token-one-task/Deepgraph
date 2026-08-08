import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import auto_advance


class AutoAdvanceGuardTests(unittest.TestCase):
    def test_gpu_failure_is_recyclable(self):
        self.assertIn(("failed", "gpu_failed"), auto_advance.DEAD_END)

    def test_recycle_reuses_live_grant_without_reserving_another_cap(self):
        job = {
            "id": 99,
            "deep_insight_id": 105,
            "status": "failed",
            "stage": "gpu_failed",
            "resource_grant_id": 17,
            "last_error": "reproduction failure",
            "token_cap": 40000,
            "grant_status": "active",
            "grant_live": True,
        }
        args = mock.Mock(
            agenda=[10, 11],
            grant_token_cap=40000,
            spend_limit=120000,
        )
        journal = mock.Mock()
        state = {"recycles": {"105": 1}}

        with (
            mock.patch.object(auto_advance, "_rows", return_value=[job]),
            mock.patch.object(
                auto_advance,
                "_spent_delta",
                side_effect=AssertionError("must not reserve a second grant"),
            ),
            mock.patch.object(auto_advance, "_requeue_for_consumer") as requeue,
        ):
            auto_advance.recycle_stranded(11, state, journal, args)

        requeue.assert_called_once_with(11, 105, 17, journal, args, 2)
        self.assertEqual(state["recycles"]["105"], 2)

    def test_deployed_recycle_epoch_resets_old_operational_retry_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"
            path.write_text(
                json.dumps(
                    {
                        "spend_baseline": {"11": 65879},
                        "frontier_packets": {"11": 3},
                        "recycles": {"105": 3},
                        "recycle_epoch": "old-code",
                    }
                ),
                encoding="utf-8",
            )

            state = auto_advance._load_state(path)

        self.assertEqual(state["recycles"], {})
        self.assertEqual(state["recycle_epoch"], auto_advance.RECYCLE_EPOCH)
        self.assertEqual(state["spend_baseline"], {"11": 65879})

    def test_spend_guard_counts_expired_metered_usage_and_live_grant_cap(self):
        state = {"spend_baseline": {"11": 65879}}

        def fetchone(sql, params=()):
            if "FROM research_agendas" in sql:
                return {"s": 31817}
            if "resource_grant_usage_reservations" in sql:
                self.assertIn("NOT (g.status='active'", sql)
                return {"s": 113047}
            if "FROM resource_grants" in sql:
                return {"s": 40000}
            raise AssertionError(sql)

        with mock.patch.object(auto_advance.db, "fetchone", side_effect=fetchone):
            spent = auto_advance._spent_delta(state, [11])

        self.assertEqual(spent, 118985)


if __name__ == "__main__":
    unittest.main()
