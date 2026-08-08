import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import auto_advance


class AutoAdvanceGuardTests(unittest.TestCase):
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
                return {"s": 113047}
            if "FROM resource_grants" in sql:
                return {"s": 40000}
            raise AssertionError(sql)

        with mock.patch.object(auto_advance.db, "fetchone", side_effect=fetchone):
            spent = auto_advance._spent_delta(state, [11])

        self.assertEqual(spent, 118985)


if __name__ == "__main__":
    unittest.main()
