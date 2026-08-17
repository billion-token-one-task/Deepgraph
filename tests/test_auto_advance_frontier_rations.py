"""A ration spent against a dead route must not retire the problem forever.

auto_advance rations frontier bootstrap attempts per research problem. On
2026-08-10 an unreachable evaluator route consumed 28 of those rations; the
route was marked transient the same day and reconfigured on 2026-08-17, but the
counter lives in the state file and nothing reset it. Seventeen problems across
six agendas stayed at tries=4, so every later pass logged
frontier_attempts_exhausted and no agenda could obtain a frontier packet again.

The state file already had the right mechanism for this -- an epoch reset whose
stated purpose is that "a deployed repair must get one fresh autonomous
attempt" -- it just did not cover frontier_attempts.
"""

import json
import tempfile
import unittest
from pathlib import Path

from scripts import auto_advance


class FrontierRationEpochTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "state.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, payload):
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def test_a_stale_epoch_clears_exhausted_frontier_rations(self):
        self._write(
            {
                "recycle_epoch": "an-older-epoch",
                "recycles": {"11:105": 3},
                "frontier_attempts": {"1:37": 4, "2:39": 4, "10:3": 4},
                "frontier_issues": {"1:37": 4, "2:39": 4, "10:3": 4},
            }
        )
        state = auto_advance._load_state(self.path)
        self.assertEqual(state["frontier_attempts"], {})
        self.assertEqual(state["recycles"], {})
        self.assertEqual(state["recycle_epoch"], auto_advance.RECYCLE_EPOCH)

    def test_the_idempotency_serial_survives_the_reset(self):
        """Burned authority keys can never be reused, so this must not restart."""
        self._write(
            {
                "recycle_epoch": "an-older-epoch",
                "frontier_attempts": {"1:37": 4},
                "frontier_issues": {"1:37": 4},
            }
        )
        state = auto_advance._load_state(self.path)
        self.assertEqual(state["frontier_issues"], {"1:37": 4})

    def test_a_current_epoch_leaves_rations_alone(self):
        """Within one epoch the ration must still bound a genuinely bad problem."""
        self._write(
            {
                "recycle_epoch": auto_advance.RECYCLE_EPOCH,
                "recycles": {"11:105": 2},
                "frontier_attempts": {"1:37": 4},
            }
        )
        state = auto_advance._load_state(self.path)
        self.assertEqual(state["frontier_attempts"], {"1:37": 4})
        self.assertEqual(state["recycles"], {"11:105": 2})

    def test_a_reset_problem_is_retryable_again(self):
        """The point of the reset: tries drops below the cap."""
        self._write(
            {"recycle_epoch": "an-older-epoch", "frontier_attempts": {"1:37": 4}}
        )
        state = auto_advance._load_state(self.path)
        tries = int(state["frontier_attempts"].get("1:37", 0))
        self.assertLess(tries, auto_advance.FRONTIER_MAX_TRIES)

    def test_the_epoch_names_a_deployed_repair(self):
        """The file's own rule: every bump must name a repair already shipped."""
        self.assertRegex(auto_advance.RECYCLE_EPOCH, r"\d{4}-\d{2}-\d{2}$")


if __name__ == "__main__":
    unittest.main()
