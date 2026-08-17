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



class AuthoritySerialTests(unittest.TestCase):
    """The database, not the state file, decides which keys are already spent.

    On 2026-08-17 the file counter restarted at 0 while the authorities table
    still held t1..t4 for the same problems, revoked since 2026-08-10. issue()
    replayed the old row and every agenda failed with
    authority_expired,authority_revoked.
    """

    def test_the_serial_is_read_from_the_burned_keys(self):
        from unittest import mock

        rows = [
            {"idempotency_key": "auto-advance-v1:agenda1:problem37:t1"},
            {"idempotency_key": "auto-advance-v1:agenda1:problem37:t4"},
            {"idempotency_key": "auto-advance-v1:agenda1:problem37:t2"},
        ]
        with mock.patch.object(auto_advance, "_rows", return_value=rows):
            self.assertEqual(auto_advance._burned_authority_serial(1, 37), 4)

    def test_keys_from_another_issuer_are_ignored(self):
        from unittest import mock

        rows = [
            {"idempotency_key": "some-other-tool:agenda1:problem37:t9"},
            {"idempotency_key": "auto-advance-v1:agenda1:problem37:t2"},
        ]
        with mock.patch.object(auto_advance, "_rows", return_value=rows):
            self.assertEqual(auto_advance._burned_authority_serial(1, 37), 2)

    def test_a_malformed_suffix_does_not_crash_the_pass(self):
        from unittest import mock

        rows = [{"idempotency_key": "auto-advance-v1:agenda1:problem37:tXX"}]
        with mock.patch.object(auto_advance, "_rows", return_value=rows):
            self.assertEqual(auto_advance._burned_authority_serial(1, 37), 0)

    def test_no_prior_authority_starts_at_zero(self):
        from unittest import mock

        with mock.patch.object(auto_advance, "_rows", return_value=[]):
            self.assertEqual(auto_advance._burned_authority_serial(1, 37), 0)



class RoleDefaultTests(unittest.TestCase):
    """A role's deployed identity has one home: the role-route env refs.

    The advancer used to hard-code sora2_claude/claude-opus-4-6-thinking. That
    provider stopped being declared, and every frontier bootstrap on 2026-08-17
    failed with "frontier evaluator route is unavailable" even though the
    configured evaluator was answering normally.
    """

    def test_defaults_follow_the_deployed_route(self):
        import os
        from unittest import mock

        env = {
            "DEEPGRAPH_LLM_EVALUATOR": "novita_deepseek",
            "DEEPGRAPH_LLM_EVALUATOR_MODEL": "deepseek/deepseek-v4-flash-0731",
            "DEEPGRAPH_LLM_EVALUATOR_FAMILY": "deepseek",
            "DEEPGRAPH_LLM_PRIMARY": "sora2_gemini",
            "DEEPGRAPH_LLM_PRIMARY_FAMILY": "gemini-flash",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(auto_advance._role_default("EVALUATOR", ""), "novita_deepseek")
            self.assertEqual(
                auto_advance._role_default("EVALUATOR", "_MODEL"),
                "deepseek/deepseek-v4-flash-0731",
            )
            self.assertEqual(auto_advance._role_default("PRIMARY", "_FAMILY"), "gemini-flash")

    def test_evaluator_and_proposer_families_differ_in_the_deployed_config(self):
        """The bootstrap refuses an evaluator that shares the proposer's family."""
        import os
        from unittest import mock

        env = {
            "DEEPGRAPH_LLM_EVALUATOR_FAMILY": "deepseek",
            "DEEPGRAPH_LLM_PRIMARY_FAMILY": "gemini-flash",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertNotEqual(
                auto_advance._role_default("EVALUATOR", "_FAMILY"),
                auto_advance._role_default("PRIMARY", "_FAMILY"),
            )

    def test_an_unset_role_yields_empty_not_a_stale_name(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(auto_advance._role_default("EVALUATOR", ""), "")


if __name__ == "__main__":
    unittest.main()
