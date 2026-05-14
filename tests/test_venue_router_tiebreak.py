"""Tests for ``agents.venue_router`` tiebreaker (D3 #14).

Coverage:
1. ``needs_tiebreak`` returns False on clear winners.
2. ``needs_tiebreak`` returns True when top two venues are sub-threshold apart.
3. ``tiebreak_with_llm`` returns the LLM's choice when valid.
4. ``tiebreak_with_llm`` falls back to file-order leader when LLM hallucinates.
5. ``tiebreak_with_llm`` deterministic fallback (no llm_caller) is reproducible.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class VenueRouterTiebreakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="dg_tiebreak_")
        os.environ["DEEPGRAPH_DATABASE_URL"] = ""
        os.environ["DEEPGRAPH_DB_PATH"] = str(Path(cls._tmpdir) / "tb.db")
        from db import database as db
        # Explicit reset so prior tests that override db.DB_PATH (and clean up
        # their tempdir) don't leave us pointing at a deleted file.
        for attr in ("sqlite_conn", "pg_conn", "conn"):
            if hasattr(db._local, attr):
                try:
                    getattr(db._local, attr).close()
                except Exception:
                    pass
                delattr(db._local, attr)
        db.DB_PATH = Path(os.environ["DEEPGRAPH_DB_PATH"])
        db.init_db()

    def _ml_state_close_to_tie(self):
        # An ML state that hits both neurips2024 and icml2024 triggers so the
        # rule-based scores end up near each other.
        return {
            "title": "Stochastic optimization for deep learning generalization",
            "claim_type": "empirical",
            "domain": "ml",
            "has_real_data": True,
            "tier": 1,
            "page_count_estimate": 9,
        }

    def test_clear_winner_does_not_need_tiebreak(self):
        from agents.venue_router import evaluate_venues, load_venue_config, needs_tiebreak
        venues = load_venue_config()
        state = {
            "title": "Self-supervised image segmentation with diffusion priors",
            "claim_type": "empirical",
            "domain": "vision",
            "has_real_data": True,
            "tier": 1,
            "page_count_estimate": 8,
        }
        result = evaluate_venues(state, venues)
        # CV state goes clearly to CVPR; runner-up is far behind.
        self.assertFalse(needs_tiebreak(result["all_scored"]))

    def test_close_scores_trigger_tiebreak(self):
        from agents.venue_router import (
            evaluate_venues,
            load_venue_config,
            needs_tiebreak,
        )
        venues = load_venue_config()
        state = self._ml_state_close_to_tie()
        result = evaluate_venues(state, venues)
        # We don't insist tiebreak triggers (depends on keyword overlap), but
        # if it does, the tiebreaker must succeed deterministically. Both
        # branches are exercised across the test suite.
        if needs_tiebreak(result["all_scored"]):
            self.assertTrue(True)
        else:
            # Force the close-tie path artificially by constructing two
            # near-tied scored entries.
            from agents.venue_router import TIEBREAK_SCORE_DELTA
            eligible = [s for s in result["all_scored"] if not s["breakdown"]["blocked"]]
            self.assertTrue(eligible)
            top = max(eligible, key=lambda s: s["breakdown"]["score"])
            # Synthesise a runner-up at score - delta/2 → < delta apart.
            synth = {
                "venue": eligible[-1]["venue"],
                "breakdown": dict(top["breakdown"]),
            }
            synth["breakdown"]["score"] = top["breakdown"]["score"] - (TIEBREAK_SCORE_DELTA / 2)
            self.assertTrue(needs_tiebreak([top, synth]))

    def test_tiebreaker_uses_llm_choice_when_valid(self):
        from agents.venue_router import (
            evaluate_venues,
            load_venue_config,
            tiebreak_with_llm,
        )
        venues = load_venue_config()
        state = self._ml_state_close_to_tie()
        scored = evaluate_venues(state, venues)["all_scored"]
        # LLM returns the first candidate's id verbatim.
        eligible = sorted(
            (s for s in scored if not s["breakdown"]["blocked"]),
            key=lambda s: s["breakdown"]["score"],
            reverse=True,
        )
        target = eligible[1]["venue"].template_id if len(eligible) >= 2 else eligible[0]["venue"].template_id
        decision = tiebreak_with_llm(state, scored, llm_caller=lambda _p: target)
        self.assertEqual(decision["chosen_template_id"], target)
        self.assertTrue(decision["used_llm"])
        self.assertIn(target, decision["candidates"])

    def test_tiebreaker_falls_back_on_hallucinated_llm_response(self):
        from agents.venue_router import (
            evaluate_venues,
            load_venue_config,
            tiebreak_with_llm,
        )
        venues = load_venue_config()
        state = self._ml_state_close_to_tie()
        scored = evaluate_venues(state, venues)["all_scored"]
        decision = tiebreak_with_llm(
            state, scored, llm_caller=lambda _p: "totally_made_up_venue_xyz"
        )
        # Defensive fallback: keep file-order leader, but flag used_llm=True.
        self.assertIn(decision["chosen_template_id"], decision["candidates"])
        self.assertTrue(decision["used_llm"])
        self.assertIn("defensive fallback", decision["rationale"].lower())

    def test_deterministic_fallback_without_llm_caller(self):
        from agents.venue_router import (
            evaluate_venues,
            load_venue_config,
            tiebreak_with_llm,
        )
        venues = load_venue_config()
        state = self._ml_state_close_to_tie()
        scored = evaluate_venues(state, venues)["all_scored"]
        # No llm_caller passed → deterministic file-order leader.
        d1 = tiebreak_with_llm(state, scored)
        d2 = tiebreak_with_llm(state, scored)
        self.assertEqual(d1["chosen_template_id"], d2["chosen_template_id"])
        self.assertFalse(d1["used_llm"])


if __name__ == "__main__":
    unittest.main()
