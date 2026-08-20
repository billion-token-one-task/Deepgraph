"""A funded agenda may preempt discovery, but it may not own it.

Only discovery realizes a funded proposal design, and proposal grants carry a
short TTL, so a funded agenda preempts the rotation rather than waiting its
turn. The pin was assumed to self-clear: the job leaves
proposal_generation_granted on realization, and the grant leaves 'active' on
expiry.

That assumption fails when the candidate can never realize. On 2026-08-19
agenda 11's idea 110 held grant 147 in deferred/proposal_generation_granted
while its research problem was blocked from seeding ("research problem 9 is
held by spent proposal candidate 110"). Discovery therefore ran on agenda 11
on every single pass -- and each time the grant expired, the regrant path
issued a fresh one and re-pinned it. Agenda 10, the only agenda with work
ready to run, received zero discovery passes for over an hour while the M2
acceptance window sat empty.

Preemption without a fairness bound is starvation. The pin now yields after a
bounded number of consecutive passes regardless of WHY it is stuck, which is
the property that matters: the fix must not depend on diagnosing the reason.
"""

import unittest
from unittest import mock

import scripts.auto_advance as auto_advance


class DiscoveryPreemptFairnessTests(unittest.TestCase):
    AGENDAS = [7, 10, 11]

    def _select(self, state, funded_agendas):
        rows = [{"agenda_id": a, "first_expiry": f"2026-08-20T0{i}:00:00Z"}
                for i, a in enumerate(funded_agendas)]
        with mock.patch.object(auto_advance, "_rows", return_value=rows):
            return auto_advance._next_discovery_agenda(self.AGENDAS, state)

    def test_a_stuck_agenda_cannot_own_the_rotation(self):
        # agenda 11 is permanently funded-and-unrealizable, exactly like 110/147
        state = {}
        picks = [self._select(state, [11]) for _ in range(6)]
        self.assertLessEqual(
            picks.count(11),
            4,
            f"agenda 11 monopolised discovery: {picks}",
        )
        self.assertTrue(
            set(picks) - {11},
            f"no other agenda ever got a discovery pass: {picks}",
        )

    def test_agenda_10_gets_discovery_while_11_is_pinned(self):
        state = {}
        picks = [self._select(state, [11]) for _ in range(8)]
        self.assertIn(10, picks, f"the agenda with ready work never ran: {picks}")

    def test_a_funded_agenda_still_preempts_promptly(self):
        # the behaviour the preemption exists for: realization needs one pass
        state = {"discovery_rotation_last": 7}
        self.assertEqual(self._select(state, [11]), 11)

    def test_rotation_is_unaffected_when_nothing_is_funded(self):
        state = {"discovery_rotation_last": 7}
        self.assertEqual(self._select(state, []), 10)
        self.assertEqual(self._select(state, []), 11)
        self.assertEqual(self._select(state, []), 7)

    def test_the_rotation_cursor_is_not_dragged_by_preemption(self):
        # the flaw this test found in the first version of the fix: rotation
        # resumed from the PINNED agenda, so it only ever stepped to that
        # agenda's single successor and never walked the full ring
        state = {"discovery_rotation_last": 7}
        self._select(state, [11])  # preempted
        self.assertEqual(state["discovery_rotation_last"], 7)
        self.assertEqual(self._select(state, []), 10)

    def test_yielding_resets_the_streak_so_funding_can_preempt_again(self):
        state = {}
        for _ in range(3):
            self._select(state, [11])
        # after rotation has been reached the streak table is cleared
        self.assertEqual(state.get("discovery_preempt_streak"), {})


if __name__ == "__main__":
    unittest.main()


class DiscoverySupplyTests(unittest.TestCase):
    """Candidate supply, not GPU, was throttling the system.

    Measured 2026-08-20: mean GPU parallelism 0.38 against three lanes, all
    three idle with candidates queued behind them, and 99.87% of the token
    budget unspent. One discovery slot per pass meant each agenda waited
    ring-length turns -- about ninety minutes across nine agendas -- for a
    chance to invent anything.

    The ring also had to become stable. The rotation cursor is an INDEX into
    it, and ordering by updated_at made that index mean something different
    every pass: the busiest agenda is the most recently updated, so agenda 10
    sat permanently last while being the only one producing measurable runs.
    """

    def test_the_ring_order_is_stable(self):
        import inspect

        import scripts.auto_advance as aa

        source = inspect.getsource(aa.active_agenda_ids)
        self.assertIn("ORDER BY id ASC", source)
        self.assertNotIn("ORDER BY updated_at", source)

    def test_several_agendas_get_a_slot_each_pass(self):
        from scripts.auto_advance import DISCOVERY_AGENDAS_PER_PASS

        self.assertGreater(DISCOVERY_AGENDAS_PER_PASS, 1)

    def test_the_pass_loops_over_slots(self):
        import inspect

        import scripts.auto_advance as aa

        source = inspect.getsource(aa.main)
        self.assertIn("for _slot in range(DISCOVERY_AGENDAS_PER_PASS):", source)
        # an exhausted ring stops the loop rather than spinning
        self.assertIn("if discovery_agenda_id is None:", source)

    def test_every_agenda_is_reached_within_one_ring(self):
        # with a stable ring and nothing funded, rotation must visit them all
        state = {}
        agendas = [1, 2, 3, 4, 5, 6, 7, 10, 11]
        with mock.patch.object(auto_advance, "_rows", return_value=[]):
            seen = {
                auto_advance._next_discovery_agenda(agendas, state)
                for _ in range(len(agendas))
            }
        self.assertEqual(seen, set(agendas))

    def test_one_slot_per_agenda_per_pass(self):
        # a short ring hands back the same agenda repeatedly; running
        # discovery on it twice in a pass pays the LLM cost for nothing
        import inspect

        import scripts.auto_advance as aa

        source = inspect.getsource(aa.main)
        self.assertIn("discovered_this_pass", source)
        self.assertIn("if discovery_agenda_id in discovered_this_pass:", source)
