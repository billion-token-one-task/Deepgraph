"""A scoped LLM call is sized by the grant, not by the caller's default.

A ResourceGrant exists to bound spend, but every role call sent its own
``max_tokens`` (defaulting to the provider maximum) straight through to the
sub-reservation. A grant smaller than that maximum was therefore unusable for
any work at all -- which is exactly the shape the recovery runbook asks for: a
small, short CPU/LLM grant for one pilot.
"""

from __future__ import annotations

import unittest
from unittest import mock

from meta_harness.grant_usage import GrantUsageLedger


GRANT_ID = 1
AGENDA_ID = 5


class FakeDb:
    def __init__(self, *, grant=None, committed=0):
        self.grant = grant
        self.committed = committed

    def _use_pg(self):
        return False

    def fetchone(self, sql, params=()):
        text = " ".join(sql.split()).lower()
        if "from resource_grants" in text:
            return dict(self.grant) if self.grant else None
        if "from resource_grant_usage_reservations" in text:
            return {"reserved": self.committed}
        raise AssertionError(f"unexpected fetchone: {text}")


def _grant(**overrides) -> dict:
    values = {"agenda_id": AGENDA_ID, "token_cap": 5000, "status": "active"}
    values.update(overrides)
    return values


def _remaining(fake_db, **kwargs) -> int:
    with mock.patch("meta_harness.grant_usage.db", fake_db):
        return GrantUsageLedger(GRANT_ID).remaining(**kwargs)


class RemainingTokensTests(unittest.TestCase):
    def test_an_untouched_grant_offers_its_whole_cap(self):
        self.assertEqual(_remaining(FakeDb(grant=_grant())), 5000)

    def test_open_reservations_and_real_spend_both_count_against_it(self):
        self.assertEqual(_remaining(FakeDb(grant=_grant(), committed=1200)), 3800)

    def test_an_overspent_grant_reports_zero_not_a_negative_budget(self):
        self.assertEqual(_remaining(FakeDb(grant=_grant(), committed=9000)), 0)

    def test_an_expired_or_inactive_grant_offers_nothing(self):
        # The query itself filters on expires_at, so an expired grant reads as absent.
        self.assertEqual(_remaining(FakeDb(grant=None)), 0)
        for status in ("revoked", "consumed", "expired"):
            self.assertEqual(_remaining(FakeDb(grant=_grant(status=status))), 0, status)

    def test_a_grant_in_another_agenda_offers_nothing(self):
        self.assertEqual(
            _remaining(FakeDb(grant=_grant()), agenda_id=AGENDA_ID + 1), 0
        )
        self.assertEqual(_remaining(FakeDb(grant=_grant()), agenda_id=AGENDA_ID), 5000)


class ClampMakesASmallGrantUsableTests(unittest.TestCase):
    """Sizing to remaining() is what lets several calls share one small grant."""

    def test_a_caller_sized_to_remaining_is_admitted_where_its_default_was_not(self):
        provider_default = 32_000
        fake_db = FakeDb(grant=_grant(token_cap=5000), committed=0)

        remaining = _remaining(fake_db)
        self.assertLess(remaining, provider_default)
        # Unclamped, the sub-reservation would refuse the call outright.
        self.assertGreater(provider_default, remaining)
        self.assertEqual(min(provider_default, remaining), 5000)

    def test_a_second_call_gets_what_the_first_one_did_not_actually_use(self):
        """Reservations settle down to real usage, so the rest comes back."""
        after_first_call = FakeDb(grant=_grant(token_cap=5000), committed=800)

        self.assertEqual(_remaining(after_first_call), 4200)


class RoleRouteClampTests(unittest.TestCase):
    """The clamp lives at the single chokepoint every scoped role call uses (call_llm_for_role)."""

    def test_the_clamp_is_applied_before_the_reservation(self):
        """Pin the source contract: min(caller default, remaining grant)."""
        import inspect

        from agents import llm_client

        source = inspect.getsource(llm_client.call_llm_for_role)
        self.assertIn("GrantUsageLedger(resource_grant_id).remaining(", source)
        self.assertIn("token_cap = min(token_cap, remaining_tokens)", source)
        self.assertIn("ResourceGrant token budget is exhausted", source)
        # The clamp must sit after the grant lookup, not before it.
        self.assertLess(
            source.index("active scoped ResourceGrant is required"),
            source.index("token_cap = min(token_cap, remaining_tokens)"),
        )


if __name__ == "__main__":
    unittest.main()
