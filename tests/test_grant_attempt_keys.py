"""An unusable provider response must not retire the operation forever.

The idempotency key had two jobs it could not both do. As a charge guard it
must refuse a key that was already settled. As the de-facto retry policy that
meant a call which succeeded, was billed, and returned nothing usable could
never be attempted again - and the candidate and grant it belonged to were
stranded permanently.

Measured on 2026-08-10: agenda 11 lost both candidates this way. Idea 128 spent
10278 tokens on a proposal whose realized idea was then discarded; idea 127
spent 3965 on a method invention that produced no method. Neither could ever be
retried, so the agenda had no live candidate left.

Attempt identity is now separate from operation identity, and the ledger owns
the allocation so no ring has to remember to do it.
"""

from __future__ import annotations

import re
import unittest
from unittest import mock

from meta_harness import grant_usage
from meta_harness.grant_usage import GrantUsageError, GrantUsageLedger


BASE = "proposal-method:11:127"


class AttemptKeyTests(unittest.TestCase):
    def _ledger(self, prior_count):
        ledger = GrantUsageLedger(20)
        patch = mock.patch.object(
            grant_usage.db, "fetchone", return_value={"c": prior_count}
        )
        return ledger, patch

    def test_first_attempt_is_serialised(self):
        ledger, patch = self._ledger(0)
        with patch:
            self.assertEqual(ledger.next_attempt_key(BASE), f"{BASE}:t1")

    def test_a_burned_key_yields_a_fresh_attempt(self):
        """The exact defect: one settled key used to end the operation."""

        ledger, patch = self._ledger(1)
        with patch:
            self.assertEqual(ledger.next_attempt_key(BASE), f"{BASE}:t2")

    def test_attempts_are_bounded(self):
        ledger, patch = self._ledger(3)
        with patch, self.assertRaises(GrantUsageError) as caught:
            ledger.next_attempt_key(BASE)
        self.assertIn("all 3 attempts", str(caught.exception))

    def test_bound_is_configurable_and_must_be_positive(self):
        ledger, patch = self._ledger(3)
        with patch:
            self.assertEqual(ledger.next_attempt_key(BASE, max_attempts=5), f"{BASE}:t4")
        ledger2, patch2 = self._ledger(0)
        with patch2, self.assertRaises(GrantUsageError):
            ledger2.next_attempt_key(BASE, max_attempts=0)

    def test_legacy_unsuffixed_key_counts_as_an_attempt(self):
        """The stranded rows carry a bare key; they must not get a free t1."""

        captured = {}

        def fetchone(sql, params=()):
            captured["sql"] = " ".join(str(sql).split())
            captured["params"] = params
            return {"c": 1}

        ledger = GrantUsageLedger(20)
        with mock.patch.object(grant_usage.db, "fetchone", side_effect=fetchone):
            key = ledger.next_attempt_key(BASE)

        self.assertEqual(key, f"{BASE}:t2")
        self.assertIn("idempotency_key=?", captured["sql"])
        self.assertIn("idempotency_key LIKE ?", captured["sql"])
        self.assertEqual(captured["params"], (20, BASE, f"{BASE}:t%"))

    def test_a_blank_base_is_refused(self):
        ledger = GrantUsageLedger(20)
        with self.assertRaises(GrantUsageError):
            ledger.next_attempt_key("   ")

    def test_counting_is_scoped_to_this_grant(self):
        """Another grant's attempts must not consume this grant's budget."""

        captured = {}

        def fetchone(sql, params=()):
            captured["params"] = params
            return {"c": 0}

        with mock.patch.object(grant_usage.db, "fetchone", side_effect=fetchone):
            GrantUsageLedger(77).next_attempt_key(BASE)

        self.assertEqual(captured["params"][0], 77)


class CallSiteTests(unittest.TestCase):
    """Both resource-granted rings must allocate through the ledger."""

    def test_proposal_ring_uses_the_ledger(self):
        import inspect

        from agents import paper_idea_agent

        source = inspect.getsource(paper_idea_agent.discover_paper_ideas)
        self.assertIn("next_attempt_key", source)
        # The bare operation name may still appear as the *base* key; what must
        # never happen again is a bare key reaching the ledger as the
        # reservation's identity.
        keys = re.findall(r"idempotency_key=([^,\n]+)", source)
        self.assertEqual(len(keys), 2, f"unexpected proposal call sites: {keys}")
        self.assertEqual(
            sorted(item.strip() for item in keys), ["experiment_key", "method_key"]
        )

    def test_forge_ring_no_longer_keeps_its_own_copy(self):
        import inspect

        from agents import experiment_forge

        source = inspect.getsource(experiment_forge)
        self.assertIn("next_attempt_key", source)
        self.assertNotIn("idempotency_key LIKE", source)


if __name__ == "__main__":
    unittest.main()
