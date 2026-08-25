"""Reopening and closing an agenda must not create stranded work.

Agenda 14 held the only `supported` outcome in the system and twelve unfinished
jobs, yet was closed, and nothing short of a raw UPDATE could reopen it. The
operator surface added on 2026-08-25 has to preserve two properties this
repository has already paid to learn:

* an activated agenda with no fundable headroom produces candidates that can
  never receive a grant -- the stranded-candidate failure mode;
* a closed agenda holding an active grant leaves that authority with no
  legitimate settlement path.
"""

import unittest
from unittest import mock

from agents.agenda_repository import AgendaNotFoundError, AgendaRepository


class _Cursor:
    def __init__(self, rowcount=1):
        self.rowcount = rowcount


class _FakeDb:
    """Enough of the db module for the activation path, and nothing more."""

    def __init__(self, agenda=None, open_grants=0, rowcount=1):
        self.agenda = agenda
        self.open_grants = open_grants
        self.rowcount = rowcount
        self.executed = []
        self.committed = False
        self.rolled_back = False

    def fetchone(self, sql, params=()):
        if "FROM resource_grants" in sql:
            return {"n": self.open_grants}
        return dict(self.agenda) if self.agenda else None

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        return _Cursor(self.rowcount)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def _agenda(**overrides):
    row = {
        "id": 14,
        "status": "closed",
        "is_active": 0,
        "token_budget": 3_000_000,
        "token_spent": 367_387,
        "token_reserved": 0,
    }
    row.update(overrides)
    return row


class ActivationTests(unittest.TestCase):
    def _repo(self, fake):
        patcher = mock.patch("agents.agenda_repository.db", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return AgendaRepository()

    def test_reopening_writes_status_and_is_active_together(self):
        fake = _FakeDb(_agenda())
        change = self._repo(fake).set_active(14, True)
        self.assertEqual(change["before"], {"status": "closed", "is_active": 0})
        self.assertEqual(change["after"], {"status": "active", "is_active": 1})
        sql, params = fake.executed[-1]
        self.assertIn("is_active=?", sql)
        self.assertIn("status=?", sql)
        self.assertEqual(params[:2], (1, "active"))
        self.assertTrue(fake.committed)

    def test_activation_is_refused_without_fundable_headroom(self):
        exhausted = _agenda(token_budget=500_000, token_spent=494_765,
                            token_reserved=5_235)
        fake = _FakeDb(exhausted)
        with self.assertRaises(ValueError) as caught:
            self._repo(fake).set_active(14, True)
        self.assertIn("no fundable headroom", str(caught.exception))
        self.assertEqual(fake.executed, [])
        self.assertFalse(fake.committed)

    def test_reserved_tokens_count_against_headroom(self):
        # Budget looks ample until an outstanding reservation is counted.
        fake = _FakeDb(_agenda(token_budget=1_000_000, token_spent=400_000,
                               token_reserved=600_000))
        with self.assertRaises(ValueError):
            self._repo(fake).set_active(14, True)

    def test_closing_is_refused_while_a_grant_is_outstanding(self):
        fake = _FakeDb(_agenda(status="active", is_active=1), open_grants=1)
        with self.assertRaises(ValueError) as caught:
            self._repo(fake).set_active(14, False)
        self.assertIn("active grant", str(caught.exception))
        self.assertEqual(fake.executed, [])

    def test_closing_settled_agenda_writes_closed_state(self):
        fake = _FakeDb(_agenda(status="active", is_active=1), open_grants=0)
        change = self._repo(fake).set_active(14, False)
        self.assertEqual(change["after"], {"status": "closed", "is_active": 0})
        self.assertEqual(fake.executed[-1][1][:2], (0, "closed"))

    def test_unknown_agenda_is_reported_not_silently_ignored(self):
        fake = _FakeDb(None)
        with self.assertRaises(AgendaNotFoundError):
            self._repo(fake).set_active(999, True)

    def test_a_lost_update_rolls_back_instead_of_committing(self):
        fake = _FakeDb(_agenda(), rowcount=0)
        with self.assertRaises(AgendaNotFoundError):
            self._repo(fake).set_active(14, True)
        self.assertTrue(fake.rolled_back)
        self.assertFalse(fake.committed)


if __name__ == "__main__":
    unittest.main()
