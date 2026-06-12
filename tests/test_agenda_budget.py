"""Tests for per-agenda token budgets (agents.agenda_budget + llm_client hook).

Covers ledger accounting, pause on budget exhaustion, the check-before-call
guard (no provider request once paused), resume (explicit endpoint or raised
budget), and the /api/token_usage summary endpoint.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ["DEEPGRAPH_DATABASE_URL"] = ""  # force SQLite tmpdir; never touch a real DB from the environment


SAMPLE_AGENDA = {
    "version": "v1",
    "name": "budget_test_agenda",
    "focus": ["long context"],
    "prefer": {"keywords": ["linear attention"]},
    "submitter": "alice",
    "token_budget": 1500,
}


class BudgetTestBase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["DEEPGRAPH_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        from db import database as db

        self._original_db_path = db.DB_PATH
        for attr in ("sqlite_conn", "pg_conn", "conn"):
            if hasattr(db._local, attr):
                try:
                    getattr(db._local, attr).close()
                except Exception:
                    pass
                delattr(db._local, attr)
        db.DB_PATH = Path(os.environ["DEEPGRAPH_DB_PATH"])
        db.init_db()
        self.db = db

    def tearDown(self):
        from db import database as db

        for attr in ("sqlite_conn", "pg_conn", "conn"):
            if hasattr(db._local, attr):
                try:
                    getattr(db._local, attr).close()
                except Exception:
                    pass
                delattr(db._local, attr)
        db.DB_PATH = self._original_db_path
        self._tmpdir.cleanup()
        os.environ.pop("DEEPGRAPH_DB_PATH", None)

    def _save_agenda(self, **overrides):
        from agents.agenda_loader import parse_agenda, save_agenda

        agenda = parse_agenda({**SAMPLE_AGENDA, **overrides})
        save_agenda(agenda)
        return agenda


class BudgetAccountingTests(BudgetTestBase):
    def test_default_budget_comes_from_config(self):
        from agents import agenda_budget
        from config import AGENDA_TOKEN_BUDGET_DEFAULT

        agenda = self._save_agenda(name="no_budget", token_budget=None)
        state = agenda_budget.get_budget_state(agenda.agenda_id)
        self.assertEqual(state["token_budget"], AGENDA_TOKEN_BUDGET_DEFAULT)
        self.assertEqual(state["token_spent"], 0)
        self.assertEqual(state["status"], "active")

    def test_record_usage_accumulates_and_writes_ledger(self):
        from agents import agenda_budget

        agenda = self._save_agenda()
        agenda_budget.record_usage(agenda.agenda_id, "tier1_discovery", 400)
        agenda_budget.record_usage(agenda.agenda_id, "tier2_discovery", 300)

        state = agenda_budget.get_budget_state(agenda.agenda_id)
        self.assertEqual(state["token_spent"], 700)
        self.assertEqual(state["status"], "active")

        rows = self.db.fetchall(
            "SELECT operation, tokens FROM agenda_token_ledger WHERE agenda_id=? ORDER BY id",
            (agenda.agenda_id,),
        )
        self.assertEqual(
            [(r["operation"], r["tokens"]) for r in rows],
            [("tier1_discovery", 400), ("tier2_discovery", 300)],
        )

    def test_exceeding_budget_pauses_agenda_and_blocks_checks(self):
        from agents import agenda_budget

        agenda = self._save_agenda()  # budget 1500
        agenda_budget.record_usage(agenda.agenda_id, "op", 1600)

        state = agenda_budget.get_budget_state(agenda.agenda_id)
        self.assertEqual(state["status"], "paused_budget")
        with self.assertRaises(agenda_budget.AgendaBudgetExceededError):
            agenda_budget.check_budget(agenda.agenda_id)

    def test_resume_reactivates(self):
        from agents import agenda_budget

        agenda = self._save_agenda()
        agenda_budget.record_usage(agenda.agenda_id, "op", 1600)
        # Resume with a raised budget: checks pass again
        state = agenda_budget.resume_agenda(agenda.agenda_id, token_budget=5000)
        self.assertEqual(state["status"], "active")
        agenda_budget.check_budget(agenda.agenda_id)  # must not raise

    def test_raising_budget_alone_unblocks(self):
        from agents import agenda_budget

        agenda = self._save_agenda()
        agenda_budget.record_usage(agenda.agenda_id, "op", 1600)
        self.db.execute(
            "UPDATE research_agendas SET token_budget=10000 WHERE id=?",
            (agenda.agenda_id,),
        )
        self.db.commit()
        agenda_budget.check_budget(agenda.agenda_id)  # must not raise
        state = agenda_budget.get_budget_state(agenda.agenda_id)
        self.assertEqual(state["status"], "active")

    def test_zero_budget_disables_cap(self):
        from agents import agenda_budget

        agenda = self._save_agenda(name="uncapped", token_budget=0)
        agenda_budget.record_usage(agenda.agenda_id, "op", 10_000_000)
        agenda_budget.check_budget(agenda.agenda_id)  # must not raise

    def test_zero_budget_still_accounts_usage(self):
        """budget 0 = no cap, never paused, but ledger + token_spent keep counting."""
        from agents import agenda_budget

        agenda = self._save_agenda(name="uncapped_accounting", token_budget=0)
        agenda_budget.record_usage(agenda.agenda_id, "tier1_discovery", 4_000_000)
        agenda_budget.record_usage(agenda.agenda_id, "tier2_discovery", 6_000_000)
        agenda_budget.check_budget(agenda.agenda_id)  # must not raise

        state = agenda_budget.get_budget_state(agenda.agenda_id)
        self.assertEqual(state["token_spent"], 10_000_000)
        self.assertEqual(state["status"], "active")  # never paused_budget
        self.assertFalse(state["exhausted"])
        rows = self.db.fetchall(
            "SELECT operation, tokens FROM agenda_token_ledger WHERE agenda_id=? ORDER BY id",
            (agenda.agenda_id,),
        )
        self.assertEqual(
            [(r["operation"], r["tokens"]) for r in rows],
            [("tier1_discovery", 4_000_000), ("tier2_discovery", 6_000_000)],
        )

    def test_null_budget_with_zero_default_is_uncapped(self):
        """token_budget NULL + default 0: enforcement off, accounting on."""
        import unittest as _ut

        from agents import agenda_budget
        from config import AGENDA_TOKEN_BUDGET_DEFAULT

        if AGENDA_TOKEN_BUDGET_DEFAULT > 0:
            raise _ut.SkipTest("env overrides the shipped default of 0")

        agenda = self._save_agenda(name="null_budget", token_budget=None)
        agenda_budget.record_usage(agenda.agenda_id, "op", 3_000_000)
        agenda_budget.check_budget(agenda.agenda_id)  # must not raise
        state = agenda_budget.get_budget_state(agenda.agenda_id)
        self.assertEqual(state["token_spent"], 3_000_000)
        self.assertEqual(state["status"], "active")


class LlmClientHookTests(BudgetTestBase):
    """call_llm must meter scoped calls and stop cleanly once the budget is spent."""

    def setUp(self):
        super().setUp()
        import agents.llm_client as llm

        self.llm = llm
        self._saved = (
            llm._providers,
            llm._provider_stats,
            llm._rate_limiters,
            llm._provider_cooldown,
            llm._call_provider,
        )
        llm._providers = [
            {
                "name": "fake",
                "base_url": "http://localhost",
                "api_key": "k",
                "model": "m",
                "protocol": "chat_completions",
                "rpm": 0,
            }
        ]
        llm._provider_stats = {
            "fake": {
                "calls": 0, "tokens": 0, "errors": 0, "total_latency": 0,
                "in_flight": 0, "cached_tokens": 0, "input_tokens": 0,
            }
        }
        llm._rate_limiters = {}
        llm._provider_cooldown = {}
        self.provider_calls = []

        def _fake_call(provider, system_prompt, user_prompt, max_tokens):
            self.provider_calls.append(provider["name"])
            return ("response text long enough to count", 1000, 0, 500)

        llm._call_provider = _fake_call

    def tearDown(self):
        llm = self.llm
        (
            llm._providers,
            llm._provider_stats,
            llm._rate_limiters,
            llm._provider_cooldown,
            llm._call_provider,
        ) = self._saved
        super().tearDown()

    def test_scoped_calls_are_metered_and_capped(self):
        from agents import agenda_budget

        agenda = self._save_agenda()  # budget 1500, fake call = 1000 tokens

        with agenda_budget.agenda_scope(agenda.agenda_id, "unit_test"):
            self.llm.call_llm("sys", "user")  # spend 1000 (< 1500: allowed)
            self.llm.call_llm("sys", "user")  # spend 2000 (pauses the agenda)
            with self.assertRaises(agenda_budget.AgendaBudgetExceededError):
                self.llm.call_llm("sys", "user")  # blocked BEFORE the provider

        # Third call never reached the provider: clean stop, no token spent
        self.assertEqual(len(self.provider_calls), 2)
        state = agenda_budget.get_budget_state(agenda.agenda_id)
        self.assertEqual(state["token_spent"], 2000)
        self.assertEqual(state["status"], "paused_budget")
        ledger = self.db.fetchall(
            "SELECT operation, tokens FROM agenda_token_ledger WHERE agenda_id=?",
            (agenda.agenda_id,),
        )
        self.assertEqual(len(ledger), 2)
        self.assertEqual(ledger[0]["operation"], "unit_test")

    def test_resume_allows_further_calls(self):
        from agents import agenda_budget

        agenda = self._save_agenda()
        with agenda_budget.agenda_scope(agenda.agenda_id, "unit_test"):
            self.llm.call_llm("sys", "user")
            self.llm.call_llm("sys", "user")
            with self.assertRaises(agenda_budget.AgendaBudgetExceededError):
                self.llm.call_llm("sys", "user")

            agenda_budget.resume_agenda(agenda.agenda_id, token_budget=10_000)
            self.llm.call_llm("sys", "user")  # works again

        self.assertEqual(len(self.provider_calls), 3)
        state = agenda_budget.get_budget_state(agenda.agenda_id)
        self.assertEqual(state["token_spent"], 3000)
        self.assertEqual(state["status"], "active")

    def test_unscoped_calls_not_metered(self):
        self.llm.call_llm("sys", "user")
        rows = self.db.fetchall("SELECT * FROM agenda_token_ledger")
        self.assertEqual(rows, [])


class BudgetRoutesTests(BudgetTestBase):
    def setUp(self):
        super().setUp()
        from web import app as app_module

        self.client = app_module.app.test_client()

    def test_resume_endpoint_and_token_usage(self):
        from agents import agenda_budget

        agenda = self._save_agenda()
        agenda_budget.record_usage(agenda.agenda_id, "tier1_discovery", 1600)

        r = self.client.get("/api/token_usage")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["totals"]["tokens"], 1600)
        row = next(a for a in body["agendas"] if a["agenda_id"] == agenda.agenda_id)
        self.assertEqual(row["status"], "paused_budget")
        self.assertEqual(row["token_spent"], 1600)
        self.assertEqual(row["ledger_entries"], 1)

        r = self.client.post(
            f"/api/research_agenda/{agenda.agenda_id}/resume",
            json={"token_budget": 5000},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["budget"]["status"], "active")
        self.assertEqual(r.get_json()["budget"]["token_budget"], 5000)

    def test_resume_endpoint_validates_input(self):
        agenda = self._save_agenda()
        r = self.client.post(
            f"/api/research_agenda/{agenda.agenda_id}/resume",
            json={"token_budget": "lots"},
        )
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/research_agenda/999999/resume", json={})
        self.assertEqual(r.status_code, 404)

    def test_agenda_insights_endpoint_isolated(self):
        agenda_a = self._save_agenda(name="iso_a")
        agenda_b = self._save_agenda(name="iso_b")
        self.db.execute(
            "INSERT INTO deep_insights (id, tier, status, title, agenda_id) "
            "VALUES (1, 2, 'candidate', 'insight for A', ?)",
            (agenda_a.agenda_id,),
        )
        self.db.execute(
            "INSERT INTO deep_insights (id, tier, status, title, agenda_id) "
            "VALUES (2, 2, 'candidate', 'insight for B', ?)",
            (agenda_b.agenda_id,),
        )
        self.db.commit()

        r = self.client.get(f"/api/research_agenda/{agenda_a.agenda_id}/insights")
        self.assertEqual(r.status_code, 200)
        titles = [i["title"] for i in r.get_json()["insights"]]
        self.assertEqual(titles, ["insight for A"])

        r = self.client.get(f"/api/research_agenda/{agenda_b.agenda_id}/selections")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["selections"], [])


if __name__ == "__main__":
    unittest.main()


class LedgerMigrationTest(BudgetTestBase):
    """Existing DBs created before agenda_token_ledger must get the table
    from _ensure_agenda_isolation_schema, not only from the schema files
    (on Postgres the best-effort schema replay can be skipped when an
    earlier statement aborts the transaction)."""

    def test_ensure_schema_recreates_ledger_table(self):
        db = self.db
        db.execute("DROP TABLE agenda_token_ledger")
        db.get_conn().commit()
        rows = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='agenda_token_ledger'"
        )
        self.assertEqual(rows, [])

        db._ensure_agenda_isolation_schema()

        rows = db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='agenda_token_ledger'"
        )
        self.assertEqual(len(rows), 1)

        from agents import agenda_loader, agenda_budget

        agenda = agenda_loader.parse_agenda(dict(SAMPLE_AGENDA))
        agenda_id = agenda_loader.save_agenda(agenda)
        agenda_budget.record_usage(agenda_id, "test_op", tokens=42)
        total = db.fetchone(
            "SELECT SUM(tokens) AS t FROM agenda_token_ledger WHERE agenda_id = ?",
            (agenda_id,),
        )
        self.assertEqual(total["t"], 42)
