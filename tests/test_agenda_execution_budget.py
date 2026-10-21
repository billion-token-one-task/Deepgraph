"""Execution-path LLM calls are charged to the agenda that owns them.

Discovery already metered itself. auto_research workers did not enter an
agenda scope, and grant sub-reservations never wrote agenda_token_ledger, so
execution spend stayed invisible until a whole grant settled.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents import llm_client
from agents.agenda_budget import (
    AgendaBudgetExceededError,
    agenda_scope,
    charged_tokens,
    check_budget,
    current_scope,
    record_usage,
)
from db import database
from meta_harness.grant_usage import GrantUsageLedger
from orchestrator import auto_research


class _TempAgendaDb(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "agenda.db"
        self.old_db_path = database.DB_PATH
        self.old_database_url = database.DATABASE_URL
        self._close_connection()
        database.DATABASE_URL = ""
        database.DB_PATH = self.db_path
        database.execute(
            """
            CREATE TABLE research_agendas (
                id INTEGER PRIMARY KEY,
                token_budget INTEGER,
                token_spent INTEGER DEFAULT 0,
                token_reserved INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                is_active INTEGER DEFAULT 1,
                updated_at TEXT
            )
            """
        )
        database.execute(
            """
            CREATE TABLE agenda_token_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agenda_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        database.execute(
            """
            CREATE TABLE resource_grants (
                id INTEGER PRIMARY KEY,
                agenda_id INTEGER NOT NULL,
                idea_id INTEGER,
                status TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        database.execute(
            """
            CREATE TABLE resource_grant_usage_reservations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agenda_id INTEGER NOT NULL,
                resource_grant_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                token_reserved INTEGER NOT NULL,
                tokens_used INTEGER,
                cost_usd REAL,
                status TEXT NOT NULL,
                release_reason TEXT,
                settled_at TEXT
            )
            """
        )
        database.commit()

    def tearDown(self):
        self._close_connection()
        database.DATABASE_URL = self.old_database_url
        database.DB_PATH = self.old_db_path
        self.tmpdir.cleanup()

    @staticmethod
    def _close_connection():
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            conn = getattr(database._local, attr, None)
            if conn is None:
                continue
            try:
                conn.close()
            except Exception:
                pass
            setattr(database._local, attr, None)

    def _agenda(self, agenda_id: int, token_budget, token_spent: int = 0) -> None:
        database.execute(
            """
            INSERT INTO research_agendas
                (id, token_budget, token_spent, status, is_active)
            VALUES (?, ?, ?, 'active', 1)
            """,
            (agenda_id, token_budget, token_spent),
        )
        database.commit()

    def _active_grant(self, grant_id: int, agenda_id: int, idea_id: int = 7) -> None:
        database.execute(
            """
            INSERT INTO resource_grants (id, agenda_id, idea_id, status, expires_at)
            VALUES (?, ?, ?, 'active', '2099-01-01 00:00:00')
            """,
            (grant_id, agenda_id, idea_id),
        )
        database.commit()

    def _ledger_tokens(self, agenda_id: int) -> int:
        row = database.fetchone(
            "SELECT COALESCE(SUM(tokens), 0) AS s FROM agenda_token_ledger WHERE agenda_id=?",
            (agenda_id,),
        )
        return int(row["s"])

    def _status(self, agenda_id: int) -> tuple[str, int, int]:
        row = database.fetchone(
            "SELECT status, token_spent, is_active FROM research_agendas WHERE id=?",
            (agenda_id,),
        )
        return str(row["status"]), int(row["token_spent"]), int(row["is_active"])


class BudgetCapTests(_TempAgendaDb):
    def test_null_zero_and_negative_budgets_stay_accounting_only(self):
        for agenda_id, budget in ((1, None), (2, 0), (3, -5)):
            self._agenda(agenda_id, budget)
            record_usage(agenda_id, "auto_research_execution", 40, cost_usd=0.1)
            check_budget(agenda_id)
            status, spent, active = self._status(agenda_id)
            self.assertEqual(status, "active")
            self.assertEqual(active, 1)
            self.assertEqual(spent, 40)
            self.assertEqual(self._ledger_tokens(agenda_id), 40)

    def test_crossing_the_cap_pauses_and_the_next_check_refuses(self):
        self._agenda(4, 25)
        record_usage(4, "auto_research_execution", 25)
        status, spent, active = self._status(4)
        self.assertEqual((status, spent, active), ("paused_budget", 25, 0))
        with self.assertRaises(AgendaBudgetExceededError):
            check_budget(4)

    def test_raising_the_budget_above_spend_unpauses_on_the_next_check(self):
        self._agenda(5, 10)
        record_usage(5, "auto_research_execution", 10)
        database.execute(
            "UPDATE research_agendas SET token_budget=30 WHERE id=5"
        )
        database.commit()
        check_budget(5)
        status, spent, active = self._status(5)
        self.assertEqual((status, spent, active), ("active", 10, 1))

    def test_in_flight_grant_usage_counts_before_token_spent_moves(self):
        self._agenda(6, 15, token_spent=0)
        self._active_grant(60, 6)
        database.execute(
            """
            INSERT INTO resource_grant_usage_reservations
                (agenda_id, resource_grant_id, operation, idempotency_key,
                 token_reserved, tokens_used, status)
            VALUES (6, 60, 'validation_code_iteration', 'k', 15, 15, 'settled')
            """
        )
        database.commit()
        self.assertEqual(charged_tokens(6, 0), 15)
        with self.assertRaises(AgendaBudgetExceededError) as caught:
            check_budget(6)
        self.assertEqual(caught.exception.token_spent, 15)
        self.assertEqual(self._status(6)[0], "paused_budget")
        self.assertEqual(self._status(6)[1], 0)


class GrantSettlementLedgerTests(_TempAgendaDb):
    def test_settling_execution_usage_writes_the_ledger_without_double_spending(self):
        self._agenda(8, 100, token_spent=4)
        self._active_grant(80, 8)
        database.execute(
            """
            INSERT INTO resource_grant_usage_reservations
                (agenda_id, resource_grant_id, operation, idempotency_key,
                 token_reserved, status)
            VALUES (8, 80, 'validation_code_iteration', 'once', 40, 'reserved')
            """
        )
        database.commit()
        reservation_id = int(
            database.fetchone(
                "SELECT id FROM resource_grant_usage_reservations WHERE idempotency_key='once'"
            )["id"]
        )
        GrantUsageLedger(80).settle(reservation_id, tokens_used=12, cost_usd=0.2)
        GrantUsageLedger(80).settle(reservation_id, tokens_used=12, cost_usd=0.2)

        self.assertEqual(self._ledger_tokens(8), 12)
        row = database.fetchone(
            "SELECT operation, cost_usd FROM agenda_token_ledger WHERE agenda_id=8"
        )
        self.assertEqual(row["operation"], "validation_code_iteration")
        self.assertAlmostEqual(float(row["cost_usd"]), 0.2)
        status, spent, _active = self._status(8)
        self.assertEqual(spent, 4)
        self.assertEqual(status, "active")

    def test_settlement_pauses_when_in_flight_usage_reaches_the_cap(self):
        self._agenda(9, 10)
        self._active_grant(90, 9)
        database.execute(
            """
            INSERT INTO resource_grant_usage_reservations
                (agenda_id, resource_grant_id, operation, idempotency_key,
                 token_reserved, status)
            VALUES (9, 90, 'auto_research_execution', 'cap', 10, 'reserved')
            """
        )
        database.commit()
        reservation_id = int(
            database.fetchone("SELECT id FROM resource_grant_usage_reservations")["id"]
        )
        GrantUsageLedger(90).settle(reservation_id, tokens_used=10)
        self.assertEqual(self._status(9)[0], "paused_budget")
        self.assertEqual(self._status(9)[1], 0)
        self.assertEqual(self._ledger_tokens(9), 10)


class ExecutionScopeHookTests(_TempAgendaDb):
    def setUp(self):
        super().setUp()
        self.old_providers = list(llm_client._providers)
        self.old_stats = dict(llm_client._provider_stats)
        self.old_limiters = dict(llm_client._rate_limiters)
        self.old_cooldown = dict(llm_client._provider_cooldown)
        llm_client._providers = [
            {
                "name": "tabcode",
                "base_url": "https://example.invalid",
                "api_key": "test-key",
                "model": "test-model",
                "protocol": "chat_completions",
            }
        ]
        llm_client._provider_stats = {
            "tabcode": {
                "calls": 0,
                "tokens": 0,
                "errors": 0,
                "total_latency": 0,
                "in_flight": 0,
                "cached_tokens": 0,
                "input_tokens": 0,
            }
        }
        llm_client._rate_limiters = {}
        llm_client._provider_cooldown = {}

    def tearDown(self):
        llm_client._providers = self.old_providers
        llm_client._provider_stats = self.old_stats
        llm_client._rate_limiters = self.old_limiters
        llm_client._provider_cooldown = self.old_cooldown
        super().tearDown()

    def test_legacy_call_inside_execution_scope_checks_before_the_provider(self):
        self._agenda(11, 100)
        with (
            mock.patch.object(llm_client, "_init_providers"),
            mock.patch.object(
                llm_client,
                "_call_provider",
                return_value=("valid enough response", 18, 0, 6, 0.4, "m"),
            ) as provider,
            agenda_scope(11, "auto_research_execution", "auto-research:11:1"),
        ):
            text, tokens = llm_client.call_llm("system", "user prompt here")

        self.assertEqual((text, tokens), ("valid enough response", 18))
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(self._ledger_tokens(11), 18)
        self.assertEqual(self._status(11)[1], 18)
        cost = database.fetchone(
            "SELECT cost_usd FROM agenda_token_ledger WHERE agenda_id=11"
        )
        self.assertAlmostEqual(float(cost["cost_usd"]), 0.4)

    def test_exhausted_scope_does_not_contact_the_provider(self):
        self._agenda(12, 10, token_spent=10)
        with (
            mock.patch.object(llm_client, "_init_providers"),
            mock.patch.object(llm_client, "_call_provider") as provider,
            agenda_scope(12, "auto_research_execution", "auto-research:12:1"),
        ):
            with self.assertRaises(AgendaBudgetExceededError):
                llm_client.call_llm("system", "user prompt here")
        self.assertEqual(provider.call_count, 0)
        self.assertEqual(self._ledger_tokens(12), 0)

    def test_unscoped_call_is_not_charged_to_an_agenda(self):
        self._agenda(13, 100)
        with (
            mock.patch.object(llm_client, "_init_providers"),
            mock.patch.object(
                llm_client,
                "_call_provider",
                return_value=("valid enough response", 9, 0, 2, None, ""),
            ),
        ):
            llm_client.call_llm("system", "user prompt here")
        self.assertEqual(self._ledger_tokens(13), 0)
        self.assertEqual(self._status(13)[1], 0)

    def test_role_call_on_the_execution_path_stops_before_the_provider(self):
        self._agenda(14, 8, token_spent=8)
        self._active_grant(140, 14, idea_id=70)
        with (
            mock.patch.object(llm_client, "_call_provider") as provider,
            agenda_scope(14, "auto_research_execution", "auto-research:14:70"),
        ):
            with self.assertRaises(AgendaBudgetExceededError):
                llm_client.call_llm_for_role(
                    "system",
                    "user prompt here",
                    agenda_id=14,
                    idea_id=70,
                    role="proposer",
                    stage="pilot",
                    resource_grant_id=140,
                    operation="validation_code_iteration",
                    idempotency_key="validation:t1",
                    prompt_version="proposer-v1",
                )
        self.assertEqual(provider.call_count, 0)


class AutoResearchScopeTests(unittest.TestCase):
    def test_process_candidate_enters_the_execution_scope(self):
        seen = {}

        def _capture(insight):
            seen["scope"] = current_scope()
            seen["insight"] = insight["id"]

        with mock.patch.object(auto_research, "_process_candidate_in_scope", _capture):
            auto_research._process_candidate({"id": 9, "agenda_id": 4})

        self.assertEqual(seen["insight"], 9)
        self.assertEqual(seen["scope"][:2], (4, "auto_research_execution"))
        self.assertIsNone(current_scope())

    def test_unscoped_candidate_does_not_invent_an_agenda(self):
        seen = {}

        def _capture(_insight):
            seen["scope"] = current_scope()

        with mock.patch.object(auto_research, "_process_candidate_in_scope", _capture):
            auto_research._process_candidate({"id": 9})

        self.assertIsNone(seen["scope"])
