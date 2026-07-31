import sys
import types
import unittest
from unittest import mock

from meta_harness.scoped_llm import ScopedLLMError, proposer_json


class ScopedLLMTests(unittest.TestCase):
    def test_missing_scope_fails_before_llm_client_import(self):
        with mock.patch.dict(sys.modules, {"agents.llm_client": None}):
            with self.assertRaisesRegex(ScopedLLMError, "scope is required"):
                proposer_json(
                    "system",
                    "user",
                    llm_scope=None,
                    operation="paper_extraction:test",
                )

    def test_scope_is_forwarded_to_role_route_with_hard_cap(self):
        calls: list[dict] = []
        fake_client = types.ModuleType("agents.llm_client")
        fake_client.configured_role_prompt_version = lambda role: f"{role}_v1"

        def fake_call(_system, _user, **kwargs):
            calls.append(kwargs)
            return {"ok": True}, 7, {"provider": "test"}

        fake_client.call_llm_json_for_role = fake_call
        scope = {
            "agenda_id": 11,
            "idea_id": 12,
            "resource_grant_id": 13,
            "stage": "ingestion",
            "token_cap": 4000,
        }
        with mock.patch.dict(sys.modules, {"agents.llm_client": fake_client}):
            result = proposer_json(
                "system",
                "user",
                llm_scope=scope,
                operation="paper_extraction:test",
                token_cap=8000,
            )

        self.assertEqual(result[0], {"ok": True})
        self.assertEqual(calls[0]["agenda_id"], 11)
        self.assertEqual(calls[0]["idea_id"], 12)
        self.assertEqual(calls[0]["resource_grant_id"], 13)
        self.assertEqual(calls[0]["role"], "proposer")
        self.assertEqual(calls[0]["max_tokens"], 4000)
        self.assertTrue(
            calls[0]["idempotency_key"].startswith("paper_extraction:test:")
        )
