"""A provider is declared in TOML; only its key comes from the environment."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from agents import llm_client


def _entry(**overrides) -> dict:
    entry = {
        "name": "gpt5",
        "base_url": "https://api.example.com/v1/",
        "model": "gpt-5.4",
        "model_family": "gpt",
        "protocol": "chat_completions",
        "api_key_env": "DEEPGRAPH_TEST_PROVIDER_KEY",
        "rpm": 0,
    }
    entry.update(overrides)
    return entry


class DeclaredProviderTests(unittest.TestCase):
    def test_key_is_resolved_from_the_named_environment_variable(self):
        with mock.patch.object(llm_client, "LLM_PROVIDERS", [_entry()]), \
                mock.patch.dict(os.environ, {"DEEPGRAPH_TEST_PROVIDER_KEY": "k-123"}):
            providers = llm_client._declared_providers()

        self.assertEqual(len(providers), 1)
        self.assertEqual(providers[0]["name"], "gpt5")
        self.assertEqual(providers[0]["api_key"], "k-123")
        self.assertEqual(providers[0]["model_family"], "gpt")
        # A trailing slash in the declared base_url must not double up later.
        self.assertEqual(providers[0]["base_url"], "https://api.example.com/v1")

    def test_a_literal_key_in_toml_is_refused_not_used(self):
        with mock.patch.object(
            llm_client, "LLM_PROVIDERS", [_entry(api_key="sk-inline-secret")]
        ):
            with self.assertRaisesRegex(
                llm_client.DeclaredProviderError, "literal api_key"
            ):
                llm_client._declared_providers()

    def test_an_unset_key_skips_the_provider_instead_of_falling_back(self):
        with mock.patch.object(llm_client, "LLM_PROVIDERS", [_entry()]), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEEPGRAPH_TEST_PROVIDER_KEY", None)
            self.assertEqual(llm_client._declared_providers(), [])

    def test_incomplete_entries_are_skipped(self):
        for missing in ("base_url", "model", "api_key_env"):
            with self.subTest(missing=missing), \
                    mock.patch.object(
                        llm_client, "LLM_PROVIDERS", [_entry(**{missing: ""})]
                    ), \
                    mock.patch.dict(
                        os.environ, {"DEEPGRAPH_TEST_PROVIDER_KEY": "k-123"}
                    ):
                self.assertEqual(llm_client._declared_providers(), [])

    def test_disabled_entries_are_ignored(self):
        with mock.patch.object(llm_client, "LLM_PROVIDERS", [_entry(enabled=False)]), \
                mock.patch.dict(os.environ, {"DEEPGRAPH_TEST_PROVIDER_KEY": "k-123"}):
            self.assertEqual(llm_client._declared_providers(), [])


class ProviderPoolTests(unittest.TestCase):
    def setUp(self):
        llm_client._providers.clear()
        llm_client._provider_stats.clear()
        llm_client._rate_limiters.clear()
        self.addCleanup(llm_client._providers.clear)
        self.addCleanup(llm_client._provider_stats.clear)

    def _init(self, entries):
        with mock.patch.object(llm_client, "LLM_PROVIDERS", entries), \
                mock.patch.object(llm_client, "MINIMAX_API_KEY", ""), \
                mock.patch.object(llm_client, "LLM_USE_TABCODE", False), \
                mock.patch.object(llm_client, "LLM_SECONDARY_ENABLED", False), \
                mock.patch.object(llm_client, "LLM_EXTRA_PROVIDERS_JSON", ""), \
                mock.patch.dict(os.environ, {"DEEPGRAPH_TEST_PROVIDER_KEY": "k-123"}):
            llm_client._init_providers()
        return list(llm_client._providers)

    def test_a_declared_provider_joins_the_pool(self):
        providers = self._init([_entry()])

        self.assertEqual([p["name"] for p in providers], ["gpt5"])
        self.assertEqual(providers[0]["model_family"], "gpt")

    def test_an_explicit_family_is_kept_verbatim(self):
        # Independence is judged on model_family, so a declared value must not
        # be silently truncated at the first hyphen.
        providers = self._init([_entry(model_family="claude-opus")])

        self.assertEqual(providers[0]["model_family"], "claude-opus")

    def test_family_is_derived_only_when_not_declared(self):
        providers = self._init([_entry(model_family="", model="glm-5.2")])

        self.assertEqual(providers[0]["model_family"], "glm")

    def test_two_declared_providers_can_differ_in_family(self):
        providers = self._init(
            [
                _entry(name="glm", model="glm-5.2", model_family="glm"),
                _entry(name="gpt5", model="gpt-5.4", model_family="gpt"),
            ]
        )
        families = {p["name"]: p["model_family"] for p in providers}

        self.assertEqual(families, {"glm": "glm", "gpt5": "gpt"})
        # This is exactly what the Frontier evaluator independence rule needs.
        self.assertNotEqual(families["glm"], families["gpt5"])


class RoleRouteTests(unittest.TestCase):
    def test_a_role_route_can_name_a_declared_provider_without_env_vars(self):
        routes = {
            "evaluator": [
                {
                    "provider_ref": "gpt5",
                    "model_ref": "gpt-5.4",
                    "model_family": "gpt",
                    "prompt_version": "evaluator_v1",
                }
            ]
        }
        with mock.patch.object(llm_client, "LLM_ROLE_ROUTES", routes):
            policy = llm_client.configured_role_route_policy("evaluator")

        self.assertIn("gpt5", policy)
        self.assertEqual(policy["gpt5"]["model"], "gpt-5.4")
        self.assertEqual(policy["gpt5"]["model_family"], "gpt")
        self.assertEqual(
            llm_client.configured_role_prompt_version("evaluator"), "evaluator_v1"
        )

    def test_an_unresolvable_route_still_fails_closed(self):
        with mock.patch.object(
            llm_client,
            "LLM_ROLE_ROUTES",
            {"evaluator": [{"provider_ref": "env:DEEPGRAPH_UNSET_EVALUATOR"}]},
        ):
            with self.assertRaises(llm_client.LLMProviderUnavailableError):
                llm_client.configured_role_route_policy("evaluator")


if __name__ == "__main__":
    unittest.main()
