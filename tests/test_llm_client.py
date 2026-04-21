import time
import unittest

import httpx

from agents import llm_client


class LlmClientCooldownTests(unittest.TestCase):
    def setUp(self):
        self.old_providers = list(llm_client._providers)
        self.old_provider_idx = llm_client._provider_idx
        self.old_provider_stats = dict(llm_client._provider_stats)
        self.old_rate_limiters = dict(llm_client._rate_limiters)
        self.old_provider_cooldown = dict(llm_client._provider_cooldown)
        self.old_llm_use_tabcode = llm_client.LLM_USE_TABCODE
        self.old_llm_api_key = llm_client.LLM_API_KEY
        self.old_llm_base_url = llm_client.LLM_BASE_URL
        self.old_llm_model = llm_client.LLM_MODEL
        self.old_llm_protocol = llm_client.LLM_PROTOCOL
        self.old_llm_rpm = llm_client.LLM_RPM
        self.old_secondary_enabled = llm_client.LLM_SECONDARY_ENABLED
        self.old_secondary_api_key = llm_client.LLM_SECONDARY_API_KEY
        self.old_secondary_base_url = llm_client.LLM_SECONDARY_BASE_URL
        self.old_secondary_model = llm_client.LLM_SECONDARY_MODEL
        self.old_secondary_protocol = llm_client.LLM_SECONDARY_PROTOCOL
        self.old_secondary_rpm = llm_client.LLM_SECONDARY_RPM

    def tearDown(self):
        llm_client._providers = self.old_providers
        llm_client._provider_idx = self.old_provider_idx
        llm_client._provider_stats = self.old_provider_stats
        llm_client._rate_limiters = self.old_rate_limiters
        llm_client._provider_cooldown = self.old_provider_cooldown
        llm_client.LLM_USE_TABCODE = self.old_llm_use_tabcode
        llm_client.LLM_API_KEY = self.old_llm_api_key
        llm_client.LLM_BASE_URL = self.old_llm_base_url
        llm_client.LLM_MODEL = self.old_llm_model
        llm_client.LLM_PROTOCOL = self.old_llm_protocol
        llm_client.LLM_RPM = self.old_llm_rpm
        llm_client.LLM_SECONDARY_ENABLED = self.old_secondary_enabled
        llm_client.LLM_SECONDARY_API_KEY = self.old_secondary_api_key
        llm_client.LLM_SECONDARY_BASE_URL = self.old_secondary_base_url
        llm_client.LLM_SECONDARY_MODEL = self.old_secondary_model
        llm_client.LLM_SECONDARY_PROTOCOL = self.old_secondary_protocol
        llm_client.LLM_SECONDARY_RPM = self.old_secondary_rpm

    def test_next_provider_respects_active_cooldown(self):
        llm_client._providers = [
            {
                "name": "tabcode",
                "base_url": "https://example.invalid",
                "api_key": "test-key",
                "model": "test-model",
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
        llm_client._provider_cooldown = {"tabcode": time.time() + 60}

        with self.assertRaises(llm_client.LLMProviderUnavailableError):
            llm_client._next_provider()

    def test_is_llm_auth_error_detects_http_401(self):
        request = httpx.Request("POST", "https://example.invalid/responses")
        response = httpx.Response(401, request=request)
        error = httpx.HTTPStatusError("401 Unauthorized", request=request, response=response)

        self.assertTrue(llm_client.is_llm_auth_error(error))

    def test_is_llm_transient_provider_error_detects_http_504(self):
        request = httpx.Request("POST", "https://example.invalid/chat/completions")
        response = httpx.Response(504, request=request)
        error = httpx.HTTPStatusError("504 Gateway Timeout", request=request, response=response)

        self.assertTrue(llm_client.is_llm_transient_provider_error(error))

    def test_init_providers_includes_secondary_openai_compatible_provider(self):
        llm_client._providers = []
        llm_client._provider_stats = {}
        llm_client._rate_limiters = {}
        llm_client._provider_cooldown = {}
        llm_client.LLM_USE_TABCODE = True
        llm_client.LLM_API_KEY = "primary-key"
        llm_client.LLM_BASE_URL = "https://primary.invalid/v1"
        llm_client.LLM_MODEL = "gpt-5.4"
        llm_client.LLM_PROTOCOL = "chat_completions"
        llm_client.LLM_RPM = 12
        llm_client.LLM_SECONDARY_ENABLED = True
        llm_client.LLM_SECONDARY_API_KEY = "secondary-key"
        llm_client.LLM_SECONDARY_BASE_URL = "https://secondary.invalid/v1"
        llm_client.LLM_SECONDARY_MODEL = "gpt-5.4"
        llm_client.LLM_SECONDARY_PROTOCOL = "chat_completions"
        llm_client.LLM_SECONDARY_RPM = 8

        llm_client._init_providers()

        names = [provider["name"] for provider in llm_client._providers]
        self.assertEqual(names, ["tabcode", "secondary"])
        self.assertEqual(llm_client._providers[0]["rpm"], 12)
        self.assertEqual(llm_client._providers[1]["rpm"], 8)
        self.assertFalse(llm_client._providers[0]["stream_chat_completions"])
        self.assertFalse(llm_client._providers[1]["stream_chat_completions"])


if __name__ == "__main__":
    unittest.main()
