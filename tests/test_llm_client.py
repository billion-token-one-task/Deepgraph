import time
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone

import httpx

from agents import llm_client


class LlmClientCooldownTests(unittest.TestCase):
    def setUp(self):
        self.old_providers = list(llm_client._providers)
        self.old_provider_idx = llm_client._provider_idx
        self.old_provider_stats = dict(llm_client._provider_stats)
        self.old_rate_limiters = dict(llm_client._rate_limiters)
        self.old_provider_cooldown = dict(llm_client._provider_cooldown)
        self.old_prompt_cache_unsupported = set(llm_client._prompt_cache_unsupported)
        self.old_llm_use_tabcode = llm_client.LLM_USE_TABCODE
        self.old_llm_extra_providers_json = llm_client.LLM_EXTRA_PROVIDERS_JSON
        self.old_llm_api_key = llm_client.LLM_API_KEY
        self.old_llm_base_url = llm_client.LLM_BASE_URL
        self.old_llm_model = llm_client.LLM_MODEL
        self.old_llm_protocol = llm_client.LLM_PROTOCOL
        self.old_llm_rpm = llm_client.LLM_RPM
        self.old_llm_prompt_cache_enabled = llm_client.LLM_PROMPT_CACHE_ENABLED
        self.old_llm_prompt_cache_key = llm_client.LLM_PROMPT_CACHE_KEY
        self.old_llm_prompt_cache_retention = llm_client.LLM_PROMPT_CACHE_RETENTION
        self.old_secondary_enabled = llm_client.LLM_SECONDARY_ENABLED
        self.old_secondary_api_key = llm_client.LLM_SECONDARY_API_KEY
        self.old_secondary_base_url = llm_client.LLM_SECONDARY_BASE_URL
        self.old_secondary_model = llm_client.LLM_SECONDARY_MODEL
        self.old_secondary_protocol = llm_client.LLM_SECONDARY_PROTOCOL
        self.old_secondary_rpm = llm_client.LLM_SECONDARY_RPM
        self.old_transient_retries = llm_client.LLM_TRANSIENT_RETRIES
        self.old_transient_backoff_seconds = llm_client.LLM_TRANSIENT_BACKOFF_SECONDS
        self.old_transient_cooldown_seconds = llm_client.LLM_TRANSIENT_COOLDOWN_SECONDS

    def tearDown(self):
        llm_client._providers = self.old_providers
        llm_client._provider_idx = self.old_provider_idx
        llm_client._provider_stats = self.old_provider_stats
        llm_client._rate_limiters = self.old_rate_limiters
        llm_client._provider_cooldown = self.old_provider_cooldown
        llm_client._prompt_cache_unsupported = self.old_prompt_cache_unsupported
        llm_client.LLM_USE_TABCODE = self.old_llm_use_tabcode
        llm_client.LLM_EXTRA_PROVIDERS_JSON = self.old_llm_extra_providers_json
        llm_client.LLM_API_KEY = self.old_llm_api_key
        llm_client.LLM_BASE_URL = self.old_llm_base_url
        llm_client.LLM_MODEL = self.old_llm_model
        llm_client.LLM_PROTOCOL = self.old_llm_protocol
        llm_client.LLM_RPM = self.old_llm_rpm
        llm_client.LLM_PROMPT_CACHE_ENABLED = self.old_llm_prompt_cache_enabled
        llm_client.LLM_PROMPT_CACHE_KEY = self.old_llm_prompt_cache_key
        llm_client.LLM_PROMPT_CACHE_RETENTION = self.old_llm_prompt_cache_retention
        llm_client.LLM_SECONDARY_ENABLED = self.old_secondary_enabled
        llm_client.LLM_SECONDARY_API_KEY = self.old_secondary_api_key
        llm_client.LLM_SECONDARY_BASE_URL = self.old_secondary_base_url
        llm_client.LLM_SECONDARY_MODEL = self.old_secondary_model
        llm_client.LLM_SECONDARY_PROTOCOL = self.old_secondary_protocol
        llm_client.LLM_SECONDARY_RPM = self.old_secondary_rpm
        llm_client.LLM_TRANSIENT_RETRIES = self.old_transient_retries
        llm_client.LLM_TRANSIENT_BACKOFF_SECONDS = self.old_transient_backoff_seconds
        llm_client.LLM_TRANSIENT_COOLDOWN_SECONDS = self.old_transient_cooldown_seconds

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

    def test_is_llm_transient_provider_error_detects_connection_refused_message(self):
        error = RuntimeError("connect failed: Connection refused")

        self.assertTrue(llm_client.is_llm_transient_provider_error(error))

    def test_scoped_short_response_retains_usage_and_does_not_retry(self):
        """The real provider adapter must not erase billed short responses."""
        provider = {
            "name": "provider-a",
            "base_url": "https://example.invalid",
            "api_key": "test-key",
            "model": "model-a",
            "model_family": "family-a",
            "protocol": "chat_completions",
        }
        llm_client._providers = [provider]
        llm_client._provider_stats = {
            "provider-a": {
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

        class Reservation:
            reservation_id = 77

        class Ledger:
            instances = []

            def __init__(self, _grant_id):
                self.settled = []
                self.released = []
                Ledger.instances.append(self)

            def remaining(self, *, agenda_id):
                self.agenda_id = agenda_id
                return 8000

            def reserve(self, **kwargs):
                self.reserved = kwargs
                return Reservation()

            def settle(self, reservation_id, **kwargs):
                self.settled.append((reservation_id, kwargs))

            def release(self, reservation_id, *, reason):
                self.released.append((reservation_id, reason))

        class Repository:
            instances = []

            def __init__(self):
                self.observations = []
                Repository.instances.append(self)

            def save_route_observation(self, observation):
                self.observations.append(observation)

            def load_active_cooldowns(self, _route_ids, *, now):
                return {}

            def save_cooldown(self, _route, *, until, failure_category):
                raise AssertionError((until, failure_category))

        grant = {
            "id": 31,
            "agenda_id": 11,
            "idea_id": 105,
            "decision_packet_id": 9,
            "stage": "pilot",
            "token_cap": 8000,
            "gpu_class": "none",
            "max_gpu_hours": 0,
            "backend_allowlist_json": '["llm"]',
            "artifact_requirements_json": '["tagged_response"]',
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat(),
            "grant_reason": "bounded test",
            "idempotency_key": "grant-test",
            "status": "active",
            "reservation_id": 5,
        }
        with (
            unittest.mock.patch.object(llm_client, "_init_providers"),
            unittest.mock.patch.object(
                llm_client,
                "configured_role_route_policy",
                return_value={
                    "provider-a": {
                        "model": "model-a",
                        "model_family": "family-a",
                        "prompt_version": "proposer-v1",
                    }
                },
            ),
            unittest.mock.patch.object(
                llm_client,
                "_call_provider",
                return_value=("short", 8001, 0, 100, 0.02),
            ) as call_provider,
            unittest.mock.patch(
                "meta_harness.grant_usage.GrantUsageLedger",
                Ledger,
            ),
            unittest.mock.patch(
                "meta_harness.repository.MetaHarnessRepository",
                Repository,
            ),
            unittest.mock.patch(
                "db.database.fetchone",
                return_value=grant,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "failed_attempt_usage_exceeded_reserved_cap",
            ):
                llm_client.call_llm_for_role(
                    "system",
                    "user",
                    agenda_id=11,
                    idea_id=105,
                    role="proposer",
                    stage="pilot",
                    resource_grant_id=31,
                    operation="experiment_forge.capability_scaffold_tagged_repair",
                    idempotency_key="tagged:t1",
                    prompt_version="proposer-v1",
                    max_tokens=8000,
                    total_token_cap=8000,
                    max_route_attempts=1,
                )

        self.assertEqual(call_provider.call_count, 1)
        router_ledger = Ledger.instances[-1]
        self.assertEqual(router_ledger.settled[0][1]["tokens_used"], 8000)
        self.assertFalse(router_ledger.released)
        observation = Repository.instances[-1].observations[0]
        self.assertEqual(observation.input_tokens + observation.output_tokens, 8001)
        self.assertEqual(
            observation.failure_reason,
            "failed_attempt_usage_exceeded_reserved_cap",
        )

    def test_init_providers_includes_secondary_openai_compatible_provider(self):
        llm_client._providers = []
        llm_client._provider_stats = {}
        llm_client._rate_limiters = {}
        llm_client._provider_cooldown = {}
        llm_client.LLM_USE_TABCODE = True
        llm_client.LLM_EXTRA_PROVIDERS_JSON = ""
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

    def test_init_providers_includes_extra_openai_compatible_providers(self):
        llm_client._providers = []
        llm_client._provider_stats = {}
        llm_client._rate_limiters = {}
        llm_client._provider_cooldown = {}
        llm_client.LLM_USE_TABCODE = False
        llm_client.LLM_SECONDARY_ENABLED = False
        llm_client.LLM_EXTRA_PROVIDERS_JSON = """
        [
          {
            "name": "sora-gpt",
            "base_url": "https://sora.invalid/v1",
            "api_key": "extra-key-1",
            "model": "gpt-5.4",
            "protocol": "chat_completions",
            "rpm": 20
          },
          {
            "name": "sora-gpt",
            "base_url": "https://sora.invalid/v1",
            "api_key": "extra-key-2",
            "model": "glm-5.1",
            "protocol": "chat_completions",
            "stream_chat_completions": true
          }
        ]
        """

        llm_client._init_providers()

        names = [provider["name"] for provider in llm_client._providers]
        self.assertEqual(names, ["sora-gpt", "sora-gpt_2"])
        self.assertEqual(llm_client._providers[0]["model"], "gpt-5.4")
        self.assertEqual(llm_client._providers[0]["rpm"], 20)
        self.assertEqual(llm_client._providers[1]["model"], "glm-5.1")
        self.assertTrue(llm_client._providers[1]["stream_chat_completions"])

    def test_call_llm_retries_transient_error_before_cooling_provider(self):
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
        llm_client.LLM_TRANSIENT_RETRIES = 1
        llm_client.LLM_TRANSIENT_BACKOFF_SECONDS = 1
        llm_client.LLM_TRANSIENT_COOLDOWN_SECONDS = 60

        connect_error = httpx.ConnectError("Connection refused")
        with (
            unittest.mock.patch.object(llm_client, "_init_providers"),
            unittest.mock.patch.object(
                llm_client,
                "_call_provider",
                side_effect=[connect_error, ("valid enough response", 17, 0, 9, None)],
            ) as call_provider,
            unittest.mock.patch.object(llm_client.time, "sleep"),
        ):
            text, tokens = llm_client.call_llm("system", "user")

        self.assertEqual(text, "valid enough response")
        self.assertEqual(tokens, 17)
        self.assertEqual(call_provider.call_count, 2)
        self.assertEqual(llm_client._provider_cooldown.get("tabcode", 0), 0)

    def test_extract_responses_output_text_from_completed_object(self):
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "final "},
                        {"type": "output_text", "text": "answer"},
                    ],
                }
            ]
        }

        self.assertEqual(llm_client._extract_responses_output_text(response), "final answer")

    def test_gpt55_responses_payload_omits_max_output_tokens(self):
        captured_payloads = []

        class FakeStreamResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self):
                yield 'data: {"type":"response.output_text.delta","delta":"valid enough response"}'
                yield 'data: {"type":"response.completed","response":{"usage":{"total_tokens":7,"input_tokens":3}}}'
                yield 'data: [DONE]'

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, json, headers):
                captured_payloads.append(dict(json))
                return FakeStreamResponse()

        provider = {
            "name": "tabcode",
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
            "model": "gpt-5.5",
            "protocol": "responses",
        }

        with unittest.mock.patch.object(llm_client.httpx, "Client", FakeClient):
            text, tokens, cached, input_tokens, cost_usd = llm_client._call_responses_api(provider, "system", "user", 1234)

        self.assertEqual(text, "valid enough response")
        self.assertEqual(tokens, 7)
        self.assertEqual(cached, 0)
        self.assertEqual(input_tokens, 3)
        self.assertIsNone(cost_usd)
        self.assertEqual(captured_payloads[0]["model"], "gpt-5.5")
        self.assertNotIn("max_output_tokens", captured_payloads[0])

    def test_gpt55_chat_payload_omits_max_tokens(self):
        captured_payloads = []

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [{"message": {"content": "valid enough response"}}],
                    "usage": {"total_tokens": 5, "prompt_tokens": 2},
                }

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, json, headers):
                captured_payloads.append(dict(json))
                return FakeResponse()

        provider = {
            "name": "tabcode",
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
            "model": "openai/gpt-5.5",
            "protocol": "chat_completions",
            "stream_chat_completions": False,
        }

        with unittest.mock.patch.object(llm_client.httpx, "Client", FakeClient):
            text, tokens, cached, input_tokens, cost_usd = llm_client._call_chat_completions(provider, "system", "user", 1234)

        self.assertEqual(text, "valid enough response")
        self.assertEqual(tokens, 5)
        self.assertEqual(cached, 0)
        self.assertEqual(input_tokens, 2)
        self.assertIsNone(cost_usd)
        self.assertEqual(captured_payloads[0]["model"], "openai/gpt-5.5")
        self.assertNotIn("max_tokens", captured_payloads[0])

    def test_bounded_chat_400_makes_one_http_request_with_output_limit(self):
        payloads = []
        llm_client.LLM_PROMPT_CACHE_ENABLED = True

        class RejectedResponse:
            status_code = 400

            def raise_for_status(self):
                request = httpx.Request("POST", "https://example.invalid/chat")
                response = httpx.Response(400, request=request)
                raise httpx.HTTPStatusError(
                    "bad request",
                    request=request,
                    response=response,
                )

            def json(self):
                return {}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, json, headers):
                payloads.append(dict(json))
                return RejectedResponse()

        provider = {
            "name": "bounded-chat",
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
            "model": "gpt-5.4",
            "protocol": "chat_completions",
            "stream_chat_completions": False,
        }
        with (
            unittest.mock.patch.object(llm_client.httpx, "Client", FakeClient),
            self.assertRaises(httpx.HTTPStatusError),
        ):
            llm_client._call_chat_completions(
                provider,
                "system",
                "user",
                3000,
                strict_single_request=True,
            )

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["max_tokens"], 3000)
        self.assertNotIn("prompt_cache_key", payloads[0])
        self.assertNotIn("prompt_cache_retention", payloads[0])

    def test_bounded_responses_400_never_retries_without_output_limit(self):
        payloads = []
        llm_client.LLM_PROMPT_CACHE_ENABLED = True

        class RejectedStream:
            status_code = 400

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                request = httpx.Request("POST", "https://example.invalid/responses")
                response = httpx.Response(400, request=request)
                raise httpx.HTTPStatusError(
                    "bad request",
                    request=request,
                    response=response,
                )

            def iter_lines(self):
                return iter(())

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, json, headers):
                payloads.append(dict(json))
                return RejectedStream()

        provider = {
            "name": "bounded-responses",
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
            "model": "gpt-5.4",
            "protocol": "responses",
        }
        with (
            unittest.mock.patch.object(llm_client.httpx, "Client", FakeClient),
            self.assertRaises(httpx.HTTPStatusError),
        ):
            llm_client._call_responses_api(
                provider,
                "system",
                "user",
                3000,
                strict_single_request=True,
            )

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["max_output_tokens"], 3000)
        self.assertNotIn("prompt_cache_key", payloads[0])
        self.assertNotIn("prompt_cache_retention", payloads[0])

    def test_bounded_route_rejects_models_that_omit_output_limits(self):
        provider = {
            "name": "unbounded-model",
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
            "model": "gpt-5.5",
            "protocol": "responses",
        }
        with self.assertRaisesRegex(
            llm_client.LLMProviderUnavailableError,
            "requires an enforceable provider output limit",
        ):
            llm_client._call_responses_api(
                provider,
                "system",
                "user",
                3000,
                strict_single_request=True,
            )

    def test_responses_payload_includes_prompt_cache_options(self):
        captured_payloads = []
        llm_client.LLM_PROMPT_CACHE_ENABLED = True
        llm_client.LLM_PROMPT_CACHE_KEY = "DeepGraph Cache!"
        llm_client.LLM_PROMPT_CACHE_RETENTION = "24h"

        class FakeStreamResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self):
                yield 'data: {"type":"response.output_text.delta","delta":"valid enough response"}'
                yield 'data: {"type":"response.completed","response":{"usage":{"total_tokens":11,"input_tokens":7,"input_tokens_details":{"cached_tokens":3}}}}'
                yield 'data: [DONE]'

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, json, headers):
                captured_payloads.append(dict(json))
                return FakeStreamResponse()

        provider = {
            "name": "tabcode",
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
            "model": "gpt-5.4",
            "protocol": "responses",
        }

        with unittest.mock.patch.object(llm_client.httpx, "Client", FakeClient):
            text, tokens, cached, input_tokens, cost_usd = llm_client._call_responses_api(provider, "stable system", "user", 1234)

        self.assertEqual(text, "valid enough response")
        self.assertEqual(tokens, 11)
        self.assertEqual(cached, 3)
        self.assertEqual(input_tokens, 7)
        self.assertIsNone(cost_usd)
        self.assertIn("prompt_cache_key", captured_payloads[0])
        self.assertLessEqual(len(captured_payloads[0]["prompt_cache_key"]), 64)
        self.assertTrue(captured_payloads[0]["prompt_cache_key"].startswith("deepgraph-cache:"))
        self.assertEqual(captured_payloads[0]["prompt_cache_retention"], "24h")

    def test_chat_payload_includes_prompt_cache_options(self):
        captured_payloads = []
        llm_client.LLM_PROMPT_CACHE_ENABLED = True
        llm_client.LLM_PROMPT_CACHE_KEY = "deepgraph-test"
        llm_client.LLM_PROMPT_CACHE_RETENTION = "24h"

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [{"message": {"content": "valid enough response"}}],
                    "usage": {
                        "total_tokens": 9,
                        "prompt_tokens": 6,
                        "prompt_tokens_details": {"cached_tokens": 2},
                        "cost_usd": 0.012,
                    },
                }

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, json, headers):
                captured_payloads.append(dict(json))
                return FakeResponse()

        provider = {
            "name": "tabcode",
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
            "model": "gpt-5.4",
            "protocol": "chat_completions",
            "stream_chat_completions": False,
        }

        with unittest.mock.patch.object(llm_client.httpx, "Client", FakeClient):
            text, tokens, cached, input_tokens, cost_usd = llm_client._call_chat_completions(provider, "stable system", "user", 1234)

        self.assertEqual(text, "valid enough response")
        self.assertEqual(tokens, 9)
        self.assertEqual(cached, 2)
        self.assertEqual(input_tokens, 6)
        self.assertEqual(cost_usd, 0.012)
        self.assertIn("prompt_cache_key", captured_payloads[0])
        self.assertEqual(captured_payloads[0]["prompt_cache_retention"], "24h")

    def test_responses_prompt_cache_options_fallback_on_unsupported_400(self):
        captured_payloads = []
        llm_client.LLM_PROMPT_CACHE_ENABLED = True
        llm_client.LLM_PROMPT_CACHE_KEY = "deepgraph-test"
        llm_client.LLM_PROMPT_CACHE_RETENTION = "24h"
        llm_client._prompt_cache_unsupported = set()

        class FakeStreamResponse:
            def __init__(self, should_fail):
                self.should_fail = should_fail

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self):
                if self.should_fail:
                    request = httpx.Request("POST", "https://example.invalid/v1/responses")
                    response = httpx.Response(400, request=request)
                    raise httpx.HTTPStatusError("unknown parameter: prompt_cache_key", request=request, response=response)

            def iter_lines(self):
                yield 'data: {"type":"response.output_text.delta","delta":"valid enough response"}'
                yield 'data: {"type":"response.completed","response":{"usage":{"total_tokens":5,"input_tokens":4}}}'
                yield 'data: [DONE]'

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, json, headers):
                captured_payloads.append(dict(json))
                return FakeStreamResponse(should_fail=len(captured_payloads) == 1)

        provider = {
            "name": "tabcode",
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
            "model": "gpt-5.4",
            "protocol": "responses",
        }

        with unittest.mock.patch.object(llm_client.httpx, "Client", FakeClient):
            text, tokens, cached, input_tokens, cost_usd = llm_client._call_responses_api(provider, "stable system", "user", 1234)

        self.assertEqual(text, "valid enough response")
        self.assertEqual(tokens, 5)
        self.assertEqual(cached, 0)
        self.assertEqual(input_tokens, 4)
        self.assertIsNone(cost_usd)
        self.assertIn("prompt_cache_key", captured_payloads[0])
        self.assertNotIn("prompt_cache_key", captured_payloads[1])
        self.assertNotIn("prompt_cache_retention", captured_payloads[1])
        self.assertIn("tabcode", llm_client._prompt_cache_unsupported)


if __name__ == "__main__":
    unittest.main()
