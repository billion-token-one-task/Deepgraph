"""The ledger must record which model answered, not which one we asked for.

Measured on 2026-09-03: this gateway substitutes silently. Three consecutive
requests for gemini-3.8-flash-high were answered by gemini-3.8-flash, and
gemini-3.7-flash-high by gemini-3.7-flash-exp-b. Nothing in the request path
can detect that; only the response says so. A ledger whose stated purpose is
that a reader can check what produced a result cannot record the request and
call it provenance.
"""

import json
import unittest.mock

import agents.llm_client as llm_client
from meta_harness.llm_routing import RouteObservation, RouteUsage


class _Resp:
    status_code = 200

    def __init__(self, body):
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        return None


class _Client:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, *a, **k):
        return _Resp({
            # What the gateway actually served, which is not what was asked for.
            "model": "gemini-3.8-flash",
            "choices": [{"message": {"content": "a long enough answer to pass"}}],
            "usage": {"total_tokens": 9, "prompt_tokens": 6},
        })


def test_chat_completions_reports_the_model_that_answered():
    provider = {
        "name": "gw",
        "base_url": "https://gw.example/v1",
        "api_key": "k",
        "model": "gemini-3.8-flash-high",
        "protocol": "chat_completions",
        "stream_chat_completions": False,
    }
    with unittest.mock.patch.object(llm_client.httpx, "Client", _Client):
        result = llm_client._call_chat_completions(provider, "sys", "user", 128)
    served = result[5]
    assert served == "gemini-3.8-flash"
    assert served != provider["model"], "the substitution must remain visible"


def test_route_usage_defaults_keep_existing_callers_valid():
    usage = RouteUsage(1, 2, None)
    assert usage.served_model is None


def test_observation_carries_both_names():
    observation = RouteObservation(
        agenda_id=1, idea_id=1, role="proposer", provider="gw",
        model="gemini-3.8-flash-high", served_model="gemini-3.8-flash",
        model_family="gemini", prompt_version="proposer_v1",
        input_tokens=6, output_tokens=3, cost_usd=None,
        status="succeeded", failure_reason=None, reservation_id=None,
    )
    assert observation.model != observation.served_model


class _StreamResp:
    status_code = 200
    text = ""

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield 'data: {"model": "gemini-3.8-flash", "choices": [{"delta": {"content": "a long enough answer"}}]}'
        yield 'data: {"model": "gemini-3.8-flash", "usage": {"total_tokens": 9, "prompt_tokens": 6}}'
        yield "data: [DONE]"


class _StreamClient:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def stream(self, *a, **k):
        class _Ctx:
            def __enter__(self_inner):
                return _StreamResp()

            def __exit__(self_inner, *a):
                return False

        return _Ctx()


def test_streaming_chat_completions_also_reports_the_served_model():
    """The live route streams, so the non-streaming path alone proves nothing."""
    provider = {
        "name": "gw",
        "base_url": "https://gw.example/v1",
        "api_key": "k",
        "model": "gemini-3.8-flash-high",
        "protocol": "chat_completions",
        "stream_chat_completions": True,
    }
    with unittest.mock.patch.object(llm_client.httpx, "Client", _StreamClient):
        result = llm_client._call_chat_completions(provider, "sys", "user", 128)
    assert result[5] == "gemini-3.8-flash"
