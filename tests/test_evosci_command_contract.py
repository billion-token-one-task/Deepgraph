"""The EvoScientist invocation must match the CLI that is actually installed.

Three call sites each hard-coded their own flag list against a CLI this
repository does not own. When EvoScientist dropped ``--auto-mode``, every one
of them started dying on a usage error before doing any work - and the only
thing that surfaced was ``evoscientist_review_failed``, with the CLI's own
error message written to a per-run log nobody reads. Measured 2026-08-10:
agenda 10's candidates 125 and 126 were both rejected this way after their
proposals had already been paid for.

The command now has one definition. These tests check it against the installed
binary's own help output, so the next interface change fails here instead of
silently rejecting every candidate in production.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from agents.evosci_requirements import (
    EVOSCI_HEADLESS_FLAGS,
    evosci_binary_path,
    evosci_command,
    evosci_installed,
)


class EvosciCommandShapeTests(unittest.TestCase):
    """Shape checks that run everywhere, installed or not."""

    def test_command_carries_workdir_prompt_and_headless_flags(self):
        command = evosci_command(workdir="/tmp/wd", prompt="do the thing")

        self.assertEqual(command[0], str(evosci_binary_path()))
        self.assertEqual(command[-2:], ["-p", "do the thing"])
        self.assertIn("--workdir", command)
        self.assertEqual(command[command.index("--workdir") + 1], "/tmp/wd")
        for flag in EVOSCI_HEADLESS_FLAGS:
            self.assertIn(flag, command)

    def test_the_dropped_flag_is_not_reintroduced(self):
        self.assertNotIn("--auto-mode", evosci_command(workdir="/tmp", prompt="x"))

    def test_every_call_site_builds_through_the_one_definition(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("agents/tier2_review_refine.py", "agents/novelty_verifier.py"):
            source = (root / name).read_text(encoding="utf-8")
            self.assertIn("evosci_command(", source, name)
            self.assertNotIn("--auto-approve", source, f"{name} rebuilt the flag list")


class EvosciInstalledContractTests(unittest.TestCase):
    """Check the flags against the binary on this machine, when there is one."""

    def setUp(self):
        if not evosci_installed():
            self.skipTest("EvoScientist is not installed on this host")

    def test_installed_cli_accepts_every_flag_we_pass(self):
        result = subprocess.run(
            [str(evosci_binary_path()), *EVOSCI_HEADLESS_FLAGS, "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode,
            0,
            "installed EvoScientist rejected our headless flags:\n"
            + (result.stdout or "") + (result.stderr or ""),
        )

    def test_help_text_documents_each_option_we_pass(self):
        result = subprocess.run(
            [str(evosci_binary_path()), "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        help_text = (result.stdout or "") + (result.stderr or "")
        # Option words only; bare values such as "cli" are arguments to them.
        for flag in ("--workdir", "-p", *(f for f in EVOSCI_HEADLESS_FLAGS if f.startswith("-"))):
            self.assertIn(flag, help_text, f"{flag} is gone from the installed CLI")

class EvosciRouteSelectionTests(unittest.TestCase):
    """EvoScientist must be handed a route this deployment actually uses.

    The primary LLM slot is gated behind LLM_USE_TABCODE and llm_client drops it
    from its provider list when that flag is off. _build_evosci_env preferred it
    anyway, so EvoScientist received a base_url that answers 502 and every
    review died on APIConnectionError - after starting up, so the failure looked
    like a review verdict rather than a misconfiguration.
    """

    def _route(self, *, use_tabcode):
        from unittest import mock

        from agents import novelty_verifier

        with (
            mock.patch.object(novelty_verifier, "LLM_USE_TABCODE", use_tabcode),
            mock.patch.object(novelty_verifier, "LLM_API_KEY", "primary-key"),
            mock.patch.object(novelty_verifier, "LLM_BASE_URL", "https://primary.invalid"),
            mock.patch.object(novelty_verifier, "LLM_MODEL", "primary-model"),
            mock.patch.object(novelty_verifier, "LLM_PROTOCOL", "chat_completions"),
            mock.patch.object(novelty_verifier, "LLM_SECONDARY_API_KEY", "secondary-key"),
            mock.patch.object(novelty_verifier, "LLM_SECONDARY_BASE_URL", "https://secondary.invalid"),
            mock.patch.object(novelty_verifier, "LLM_SECONDARY_MODEL", "secondary-model"),
            mock.patch.object(novelty_verifier, "LLM_SECONDARY_PROTOCOL", "chat_completions"),
            # This test is about which slot is chosen, not about streaming
            # capability; treat the chosen route as usable so the
            # streaming fall-through stays out of the way.
            mock.patch.object(novelty_verifier, "_supports_streaming", return_value=True),
            mock.patch.object(novelty_verifier, "_write_evosci_config", return_value="/tmp/xdg"),
        ):
            return novelty_verifier._build_evosci_env(Path("/tmp/wd"))

    def test_disabled_primary_is_not_offered(self):
        env = self._route(use_tabcode=False)
        self.assertEqual(env["CUSTOM_OPENAI_BASE_URL"], "https://secondary.invalid")

    def test_enabled_primary_is_still_preferred(self):
        env = self._route(use_tabcode=True)
        self.assertEqual(env["CUSTOM_OPENAI_BASE_URL"], "https://primary.invalid")

class StreamingCapabilityTests(unittest.TestCase):
    """EvoScientist needs a route that streams; nothing checked that it did.

    LangChain streams unconditionally and raises "No generations found in
    stream" on an empty one. Measured 2026-08-10: the sora2 relay answers a
    streaming request with Content-Type text/event-stream and a zero-byte body
    for every model it serves, while returning correct non-streaming
    completions - so reachability, model availability and a successful
    handshake all looked fine and the review still could not run. A third
    configured provider did stream and was never offered.
    """

    def test_an_empty_event_stream_is_not_streaming_support(self):
        from unittest import mock

        from agents import novelty_verifier

        class Empty:
            def read(self): return b""
            def __enter__(self): return self
            def __exit__(self, *a): return False

        with mock.patch.object(novelty_verifier.urllib.request, "urlopen", return_value=Empty()):
            self.assertFalse(
                novelty_verifier._supports_streaming(
                    {"base_url": "https://relay.invalid", "api_key": "k", "model": "m"}
                )
            )

    def test_real_sse_chunks_count_as_streaming_support(self):
        from unittest import mock

        from agents import novelty_verifier

        class Chunks:
            def read(self): return b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n'
            def __enter__(self): return self
            def __exit__(self, *a): return False

        with mock.patch.object(novelty_verifier.urllib.request, "urlopen", return_value=Chunks()):
            self.assertTrue(
                novelty_verifier._supports_streaming(
                    {"base_url": "https://relay.invalid", "api_key": "k", "model": "m"}
                )
            )

    def test_a_route_without_credentials_is_not_probed(self):
        from agents import novelty_verifier

        self.assertFalse(novelty_verifier._supports_streaming(None))
        self.assertFalse(novelty_verifier._supports_streaming({"base_url": "", "api_key": ""}))

    def test_builder_falls_through_to_a_streaming_provider(self):
        from unittest import mock

        from agents import novelty_verifier

        working = {
            "api_key": "novita-key",
            "base_url": "https://works.invalid",
            "model": "streams",
            "protocol": "chat_completions",
        }
        with (
            mock.patch.object(novelty_verifier, "LLM_USE_TABCODE", False),
            mock.patch.object(novelty_verifier, "LLM_SECONDARY_API_KEY", "k"),
            mock.patch.object(novelty_verifier, "LLM_SECONDARY_BASE_URL", "https://silent.invalid"),
            mock.patch.object(novelty_verifier, "LLM_SECONDARY_MODEL", "m"),
            mock.patch.object(novelty_verifier, "LLM_SECONDARY_PROTOCOL", "chat_completions"),
            mock.patch.object(novelty_verifier, "_supports_streaming", side_effect=lambda r, **k: r == working),
            mock.patch.object(novelty_verifier, "_configured_streaming_routes", return_value=[working]),
            mock.patch.object(novelty_verifier, "_write_evosci_config", return_value="/tmp/xdg"),
        ):
            env = novelty_verifier._build_evosci_env(Path("/tmp/wd"))
        self.assertEqual(env["CUSTOM_OPENAI_BASE_URL"], "https://works.invalid")

    def test_a_streaming_secondary_is_left_alone(self):
        """Fall-through only happens when the configured route cannot stream."""

        from unittest import mock

        from agents import novelty_verifier

        with (
            mock.patch.object(novelty_verifier, "LLM_USE_TABCODE", False),
            mock.patch.object(novelty_verifier, "LLM_SECONDARY_API_KEY", "k"),
            mock.patch.object(novelty_verifier, "LLM_SECONDARY_BASE_URL", "https://secondary.invalid"),
            mock.patch.object(novelty_verifier, "LLM_SECONDARY_MODEL", "m"),
            mock.patch.object(novelty_verifier, "LLM_SECONDARY_PROTOCOL", "chat_completions"),
            mock.patch.object(novelty_verifier, "_supports_streaming", return_value=True),
            mock.patch.object(novelty_verifier, "_configured_streaming_routes",
                              side_effect=AssertionError("must not fall through")),
            mock.patch.object(novelty_verifier, "_write_evosci_config", return_value="/tmp/xdg"),
        ):
            env = novelty_verifier._build_evosci_env(Path("/tmp/wd"))
        self.assertEqual(env["CUSTOM_OPENAI_BASE_URL"], "https://secondary.invalid")


if __name__ == "__main__":
    unittest.main()
