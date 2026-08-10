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


if __name__ == "__main__":
    unittest.main()
