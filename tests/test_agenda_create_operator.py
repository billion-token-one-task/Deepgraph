"""The agenda spec an operator hands in must be validated, not trusted.

An agenda is a budget holder and a scope: a malformed one either cannot be
funded or silently widens what the ideation layer is allowed to propose. The
create path therefore goes through the same contract validation as every other
agenda, and the four fields that decide spending -- budget, GPU hours,
concurrency, backend allowlist -- must survive the round trip exactly.
"""

import json
import unittest
from pathlib import Path

from contracts.agenda import ResearchAgenda
from contracts.base import ContractValidationError

SPEC = Path("/home/ec2-user/scratch/deepgraph-lanes-20260825/"
            "agenda-low-baseline-20260825.json")


class AgendaSpecContractTest(unittest.TestCase):
    def _spec(self, **overrides):
        base = {
            "name": "spec-fixture",
            "description": "fixture",
            "focus": ["a"],
            "token_budget": 1000,
            "gpu_hours_budget": 1.0,
            "max_concurrency": 1,
            "backend_allowlist": ["cpu", "llm"],
        }
        base.update(overrides)
        return base

    def test_a_spec_without_a_name_is_refused(self):
        agenda = ResearchAgenda(**self._spec(name=""))
        with self.assertRaises(ContractValidationError):
            agenda.validate()

    def test_spending_fields_survive_the_round_trip(self):
        agenda = ResearchAgenda(**self._spec(token_budget=3_000_000,
                                             gpu_hours_budget=8.0,
                                             max_concurrency=1,
                                             backend_allowlist=["cpu", "llm", "ssh_gpu"]))
        agenda.validate()
        self.assertEqual(agenda.token_budget, 3_000_000)
        self.assertEqual(agenda.gpu_hours_budget, 8.0)
        self.assertEqual(agenda.max_concurrency, 1)
        self.assertEqual(agenda.backend_allowlist, ["cpu", "llm", "ssh_gpu"])

    def test_focus_is_normalised_to_strings(self):
        agenda = ResearchAgenda(**self._spec(focus=["a", "b"]))
        agenda.validate()
        self.assertEqual(agenda.focus, ["a", "b"])


class LowBaselineSpecTest(unittest.TestCase):
    """The shipped spec has to stay executable by a registered runner."""

    def setUp(self):
        if not SPEC.exists():
            self.skipTest("task-local agenda spec is not present on this host")
        self.spec = json.loads(SPEC.read_text(encoding="utf-8"))

    def test_it_validates(self):
        agenda = ResearchAgenda(**self.spec)
        agenda.validate()
        self.assertTrue(agenda.name)

    def test_it_rejects_what_v1_cannot_execute(self):
        rejected = " ".join(self.spec["reject"]["keywords"]).lower()
        for unrunnable in ("weight update", "architecture change",
                           "structure-from-motion", "speculative decoding"):
            self.assertIn(unrunnable, rejected)

    def test_it_carries_the_measurement_floor_that_made_the_last_verdict_real(self):
        requirements = " ".join(
            self.spec["required_output"]["measurement_requirements"]).lower()
        self.assertIn("equal generation budget", requirements)
        self.assertIn("200 evaluation samples", requirements)
        self.assertIn("raw predictions", requirements)


if __name__ == "__main__":
    unittest.main()
