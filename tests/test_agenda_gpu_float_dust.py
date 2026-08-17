"""Float residue must not read as a negative GPU balance.

GPU hours are reserved and released as floats. Paired add/subtract sequences do
not cancel exactly in IEEE754, so on 2026-08-17 agenda 11 sat at
gpu_hours_reserved = -3.7566666666877246e-06 -- thirteen milliseconds of
negative GPU. ResearchAgenda.validate() refused the record outright, the
auto-research loop could not load its agenda, and idea generation stopped with
"agenda GPU accounting cannot be negative".

The guard still has to catch a real double-release, which is negative by minutes
or hours, not microseconds.
"""

import unittest

from contracts.agenda import GPU_HOURS_FLOAT_DUST, ResearchAgenda
from contracts.base import ContractValidationError


def _agenda(**overrides) -> ResearchAgenda:
    fields = {
        "name": "gpu-float-dust-fixture",
        "focus": ["ml"],
        "token_budget": 1000,
        "gpu_hours_budget": 100.0,
        "max_concurrency": 1,
        "backend_allowlist": ["llm"],
    }
    fields.update(overrides)
    return ResearchAgenda(**fields)


class GpuFloatDustTests(unittest.TestCase):
    def test_the_exact_value_that_stopped_the_loop_now_validates(self):
        agenda = _agenda(gpu_hours_reserved=-3.7566666666877246e-06)
        agenda.validate()
        self.assertEqual(agenda.gpu_hours_reserved, 0.0)

    def test_dust_is_snapped_on_both_accounting_fields(self):
        agenda = _agenda(gpu_hours_spent=-1e-9, gpu_hours_reserved=-1e-7)
        agenda.validate()
        self.assertEqual(agenda.gpu_hours_spent, 0.0)
        self.assertEqual(agenda.gpu_hours_reserved, 0.0)

    def test_a_real_negative_balance_is_still_refused(self):
        # A double-release of a half-hour job, not float residue.
        agenda = _agenda(gpu_hours_reserved=-0.5)
        with self.assertRaises(ContractValidationError):
            agenda.validate()

    def test_the_threshold_is_below_any_real_unit_of_gpu_accounting(self):
        # 3.6 seconds. A GPU job that short is not something the ledger tracks.
        self.assertLess(GPU_HOURS_FLOAT_DUST * 3600, 5.0)

    def test_positive_values_are_untouched(self):
        agenda = _agenda(gpu_hours_spent=2.0585855286661126, gpu_hours_reserved=0.25)
        agenda.validate()
        self.assertAlmostEqual(agenda.gpu_hours_spent, 2.0585855286661126)
        self.assertAlmostEqual(agenda.gpu_hours_reserved, 0.25)


if __name__ == "__main__":
    unittest.main()
