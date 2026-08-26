from __future__ import annotations

import unittest

from agents.experiment_forge import build_proxy_config
from config import (
    EXPERIMENT_REAL_BENCHMARK_DATASET,
    EXPERIMENT_REAL_LLM_MODEL,
)


class ProxyContractSourceTests(unittest.TestCase):
    """The proxy names the contract the validation loop enforces.

    A materialized runner pinned to anything other than proxy_config's
    benchmark_dataset/benchmark_model is refused before it can produce a
    metric, so taking the deployment default there fails every candidate whose
    science is not the default's.
    """

    def _plan(self, **extra):
        plan = {
            "compute_budget": {"total_gpu_hours": 2},
            "benchmark_targets": [{"hf_dataset": "tasksource/bigbench"}],
        }
        plan.update(extra)
        return plan

    def test_an_explicit_contract_wins_over_the_deployment_default(self):
        proxy = build_proxy_config(
            self._plan(
                execution_requirements={
                    "dataset": {
                        "repository_id": "tasksource/bigbench",
                        "config": "object_counting",
                    },
                    "model": {"repository_id": "Qwen/Qwen2.5-1.5B-Instruct"},
                }
            )
        )
        self.assertEqual(proxy["benchmark_dataset"], "tasksource/bigbench")
        self.assertEqual(proxy["benchmark_dataset_config"], "object_counting")
        self.assertEqual(proxy["benchmark_model"], "Qwen/Qwen2.5-1.5B-Instruct")

    def test_a_plan_without_a_contract_keeps_the_deployment_default(self):
        proxy = build_proxy_config(self._plan())
        self.assertEqual(proxy["benchmark_dataset"], EXPERIMENT_REAL_BENCHMARK_DATASET)
        self.assertEqual(proxy["benchmark_model"], EXPERIMENT_REAL_LLM_MODEL)

    def test_a_partial_contract_only_overrides_what_it_names(self):
        proxy = build_proxy_config(
            self._plan(
                execution_requirements={
                    "dataset": {"repository_id": "tasksource/bigbench"},
                    "model": {},
                }
            )
        )
        self.assertEqual(proxy["benchmark_dataset"], "tasksource/bigbench")
        self.assertEqual(proxy["benchmark_model"], EXPERIMENT_REAL_LLM_MODEL)


if __name__ == "__main__":
    unittest.main()
