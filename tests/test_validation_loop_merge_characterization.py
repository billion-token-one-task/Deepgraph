"""Pin what the multi-model benchmark merge returns today, so it can be rewritten safely.

Characterization tests: the expected values in
fixtures/validation_loop_merge_cases.json were produced by the implementation as
it stood when these tests were added, quirks included, so that a rewrite
cannot change them by accident:
- a row with a negative count still weighs 1 in the sum while lowering the
  total, so rows of 0.4 and 0.6 with counts None and -2 merge to 1.0;
- summaries are filtered to dicts but model_results are not, so a model
  without a summary shifts the ids of every model after it (the third model's
  seeds come out labelled with the second, failed model's id).
Fixing either is a behaviour change and belongs in its own pull request.
"""

import copy
import json
import unittest
from pathlib import Path

from agents.validation_loop import _merge_model_benchmark_summaries, _weighted_merge_method_rows

CASES = json.loads((Path(__file__).parent / "fixtures" / "validation_loop_merge_cases.json").read_text())


class MergeModelBenchmarkSummariesTests(unittest.TestCase):
    def test_every_case_merges_exactly_as_recorded(self):
        for case in CASES["merge"]:
            with self.subTest(case["name"]):
                results = copy.deepcopy(case["model_results"])
                merged = _merge_model_benchmark_summaries(results, case["criteria_metric_name"])
                # Serialized, so key order and float values must match too:
                # merged summaries are written to artifact files.
                self.assertEqual(json.dumps(merged), json.dumps(case["expected"]))
                self.assertEqual(results, case["model_results"], "the inputs must not be modified")

    def test_merged_summary_is_a_copy(self):
        case = CASES["merge"][0]
        results = copy.deepcopy(case["model_results"])
        merged = _merge_model_benchmark_summaries(results, case["criteria_metric_name"])
        merged["per_method"]["cand"]["score"] = -1
        merged["seed_results"][0]["seed"] = -1
        self.assertEqual(results, case["model_results"])


class WeightedMergeMethodRowsTests(unittest.TestCase):
    def test_every_row_set_merges_exactly_as_recorded(self):
        for i, case in enumerate(CASES["weighted"]):
            with self.subTest(i):
                merged = _weighted_merge_method_rows(copy.deepcopy(case["rows"]))
                self.assertEqual(json.dumps(merged), json.dumps(case["expected"]))


if __name__ == "__main__":
    unittest.main()
