"""Do not buy a full benchmark for a candidate the pilot already showed is empty.

The evidence audit refuses a run whose arm generated nothing, but it only gets
to look after the full benchmark has been paid for. The pilot writes the same
artifacts and shows the same thing, at 200 examples instead of the full n.

Runs 235, 236 and 238 each bought a complete benchmark on a candidate whose
pilot was already 200/200 empty. Three GPU benchmarks, three refusals, one
defect -- and the M3 batch has ten candidates that all share it.

Verified against production artifacts on 2026-08-20:

    run 238  ->  'tab_cot_singlepass 200/200 empty'
    run 236  ->  'code_prompting_2shot 200/200 empty'
    run 237  ->  ''      (real measurement, proceeds)
    run 219  ->  ''      (real measurement, proceeds)
    run 9999 ->  ''      (does not exist, proceeds)

The check is read-only and non-fatal on purpose. An unreadable or missing pilot
artifact returns "", the benchmark proceeds exactly as before, and the audit
stays the gate that actually refuses. This one only ever saves money; it must
never be the reason a good run stops. The tests below spend most of their
attention on that half.
"""

import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from scripts.auto_advance import _pilot_arm_that_measured_nothing


class _Workdir:
    def __init__(self, tmp):
        self.path = Path(tmp)
        (self.path / "results").mkdir(parents=True, exist_ok=True)

    def write(self, rows):
        with (self.path / "results" / "raw_predictions.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def fetchone(self, _sql, _params=None):
        return {"workdir": str(self.path)}


def _rows(*, candidate_blank, n=10):
    out = []
    for index in range(n):
        out.append({"method": "baseline", "prediction": "#### 4"})
        out.append(
            {
                "method": "candidate",
                "prediction": "" if index < candidate_blank else "#### 4",
            }
        )
    return out


class PilotCheckTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.work = _Workdir(self._tmp.name)

    def _check(self):
        with patch("scripts.auto_advance.db.fetchone", self.work.fetchone):
            return _pilot_arm_that_measured_nothing(1)

    def test_an_empty_candidate_arm_is_named(self):
        self.work.write(_rows(candidate_blank=10))
        result = self._check()
        self.assertIn("candidate", result)
        self.assertIn("10/10", result)

    def test_an_empty_baseline_arm_is_named_too(self):
        rows = _rows(candidate_blank=0)
        for row in rows:
            if row["method"] == "baseline":
                row["prediction"] = ""
        self.work.write(rows)
        self.assertIn("baseline", self._check())


class ItNeverStopsAGoodRunTests(unittest.TestCase):
    """The half that matters: a saving that costs a real run is not a saving."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.work = _Workdir(self._tmp.name)

    def _check(self):
        with patch("scripts.auto_advance.db.fetchone", self.work.fetchone):
            return _pilot_arm_that_measured_nothing(1)

    def test_a_real_measurement_proceeds(self):
        self.work.write(_rows(candidate_blank=0))
        self.assertEqual(self._check(), "")

    def test_a_weak_but_present_candidate_proceeds(self):
        # Scoring badly is a result. Only generating nothing is not.
        self.work.write(_rows(candidate_blank=4))
        self.assertEqual(self._check(), "")

    def test_a_missing_artifact_proceeds(self):
        self.assertEqual(self._check(), "")

    def test_a_corrupt_artifact_proceeds(self):
        (self.work.path / "results" / "raw_predictions.jsonl").write_text("{not json")
        self.assertEqual(self._check(), "")

    def test_an_unknown_run_proceeds(self):
        with patch("scripts.auto_advance.db.fetchone", lambda *a, **k: None):
            self.assertEqual(_pilot_arm_that_measured_nothing(9999), "")

    def test_a_database_failure_proceeds(self):
        def _down(*_a, **_k):
            raise RuntimeError("db down")

        with patch("scripts.auto_advance.db.fetchone", _down):
            self.assertEqual(_pilot_arm_that_measured_nothing(1), "")


if __name__ == "__main__":
    unittest.main()
