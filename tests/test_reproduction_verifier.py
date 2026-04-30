import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.reproduction_verifier import run_reproduction_check
from tests.temp_utils import temporary_workdir


class FakeReproductionDb:
    def __init__(self, run):
        self.run = run

    def fetchone(self, sql, params=()):
        if "FROM experiment_runs" in sql:
            return self.run
        return None


class ReproductionVerifierTests(unittest.TestCase):
    def test_reproduction_check_runs_command_and_writes_artifact(self):
        with temporary_workdir() as workdir:
            fake_db = FakeReproductionDb({
                "id": 44,
                "workdir": str(workdir),
            })
            command = [sys.executable, "-c", "print('repro ok')"]

            with patch("agents.reproduction_verifier.db", fake_db):
                result = run_reproduction_check(44, commands=[command])

            self.assertEqual(result["status"], "ok")
            path = workdir / "artifacts" / "results" / "reproduction_check.json"
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["checks"][0]["exit_code"], 0)
            self.assertIn("repro ok", payload["checks"][0]["stdout_tail"])


if __name__ == "__main__":
    unittest.main()
