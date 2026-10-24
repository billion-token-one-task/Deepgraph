"""What _run_experiment returns and logs when the runner cannot start or times out.

Characterization tests: the full result, key order included, is part of the
contract, because callers persist it.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents import validation_loop as vl

KEYS = ["status", "duration", "error", "failure_type", "final_results_present", "command_tokens", "log_path",
        "benchmark_env"]


class _TimesOut:
    def __init__(self, *args, **kwargs):
        self.killed = False
        self.returncode = None

    def wait(self, timeout=None):
        if not self.killed:
            raise subprocess.TimeoutExpired(cmd="train.py", timeout=timeout)
        return -9

    def kill(self):
        self.killed = True


class LaunchFailureTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.workdir = Path(tmp.name)
        self.code_dir = self.workdir / "code"
        self.code_dir.mkdir()
        (self.code_dir / "train.py").write_text("print('x')\n")
        for name, value in (("_runner_contract_violations", []), ("_repair_generated_runner_json_literals", []),
                            ("_benchmark_env_for_execution", {"DEEPGRAPH_BENCHMARK": "1"})):
            patcher = mock.patch.object(vl, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(vl, "RUNTIME_PYTHON", sys.executable)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self):
        return vl._run_experiment(self.workdir, self.code_dir, 60)

    def test_a_launcher_error_is_reported_with_its_message(self):
        with mock.patch.object(vl.subprocess, "Popen", side_effect=OSError("runner missing")):
            result = self._run()
        self.assertEqual(list(result), KEYS)
        self.assertEqual(
            {k: v for k, v in result.items() if k != "duration"},
            {
                "status": "crash",
                "error": "runner missing",
                "failure_type": "launcher_error",
                "final_results_present": False,
                "command_tokens": [sys.executable, "train.py"],
                "log_path": str(self.workdir / "run.log"),
                "benchmark_env": {"DEEPGRAPH_BENCHMARK": "1"},
            },
        )
        log = (self.workdir / "run.log").read_text()
        self.assertRegex(log, r"^Experiment launcher failed after \d+\.\ds\n")
        self.assertTrue(log.endswith(f"Command: {[sys.executable, 'train.py']}\nError: runner missing\n"))

    def test_a_timeout_kills_the_runner_and_is_reported(self):
        with mock.patch.object(vl.subprocess, "Popen", _TimesOut):
            result = self._run()
        self.assertEqual(list(result), KEYS)
        self.assertEqual((result["error"], result["failure_type"], result["final_results_present"]),
                         ("timeout", "timeout", False))
        log = (self.workdir / "run.log").read_text()
        self.assertRegex(log, r"^Experiment timed out after \d+\.\ds\nCommand: ")
        self.assertIn("timed out after", log.splitlines()[-1])


if __name__ == "__main__":
    unittest.main()
