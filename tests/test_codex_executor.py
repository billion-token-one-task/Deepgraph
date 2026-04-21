import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents import codex_executor


class CodexExecutorTests(unittest.TestCase):
    def test_run_codex_iteration_persists_thread_and_prefers_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "train.py").write_text("print('x')\n", encoding="utf-8")

            session_path = workdir / "codex" / "session.json"
            session_path.parent.mkdir(parents=True, exist_ok=True)
            session_path.write_text(json.dumps({"thread_id": "thread-123"}), encoding="utf-8")

            def _fake_run(cmd, cwd, env, timeout, capture_output, text):
                output_path = Path(cmd[cmd.index("-o") + 1])
                output_path.write_text(
                    json.dumps(
                        {
                            "summary": "resumed codex turn",
                            "files_changed": ["train.py"],
                            "commands_run": ["sed -n '1,20p' train.py"],
                            "validation_status": "passed",
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="",
                    stderr="",
                )

            import subprocess

            with (
                mock.patch.object(codex_executor, "codex_binary", return_value="/bin/codex"),
                mock.patch.object(codex_executor.subprocess, "run", side_effect=_fake_run) as run,
            ):
                result = codex_executor.run_codex_iteration(
                    workdir=workdir,
                    code_dir=code_dir,
                    iteration=2,
                    method_desc="Method",
                    best_so_far=0.8,
                    baseline=0.7,
                    history=[],
                    proxy={},
                    success_criteria={"metric_name": "acc"},
                    experimental_plan={},
                    evidence_plan={},
                    supervisor_plan={"mode": "refine"},
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["session_mode"], "resume")
        self.assertEqual(run.call_count, 1)
        invoked = run.call_args[0][0]
        self.assertIn("resume", invoked)
        self.assertIn("thread-123", invoked)


if __name__ == "__main__":
    unittest.main()
