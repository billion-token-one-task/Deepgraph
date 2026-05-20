import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents import codex_executor


class CodexExecutorTests(unittest.TestCase):
    def test_iteration_agents_md_forbids_candidate_only_scoring_shortcuts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)

            path = codex_executor.write_iteration_agents_md(
                code_dir=code_dir,
                method_desc="Method",
                baseline=0.7,
                best_so_far=0.8,
                iteration=1,
                history=[],
                proxy={},
                success_criteria={"metric_name": "acc"},
                experimental_plan={},
                evidence_plan={},
                supervisor_plan={},
            )

            text = path.read_text(encoding="utf-8")

        self.assertIn("Do not change scoring, answer normalization", text)
        self.assertIn("Do not add dataset/example-specific lexical shortcuts", text)
        self.assertIn("upper_bound", text)

    def test_run_codex_iteration_persists_thread_and_prefers_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "train.py").write_text("print('x')\n", encoding="utf-8")

            session_path = workdir / "codex" / "session.json"
            session_path.parent.mkdir(parents=True, exist_ok=True)
            session_path.write_text(json.dumps({"thread_id": "thread-123"}), encoding="utf-8")

            def _fake_run(cmd, cwd, env, timeout, capture_output, text, **kwargs):
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

    def test_run_codex_iteration_starts_fresh_after_fairness_discard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "train.py").write_text("print('x')\n", encoding="utf-8")
            session_path = workdir / "codex" / "session.json"
            session_path.parent.mkdir(parents=True, exist_ok=True)
            session_path.write_text(json.dumps({"thread_id": "thread-123"}), encoding="utf-8")

            def _fake_run(cmd, cwd, env, timeout, capture_output, text, **kwargs):
                output_path = Path(cmd[cmd.index("-o") + 1])
                output_path.write_text(
                    json.dumps(
                        {
                            "summary": "fresh codex turn",
                            "files_changed": ["train.py"],
                            "commands_run": ["sed -n '1,20p' train.py"],
                            "validation_status": "passed",
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with (
                mock.patch.object(codex_executor, "codex_binary", return_value="/bin/codex"),
                mock.patch.object(codex_executor.subprocess, "run", side_effect=_fake_run) as run,
            ):
                result = codex_executor.run_codex_iteration(
                    workdir=workdir,
                    code_dir=code_dir,
                    iteration=3,
                    method_desc="Method",
                    best_so_far=0.8,
                    baseline=0.7,
                    history=[{"status": "discard", "description": "benchmark_fairness_risk: candidate-only canonicalizer"}],
                    proxy={},
                    success_criteria={"metric_name": "acc"},
                    experimental_plan={},
                    evidence_plan={},
                    supervisor_plan={"mode": "redirect"},
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["session_mode"], "fresh")
        invoked = run.call_args[0][0]
        self.assertNotIn("resume", invoked)
        self.assertIn("-C", invoked)

    def test_run_codex_iteration_starts_fresh_after_repeated_no_gain_discards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "train.py").write_text("print('x')\n", encoding="utf-8")
            session_path = workdir / "codex" / "session.json"
            session_path.parent.mkdir(parents=True, exist_ok=True)
            session_path.write_text(json.dumps({"thread_id": "thread-123"}), encoding="utf-8")

            def _fake_run(cmd, cwd, env, timeout, capture_output, text, **kwargs):
                output_path = Path(cmd[cmd.index("-o") + 1])
                output_path.write_text(
                    json.dumps(
                        {
                            "summary": "fresh codex turn",
                            "files_changed": ["train.py"],
                            "commands_run": ["sed -n '1,20p' train.py"],
                            "validation_status": "passed",
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            history = [
                {"status": "discard", "description": "judge=no_gain Metric did not improve; discard the change."},
                {"status": "discard", "description": "judge=no_gain Metric did not improve; discard the change."},
                {"status": "discard", "description": "judge=no_gain Metric did not improve; discard the change."},
            ]
            with (
                mock.patch.object(codex_executor, "codex_binary", return_value="/bin/codex"),
                mock.patch.object(codex_executor.subprocess, "run", side_effect=_fake_run) as run,
            ):
                result = codex_executor.run_codex_iteration(
                    workdir=workdir,
                    code_dir=code_dir,
                    iteration=4,
                    method_desc="Method",
                    best_so_far=0.8,
                    baseline=0.7,
                    history=history,
                    proxy={},
                    success_criteria={"metric_name": "acc"},
                    experimental_plan={},
                    evidence_plan={},
                    supervisor_plan={"mode": "redirect"},
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["session_mode"], "fresh")
        invoked = run.call_args[0][0]
        self.assertNotIn("resume", invoked)
        self.assertIn("-C", invoked)

    def test_run_codex_iteration_starts_fresh_after_kept_microtuning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "train.py").write_text("print('x')\n", encoding="utf-8")
            session_path = workdir / "codex" / "session.json"
            session_path.parent.mkdir(parents=True, exist_ok=True)
            session_path.write_text(json.dumps({"thread_id": "thread-123"}), encoding="utf-8")

            def _fake_run(cmd, cwd, env, timeout, capture_output, text, **kwargs):
                output_path = Path(cmd[cmd.index("-o") + 1])
                output_path.write_text(
                    json.dumps(
                        {
                            "summary": "fresh codex turn",
                            "files_changed": ["train.py"],
                            "commands_run": ["python -m py_compile train.py"],
                            "validation_status": "passed",
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            history = [
                {"status": "keep", "description": "Tightened yes/no zero-budget pass from 2 generated tokens to 1 token."},
                {"status": "discard", "description": "Metric did not improve; discard the change."},
                {"status": "keep", "description": "Easy non-yes/no zero-budget passes now cap at 12 tokens."},
            ]
            with (
                mock.patch.object(codex_executor, "codex_binary", return_value="/bin/codex"),
                mock.patch.object(codex_executor.subprocess, "run", side_effect=_fake_run) as run,
            ):
                result = codex_executor.run_codex_iteration(
                    workdir=workdir,
                    code_dir=code_dir,
                    iteration=6,
                    method_desc="Method",
                    best_so_far=0.8,
                    baseline=0.7,
                    history=history,
                    proxy={},
                    success_criteria={"metric_name": "acc"},
                    experimental_plan={},
                    evidence_plan={},
                    supervisor_plan={"mode": "redirect"},
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["session_mode"], "fresh")
        invoked = run.call_args[0][0]
        self.assertNotIn("resume", invoked)
        self.assertIn("-C", invoked)

    def test_run_codex_iteration_clears_stale_thread_after_fresh_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            code_dir = workdir / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / "train.py").write_text("print('x')\n", encoding="utf-8")
            session_path = workdir / "codex" / "session.json"
            session_path.parent.mkdir(parents=True, exist_ok=True)
            session_path.write_text(json.dumps({"thread_id": "stale-thread"}), encoding="utf-8")

            def _fake_run(cmd, cwd, env, timeout, capture_output, text, **kwargs):
                if "resume" in cmd:
                    return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="stale thread")
                output_path = Path(cmd[cmd.index("-o") + 1])
                output_path.write_text(
                    json.dumps(
                        {
                            "summary": "fresh fallback turn",
                            "files_changed": ["train.py"],
                            "commands_run": ["python -m py_compile train.py"],
                            "validation_status": "passed",
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with (
                mock.patch.object(codex_executor, "codex_binary", return_value="/bin/codex"),
                mock.patch.object(codex_executor.subprocess, "run", side_effect=_fake_run) as run,
            ):
                result = codex_executor.run_codex_iteration(
                    workdir=workdir,
                    code_dir=code_dir,
                    iteration=5,
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
            session = json.loads(session_path.read_text(encoding="utf-8"))

        self.assertTrue(result["ok"])
        self.assertEqual(result["session_mode"], "fresh_fallback")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(session["thread_id"], "")
        self.assertEqual(session["last_iteration"], 5)


if __name__ == "__main__":
    unittest.main()
