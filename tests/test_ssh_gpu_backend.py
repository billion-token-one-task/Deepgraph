import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import ssh_gpu_backend


class SshGpuBackendTests(unittest.TestCase):
    def _worker(self):
        return {
            "id": "ssh:gpu.example.com:gpu1",
            "metadata": {
                "backend": "ssh",
                "ssh_user": "root",
                "ssh_host": "gpu.example.com",
                "ssh_port": 22,
                "visible_device": "1",
                "remote_base_dir": "/remote/deepgraph",
                "python_bin": "python",
            },
        }

    def test_remote_launcher_marks_and_cleans_one_run_process_group(self):
        script = ssh_gpu_backend._remote_launcher_script(
            run_id=7,
            remote_code_dir="/remote/deepgraph/runs/run_7/code",
            remote_workdir="/remote/deepgraph/runs/run_7",
            visible_device="1",
            command_tokens=["python", "train.py", "--name", "a b"],
        )

        self.assertIn("export DEEPGRAPH_RUN_ID=7", script)
        self.assertIn("export CUDA_VISIBLE_DEVICES=1", script)
        self.assertIn("export DEEPGRAPH_BENCHMARK_MODEL=", script)
        self.assertIn("export DEEPGRAPH_BENCHMARK_MAX_EXAMPLES=", script)
        self.assertIn("export DEEPGRAPH_BENCHMARK_SEEDS=", script)
        self.assertIn(".deepgraph_run_7.pgid", script)
        self.assertIn("setsid", script)
        self.assertIn("kill -TERM -- \"-$pgid\"", script)
        self.assertIn("exec python train.py --name", script)
        self.assertIn("a b", script)

    def test_run_ssh_sends_script_over_stdin_to_avoid_double_expansion(self):
        with mock.patch.object(ssh_gpu_backend, "_run_subprocess") as run_subprocess:
            run_subprocess.return_value = subprocess.CompletedProcess(["ssh"], 0, "ok", "")
            ssh_gpu_backend._run_ssh(self._worker(), "for x in 1; do\r\n echo X=$x\r\n done", timeout=12)

        kwargs = run_subprocess.call_args.kwargs
        cmd = run_subprocess.call_args.args[0]
        self.assertEqual(cmd[-1], "bash -s")
        self.assertEqual(kwargs["input_text"], "for x in 1; do\n echo X=$x\n done")
        self.assertEqual(kwargs["timeout"], 12)

    def test_run_subprocess_uses_binary_stdin_for_remote_scripts(self):
        completed = subprocess.CompletedProcess(["ssh"], 0, b"ok\n", b"")
        with mock.patch.object(ssh_gpu_backend.subprocess, "run", return_value=completed) as run:
            result = ssh_gpu_backend._run_subprocess(["ssh", "host", "bash -s"], worker=self._worker(), input_text="echo ok\n")

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["input"], b"echo ok\n")
        self.assertFalse(kwargs["text"])
        self.assertEqual(result.stdout, "ok\n")

    def test_cleanup_remote_run_processes_uses_run_specific_paths(self):
        calls = []

        def _fake_run_ssh(worker, remote_script, timeout=None):
            calls.append(remote_script)
            return subprocess.CompletedProcess(["ssh"], 0, "", "")

        with mock.patch.object(ssh_gpu_backend, "_run_ssh", side_effect=_fake_run_ssh):
            ssh_gpu_backend.cleanup_remote_run_processes(
                worker=self._worker(),
                run_id=7,
                remote_workdir="/remote/deepgraph/runs/run_7",
            )

        self.assertEqual(len(calls), 1)
        script = calls[0]
        self.assertIn("/remote/deepgraph/runs/run_7/.deepgraph_run_7.pgid", script)
        self.assertIn("/remote/deepgraph/runs/run_7/.deepgraph_exec_run_7.sh", script)
        self.assertIn("pgrep -f \"$launcher\"", script)
        self.assertIn("for cwd_link in /proc/[0-9]*/cwd", script)
        self.assertIn("\"$remote_root\"|\"$remote_root\"/*", script)
        self.assertNotIn("run_8", script)

    def test_remote_dependency_install_logs_and_keeps_requirements_order(self):
        calls = []

        def _fake_run_ssh(worker, remote_script, timeout=None):
            calls.append(remote_script)
            return subprocess.CompletedProcess(["ssh"], 0, "ok", "")

        with (
            mock.patch.object(ssh_gpu_backend, "GPU_REMOTE_AUTO_PIP_INSTALL", True),
            mock.patch.object(ssh_gpu_backend, "_run_ssh", side_effect=_fake_run_ssh),
        ):
            ssh_gpu_backend._install_remote_repo_deps(
                worker=self._worker(),
                remote_code_dir="/remote/deepgraph/runs/run_7/code",
                remote_python="python",
                remote_workdir="/remote/deepgraph/runs/run_7",
            )

        self.assertEqual(len(calls), 1)
        script = calls[0]
        self.assertIn(".deepgraph_remote_setup.log", script)
        self.assertIn("optional pip bootstrap failed; continuing to requirements", script)
        self.assertNotIn("-U pip setuptools wheel", script)
        self.assertLess(script.index("requirements-experiment.txt"), script.index("requirements.txt"))
        self.assertLess(script.index("requirements.txt"), script.index("pip install -e ."))

    def test_remote_sha256_retries_empty_output_and_has_python_fallback(self):
        calls = []

        def _fake_run_ssh(worker, remote_script, timeout=None):
            calls.append(remote_script)
            if len(calls) == 1:
                return subprocess.CompletedProcess(["ssh"], 0, "", "")
            return subprocess.CompletedProcess(["ssh"], 0, "abc123  /remote/file.tar.gz\n", "")

        with (
            mock.patch.object(ssh_gpu_backend, "_run_ssh", side_effect=_fake_run_ssh),
            mock.patch.object(ssh_gpu_backend.time, "sleep"),
        ):
            digest = ssh_gpu_backend._remote_sha256(self._worker(), "/remote/file.tar.gz")

        self.assertEqual(digest, "abc123")
        self.assertIn("python3 -c", calls[0])
        self.assertIn("sha256sum", calls[0])

    def test_benchmark_env_prefers_locked_manifest_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            spec_dir = workdir / "spec"
            spec_dir.mkdir()
            (spec_dir / "proxy_config.json").write_text(
                """
                {
                  "benchmark_model": "Qwen/Qwen2.5-14B-Instruct",
                  "benchmark_max_examples_per_seed": 128,
                  "benchmark_seeds": 5,
                  "benchmark_manifest": {
                    "full_benchmark_stage": {
                      "models": ["Qwen/Qwen2.5-7B-Instruct"],
                      "seeds": [0, 1, 2, 3, 4]
                    }
                  }
                }
                """,
                encoding="utf-8",
            )

            env = ssh_gpu_backend.benchmark_env_from_workdir(workdir)

        self.assertEqual(env["DEEPGRAPH_BENCHMARK_MODEL"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(env["DEEPGRAPH_BENCHMARK_MAX_EXAMPLES"], "128")
        self.assertEqual(env["DEEPGRAPH_BENCHMARK_SEEDS"], "5")

    def test_run_remote_experiment_uses_contract_benchmark_env(self):
        scripts = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            code_dir = workdir / "code"
            spec_dir = workdir / "spec"
            code_dir.mkdir()
            spec_dir.mkdir()
            (spec_dir / "proxy_config.json").write_text(
                """
                {
                  "benchmark_model": "Qwen/Qwen2.5-14B-Instruct",
                  "benchmark_manifest": {
                    "full_benchmark_stage": {"models": ["Qwen/Qwen2.5-7B-Instruct"]}
                  }
                }
                """,
                encoding="utf-8",
            )

            def _run_ssh(worker, remote_script, timeout=None):
                scripts.append(remote_script)
                return subprocess.CompletedProcess(["ssh"], 0, "ok", "")

            with (
                mock.patch.object(ssh_gpu_backend, "cleanup_remote_run_processes"),
                mock.patch.object(ssh_gpu_backend, "sync_workdir_to_remote"),
                mock.patch.object(ssh_gpu_backend, "sync_workdir_from_remote"),
                mock.patch.object(ssh_gpu_backend, "_install_remote_repo_deps"),
                mock.patch.object(ssh_gpu_backend, "_run_ssh", side_effect=_run_ssh),
            ):
                ssh_gpu_backend.run_remote_experiment(
                    worker=self._worker(),
                    run_id=7,
                    local_workdir=workdir,
                    local_code_dir=code_dir,
                    time_budget=60,
                    command_tokens=["python", "train.py"],
                    local_python="python",
                )

        launcher_script = scripts[-1]
        self.assertIn("export DEEPGRAPH_BENCHMARK_MODEL=Qwen/Qwen2.5-7B-Instruct", launcher_script)
        self.assertNotIn("Qwen/Qwen2.5-14B-Instruct", launcher_script)

    def test_run_remote_experiment_cleans_and_syncs_on_local_timeout(self):
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            code_dir = workdir / "code"
            code_dir.mkdir()

            def _cleanup(**kwargs):
                events.append(("cleanup", kwargs["run_id"]))

            def _sync_to_remote(**kwargs):
                events.append(("sync_to", kwargs["remote_workdir"]))

            def _install(**kwargs):
                events.append(("install", kwargs["remote_workdir"]))

            def _run_ssh(worker, remote_script, timeout=None):
                events.append(("run", timeout))
                raise subprocess.TimeoutExpired(["ssh"], timeout=timeout)

            def _sync_from_remote(**kwargs):
                events.append(("sync_from", kwargs["remote_workdir"]))

            with (
                mock.patch.object(ssh_gpu_backend, "cleanup_remote_run_processes", side_effect=_cleanup),
                mock.patch.object(ssh_gpu_backend, "sync_workdir_to_remote", side_effect=_sync_to_remote),
                mock.patch.object(ssh_gpu_backend, "_install_remote_repo_deps", side_effect=_install),
                mock.patch.object(ssh_gpu_backend, "_run_ssh", side_effect=_run_ssh),
                mock.patch.object(ssh_gpu_backend, "sync_workdir_from_remote", side_effect=_sync_from_remote),
            ):
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    ssh_gpu_backend.run_remote_experiment(
                        worker=self._worker(),
                        run_id=7,
                        local_workdir=workdir,
                        local_code_dir=code_dir,
                        time_budget=60,
                        command_tokens=["python", "train.py"],
                        local_python="python",
                    )

        self.assertEqual(
            [name for name, _ in events],
            ["cleanup", "sync_to", "install", "run", "cleanup", "sync_from", "cleanup"],
        )


if __name__ == "__main__":
    unittest.main()
