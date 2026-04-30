import tempfile
import unittest
from pathlib import Path
from unittest import mock

from db import database
from orchestrator import gpu_scheduler


class GpuSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test.db"
        self.old_db_path = database.DB_PATH
        self.old_database_url = database.DATABASE_URL
        self.old_gpu_mode = gpu_scheduler.GPU_MODE
        self.old_gpu_visible_devices = list(gpu_scheduler.GPU_VISIBLE_DEVICES)
        self.old_gpu_remote_ssh_host = gpu_scheduler.GPU_REMOTE_SSH_HOST
        self.old_gpu_remote_ssh_user = gpu_scheduler.GPU_REMOTE_SSH_USER
        self.old_gpu_remote_ssh_port = gpu_scheduler.GPU_REMOTE_SSH_PORT
        self.old_gpu_remote_ssh_password = gpu_scheduler.GPU_REMOTE_SSH_PASSWORD
        self.old_gpu_remote_base_dir = gpu_scheduler.GPU_REMOTE_BASE_DIR
        self.old_gpu_remote_python = gpu_scheduler.GPU_REMOTE_PYTHON
        if hasattr(database._local, "conn"):
            try:
                database._local.conn.close()
            except Exception:
                pass
        database._local.conn = None
        if hasattr(database._local, "pg_conn"):
            try:
                database._local.pg_conn.close()
            except Exception:
                pass
            database._local.pg_conn = None
        database.DB_PATH = self.db_path
        database.DATABASE_URL = ""
        database.init_db()
        database.execute("INSERT INTO deep_insights (id, tier, title) VALUES (1, 2, 'GPU Insight')")
        database.execute(
            "INSERT INTO experiment_runs (id, deep_insight_id, status, workdir) VALUES (1, 1, 'pending', ?)",
            (str(Path(self.tmpdir.name) / 'run1'),),
        )
        database.commit()

    def tearDown(self):
        gpu_scheduler.stop()
        if hasattr(database._local, "conn"):
            try:
                database._local.conn.close()
            except Exception:
                pass
        database._local.conn = None
        if hasattr(database._local, "pg_conn"):
            try:
                database._local.pg_conn.close()
            except Exception:
                pass
            database._local.pg_conn = None
        database.DATABASE_URL = self.old_database_url
        gpu_scheduler.GPU_MODE = self.old_gpu_mode
        gpu_scheduler.GPU_VISIBLE_DEVICES = self.old_gpu_visible_devices
        gpu_scheduler.GPU_REMOTE_SSH_HOST = self.old_gpu_remote_ssh_host
        gpu_scheduler.GPU_REMOTE_SSH_USER = self.old_gpu_remote_ssh_user
        gpu_scheduler.GPU_REMOTE_SSH_PORT = self.old_gpu_remote_ssh_port
        gpu_scheduler.GPU_REMOTE_SSH_PASSWORD = self.old_gpu_remote_ssh_password
        gpu_scheduler.GPU_REMOTE_BASE_DIR = self.old_gpu_remote_base_dir
        gpu_scheduler.GPU_REMOTE_PYTHON = self.old_gpu_remote_python
        database.DB_PATH = self.old_db_path
        self.tmpdir.cleanup()

    def test_queue_run_creates_gpu_job(self):
        workers = gpu_scheduler.register_default_workers()
        job_id = gpu_scheduler.queue_run(
            insight_id=1,
            run_id=1,
            resource_class="gpu_small",
            priority=1,
            vram_required_gb=16,
        )
        job = database.fetchone("SELECT * FROM gpu_jobs WHERE id=?", (job_id,))
        self.assertIsNotNone(job)
        self.assertTrue(workers)
        self.assertEqual(job["resource_class"], "gpu_small")

    def test_register_ssh_workers(self):
        gpu_scheduler.GPU_MODE = "ssh"
        gpu_scheduler.GPU_VISIBLE_DEVICES = ["0", "1"]
        gpu_scheduler.GPU_REMOTE_SSH_HOST = "gpu.example.com"
        gpu_scheduler.GPU_REMOTE_SSH_USER = "root"
        gpu_scheduler.GPU_REMOTE_SSH_PORT = 55860
        gpu_scheduler.GPU_REMOTE_SSH_PASSWORD = "secret"
        gpu_scheduler.GPU_REMOTE_BASE_DIR = "/root/deepgraph-remote-worker"
        gpu_scheduler.GPU_REMOTE_PYTHON = "python"

        workers = gpu_scheduler.register_default_workers()

        self.assertEqual(len(workers), 2)
        first = database.fetchone("SELECT * FROM gpu_workers WHERE id=?", ("ssh:gpu.example.com:gpu0",))
        self.assertIsNotNone(first)
        self.assertIn('"backend": "ssh"', first["metadata"])
        self.assertIn('"ssh_host": "gpu.example.com"', first["metadata"])

    def test_run_job_bundle_failure_does_not_overwrite_completed_experiment(self):
        workers = gpu_scheduler.register_default_workers()
        worker = workers[0]
        job_id = gpu_scheduler.queue_run(
            insight_id=1,
            run_id=1,
            resource_class="gpu_small",
            priority=1,
            vram_required_gb=16,
        )
        database.execute(
            """
            INSERT INTO auto_research_jobs (deep_insight_id, status, stage, experiment_run_id)
            VALUES (1, 'queued_gpu', 'queued', 1)
            """
        )
        database.commit()
        job = database.fetchone("SELECT * FROM gpu_jobs WHERE id=?", (job_id,))

        def _fake_validation_loop(run_id, execution_context=None):
            database.execute(
                """
                UPDATE experiment_runs
                SET status='completed', hypothesis_verdict='confirmed', effect_pct=12.5
                WHERE id=?
                """,
                (run_id,),
            )
            database.commit()
            return {"run_id": run_id, "verdict": "confirmed"}

        with (
            mock.patch.object(gpu_scheduler, "run_validation_loop", side_effect=_fake_validation_loop),
            mock.patch.object(gpu_scheduler, "process_completed_run"),
            mock.patch.object(gpu_scheduler, "collect_run_artifacts", return_value=[]),
            mock.patch.object(gpu_scheduler, "generate_submission_bundle", return_value={"error": "latex failed"}),
            mock.patch.object(gpu_scheduler, "log_metrics"),
            mock.patch.object(gpu_scheduler, "log_artifact"),
        ):
            gpu_scheduler._run_job(job, worker)

        run = database.fetchone("SELECT status, hypothesis_verdict FROM experiment_runs WHERE id=1")
        gpu_job = database.fetchone("SELECT status, error_message FROM gpu_jobs WHERE id=?", (job_id,))
        auto_job = database.fetchone("SELECT status, stage, last_error FROM auto_research_jobs WHERE deep_insight_id=1")

        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["hypothesis_verdict"], "confirmed")
        self.assertEqual(gpu_job["status"], "completed")
        self.assertIn("latex failed", gpu_job["error_message"])
        self.assertEqual(auto_job["status"], "completed")
        self.assertEqual(auto_job["stage"], "closed_loop_complete")
        self.assertIn("latex failed", auto_job["last_error"])


if __name__ == "__main__":
    unittest.main()
