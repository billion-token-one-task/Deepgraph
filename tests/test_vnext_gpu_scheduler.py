import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
