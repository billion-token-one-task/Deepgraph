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
        self.old_gpu_visible_devices = list(gpu_scheduler.GPU_VISIBLE_DEVICES)
        self.old_gpu_stale_recovery_poll_seconds = gpu_scheduler.GPU_STALE_RECOVERY_POLL_SECONDS
        self.old_last_recovery_check = gpu_scheduler._last_recovery_check
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
        gpu_scheduler.GPU_VISIBLE_DEVICES = self.old_gpu_visible_devices
        gpu_scheduler.GPU_STALE_RECOVERY_POLL_SECONDS = self.old_gpu_stale_recovery_poll_seconds
        gpu_scheduler._last_recovery_check = self.old_last_recovery_check
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

    def test_next_job_fails_recipe_blocked_run_without_launching(self):
        database.execute(
            """
            UPDATE experiment_runs
            SET status='failed', phase='recipe_blocked',
                error_message='Invalid benchmark: must remain blocked'
            WHERE id=1
            """
        )
        database.execute(
            """
            INSERT INTO auto_research_jobs (deep_insight_id, status, stage, experiment_run_id)
            VALUES (1, 'queued_gpu', 'gpu_scheduler', 1)
            """
        )
        job_id = gpu_scheduler.queue_run(
            insight_id=1,
            run_id=1,
            resource_class="gpu_small",
            priority=1,
            vram_required_gb=16,
        )

        job = gpu_scheduler._next_job()

        self.assertIsNone(job)
        queued = database.fetchone("SELECT status, error_message FROM gpu_jobs WHERE id=?", (job_id,))
        auto_job = database.fetchone("SELECT status, stage, last_error FROM auto_research_jobs WHERE deep_insight_id=1")
        self.assertEqual(queued["status"], "failed")
        self.assertIn("blocked", queued["error_message"])
        self.assertEqual(auto_job["status"], "failed")
        self.assertEqual(auto_job["stage"], "gpu_blocked")

    def test_recover_stale_running_job_requeues_after_restart(self):
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
            UPDATE gpu_jobs
            SET status='running', assigned_worker=?, started_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (worker["id"], job_id),
        )
        database.execute(
            """
            INSERT INTO auto_research_jobs (deep_insight_id, status, stage, experiment_run_id, assigned_worker)
            VALUES (1, 'running_gpu', 'gpu_scheduler', 1, ?)
            """,
            (worker["id"],),
        )
        database.commit()

        recovered = gpu_scheduler.recover_stale_running_jobs(workers)

        job = database.fetchone("SELECT status, assigned_worker, error_message FROM gpu_jobs WHERE id=?", (job_id,))
        auto_job = database.fetchone("SELECT status, stage, assigned_worker, last_note FROM auto_research_jobs WHERE deep_insight_id=1")
        self.assertEqual(recovered, 1)
        self.assertEqual(job["status"], "queued")
        self.assertIsNone(job["assigned_worker"])
        self.assertIn("stale running", job["error_message"])
        self.assertEqual(auto_job["status"], "queued_gpu")
        self.assertIsNone(auto_job["assigned_worker"])

    def test_periodic_recovery_runs_after_poll_interval(self):
        gpu_scheduler.GPU_STALE_RECOVERY_POLL_SECONDS = 30
        gpu_scheduler._last_recovery_check = 10.0

        with (
            mock.patch.object(gpu_scheduler.time, "time", return_value=20.0),
            mock.patch.object(gpu_scheduler, "recover_busy_workers_without_running_jobs") as recover_workers,
        ):
            recovered = gpu_scheduler._maybe_recover_stale_jobs()

        self.assertEqual(recovered, 0)
        recover_workers.assert_not_called()

        with (
            mock.patch.object(gpu_scheduler.time, "time", return_value=41.0),
            mock.patch.object(gpu_scheduler, "recover_busy_workers_without_running_jobs", return_value=1) as recover_workers,
        ):
            recovered = gpu_scheduler._maybe_recover_stale_jobs()

        self.assertEqual(recovered, 1)
        recover_workers.assert_called_once()

    def test_claim_worker_ignores_idle_worker_with_running_job(self):
        gpu_scheduler.GPU_VISIBLE_DEVICES = ["0"]
        with mock.patch.object(gpu_scheduler, "_local_gpu_inventory", return_value={}):
            workers = gpu_scheduler.register_default_workers()
            worker = workers[0]
            database.execute(
                """
                UPDATE gpu_workers
                SET status='idle'
                WHERE id=?
                """,
                (worker["id"],),
            )
            database.execute(
                """
                INSERT INTO gpu_jobs
                (deep_insight_id, experiment_run_id, status, assigned_worker, resource_class)
                VALUES (1, 1, 'running', ?, 'gpu_small')
                """,
                (worker["id"],),
            )
            database.commit()

            claimed = gpu_scheduler._claim_idle_worker({"vram_required_gb": 0})

        self.assertIsNone(claimed)

    def test_release_worker_stays_busy_when_another_job_is_running(self):
        gpu_scheduler.GPU_VISIBLE_DEVICES = ["0"]
        with mock.patch.object(gpu_scheduler, "_local_gpu_inventory", return_value={}):
            workers = gpu_scheduler.register_default_workers()
        worker = workers[0]
        database.execute(
            """
            INSERT INTO gpu_jobs
            (deep_insight_id, experiment_run_id, status, assigned_worker, resource_class)
            VALUES (1, 1, 'running', ?, 'gpu_small')
            """,
            (worker["id"],),
        )
        running_job_id = database.fetchone("SELECT MAX(id) AS id FROM gpu_jobs")["id"]
        database.execute("UPDATE gpu_workers SET status='busy' WHERE id=?", (worker["id"],))
        database.commit()

        gpu_scheduler._release_worker_if_no_running_jobs(worker["id"], finished_job_id=999)
        database.commit()
        busy = database.fetchone("SELECT status FROM gpu_workers WHERE id=?", (worker["id"],))
        self.assertEqual(busy["status"], "busy")

        database.execute("UPDATE gpu_jobs SET status='completed' WHERE id=?", (running_job_id,))
        gpu_scheduler._release_worker_if_no_running_jobs(worker["id"], finished_job_id=999)
        database.commit()
        idle = database.fetchone("SELECT status FROM gpu_workers WHERE id=?", (worker["id"],))
        self.assertEqual(idle["status"], "idle")

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

    def test_run_job_handles_none_validation_result(self):
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

        with (
            mock.patch.object(gpu_scheduler, "run_validation_loop", return_value=None),
            mock.patch.object(gpu_scheduler, "process_completed_run"),
            mock.patch.object(gpu_scheduler, "collect_run_artifacts", return_value=[]),
            mock.patch.object(gpu_scheduler, "generate_submission_bundle", return_value={"error": "no bundle"}),
            mock.patch.object(gpu_scheduler, "log_metrics"),
            mock.patch.object(gpu_scheduler, "log_artifact"),
        ):
            gpu_scheduler._run_job(job, worker)

        gpu_job = database.fetchone("SELECT status, error_message FROM gpu_jobs WHERE id=?", (job_id,))
        auto_job = database.fetchone("SELECT status, stage, last_note, last_error FROM auto_research_jobs WHERE deep_insight_id=1")

        self.assertEqual(gpu_job["status"], "completed")
        self.assertIn("no bundle", gpu_job["error_message"])
        self.assertEqual(auto_job["status"], "completed")
        self.assertEqual(auto_job["stage"], "closed_loop_complete")
        self.assertIn("verdict=failed", auto_job["last_note"])

    def test_run_job_uses_full_benchmark_completion_stage(self):
        workers = gpu_scheduler.register_default_workers()
        worker = workers[0]
        job_id = gpu_scheduler.queue_run(
            insight_id=1,
            run_id=1,
            resource_class="gpu_large",
            priority=3,
            vram_required_gb=40,
        )
        database.execute(
            """
            INSERT INTO auto_research_jobs (deep_insight_id, status, stage, experiment_run_id)
            VALUES (1, 'queued_gpu', 'benchmark_completion_required', 1)
            """
        )
        database.commit()
        job = database.fetchone("SELECT * FROM gpu_jobs WHERE id=?", (job_id,))

        def _fake_full_completion(run_id, execution_context=None):
            self.assertTrue(execution_context.get("full_benchmark"))
            database.execute(
                """
                UPDATE experiment_runs
                SET status='completed', hypothesis_verdict='confirmed', effect_pct=5.0
                WHERE id=?
                """,
                (run_id,),
            )
            database.commit()
            return {"run_id": run_id, "verdict": "confirmed", "full_benchmark_completed": True}

        with (
            mock.patch.object(gpu_scheduler, "run_full_benchmark_completion", side_effect=_fake_full_completion) as full_run,
            mock.patch.object(gpu_scheduler, "run_validation_loop") as validation_run,
            mock.patch.object(gpu_scheduler, "process_completed_run"),
            mock.patch.object(gpu_scheduler, "collect_run_artifacts", return_value=[]),
            mock.patch.object(gpu_scheduler, "generate_submission_bundle", return_value={"bundle_ids": [3]}),
            mock.patch.object(gpu_scheduler, "log_metrics"),
            mock.patch.object(gpu_scheduler, "log_artifact"),
        ):
            gpu_scheduler._run_job(job, worker)

        full_run.assert_called_once()
        validation_run.assert_not_called()
        gpu_job = database.fetchone("SELECT status FROM gpu_jobs WHERE id=?", (job_id,))
        self.assertEqual(gpu_job["status"], "completed")


if __name__ == "__main__":
    unittest.main()
