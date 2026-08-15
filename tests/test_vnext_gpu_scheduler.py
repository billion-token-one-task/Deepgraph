import json
import tempfile
import os
import unittest
from pathlib import Path
from unittest import mock

from db import database
from orchestrator import gpu_scheduler


class GpuSchedulerTimeoutPolicyTests(unittest.TestCase):
    def test_queue_run_preserves_zero_timeout_as_uncapped(self):
        with (
            mock.patch.object(gpu_scheduler.db, "init_db"),
            mock.patch.object(gpu_scheduler, "_effective_vram_required_gb", return_value=(24, None)),
            mock.patch.object(gpu_scheduler.db, "insert_returning_id", return_value=123) as insert,
            mock.patch.object(gpu_scheduler.db, "commit"),
            mock.patch.object(gpu_scheduler.db, "emit_pipeline_event"),
        ):
            gpu_scheduler.queue_run(
                insight_id=1,
                run_id=2,
                resource_class="gpu_large",
                timeout_s=0,
            )

        params = insert.call_args.args[1]
        self.assertEqual(params[5], 0)


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
        self.old_gpu_stale_recovery_poll_seconds = gpu_scheduler.GPU_STALE_RECOVERY_POLL_SECONDS
        self.old_last_recovery_check = gpu_scheduler._last_recovery_check
        self.old_env_visible_devices = os.environ.get("DEEPGRAPH_GPU_VISIBLE_DEVICES")
        os.environ["DEEPGRAPH_GPU_VISIBLE_DEVICES"] = "0"
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
        gpu_scheduler.GPU_MODE = "single_host"
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
        gpu_scheduler.GPU_STALE_RECOVERY_POLL_SECONDS = self.old_gpu_stale_recovery_poll_seconds
        gpu_scheduler._last_recovery_check = self.old_last_recovery_check
        if self.old_env_visible_devices is None:
            os.environ.pop("DEEPGRAPH_GPU_VISIBLE_DEVICES", None)
        else:
            os.environ["DEEPGRAPH_GPU_VISIBLE_DEVICES"] = self.old_env_visible_devices
        database.DB_PATH = self.old_db_path
        self.tmpdir.cleanup()

    def test_queue_run_creates_gpu_job(self):
        workers = gpu_scheduler.register_default_workers()
        job_id = gpu_scheduler.queue_run(
            insight_id=1,
            run_id=1,
            resource_grant_id=1,
            resource_class="gpu_small",
            priority=1,
            vram_required_gb=16,
        )
        job = database.fetchone("SELECT * FROM gpu_jobs WHERE id=?", (job_id,))
        self.assertIsNotNone(job)
        self.assertTrue(workers)
        self.assertEqual(job["resource_class"], "gpu_small")

    def test_queue_run_downshifts_gpu_large_vram_to_schedulable_worker(self):
        gpu_scheduler.GPU_VISIBLE_DEVICES = ["0"]
        inventory = {"0": {"gpu_model": "NVIDIA GeForce RTX 3090", "total_mem_gb": 24.0}}

        with mock.patch.object(gpu_scheduler, "_local_gpu_inventory", return_value=inventory):
            workers = gpu_scheduler.register_default_workers()
            job_id = gpu_scheduler.queue_run(
                insight_id=1,
                run_id=1,
                resource_class="gpu_large",
                priority=2,
                vram_required_gb=40,
            )
            job = database.fetchone("SELECT * FROM gpu_jobs WHERE id=?", (job_id,))
            claimed = gpu_scheduler._claim_idle_worker(job)

        self.assertTrue(workers)
        self.assertEqual(job["vram_required_gb"], 24.0)
        self.assertIn("Adjusted vram_required_gb", job["error_message"])
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["id"], workers[0]["id"])

    def test_register_default_workers_does_not_fabricate_local_gpu_without_inventory_or_visible_devices(self):
        os.environ.pop("DEEPGRAPH_GPU_VISIBLE_DEVICES", None)
        gpu_scheduler.GPU_VISIBLE_DEVICES = ["0"]

        with mock.patch.object(gpu_scheduler, "_local_gpu_inventory", return_value={}):
            workers = gpu_scheduler.register_default_workers()

        self.assertEqual(workers, [])

    def _write_legacy_gsm8k_manifest(self):
        workdir = Path(self.tmpdir.name) / "run1"
        spec_dir = workdir / "spec"
        spec_dir.mkdir(parents=True, exist_ok=True)
        (spec_dir / "benchmark_manifest.json").write_text(
            json.dumps(
                {
                    "benchmark_protocol": {
                        "dataset_protocols": [
                            {"name": "GSM8K", "hf_dataset": "openai/gsm8k", "task_family": "math_qa"}
                        ],
                        "full_benchmark_requirements": {"required_dataset_names": ["GSM8K"]},
                    }
                }
            ),
            encoding="utf-8",
        )

    def _write_legacy_mbpp_manifest(self):
        workdir = Path(self.tmpdir.name) / "run1"
        spec_dir = workdir / "spec"
        spec_dir.mkdir(parents=True, exist_ok=True)
        (spec_dir / "benchmark_manifest.json").write_text(
            json.dumps(
                {
                    "benchmark_protocol": {
                        "dataset_protocols": [
                            {"name": "MBPP", "hf_dataset": "google-research-datasets/mbpp", "task_family": "code_generation"}
                        ],
                        "full_benchmark_requirements": {"required_dataset_names": ["MBPP"]},
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_next_job_blocks_legacy_gsm8k_manifest_for_formal_run(self):
        self._write_legacy_gsm8k_manifest()
        database.execute(
            """
            UPDATE deep_insights
            SET title=?, proposed_method=?, experimental_plan=?
            WHERE id=1
            """,
            (
                "Formal code reasoning over Lean proof-state lattices",
                json.dumps({"name": "Verifier search", "definition": "Use Lean proof-state verifier feedback."}),
                json.dumps({"datasets": [{"name": "GSM8K"}]}),
            ),
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
            resource_grant_id=1,
            resource_class="gpu_small",
            priority=1,
            vram_required_gb=16,
        )

        job = gpu_scheduler._next_job()

        self.assertIsNone(job)
        queued = database.fetchone("SELECT status, error_message FROM gpu_jobs WHERE id=?", (job_id,))
        self.assertEqual(queued["status"], "failed")
        self.assertIn("legacy benchmark manifest uses GSM8K", queued["error_message"])

    def test_next_job_blocks_legacy_gsm8k_manifest_for_agent_workflow_run(self):
        self._write_legacy_gsm8k_manifest()
        database.execute(
            """
            UPDATE deep_insights
            SET title=?, problem_statement=?, proposed_method=?, experimental_plan=?
            WHERE id=1
            """,
            (
                "Self-evolving agents and workflow self-optimization as typed stochastic program synthesis",
                "Agent workflows need executable tool-use and workflow benchmarks, not math word problems.",
                json.dumps({"name": "WorkflowSynth", "definition": "Optimize executable policy code for agent workflows."}),
                json.dumps({"datasets": [{"name": "GSM8K"}]}),
            ),
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
        self.assertEqual(queued["status"], "failed")
        self.assertIn("agent_workflow_optimization", queued["error_message"])

    def test_next_job_blocks_legacy_gsm8k_manifest_for_physical_spatial_run(self):
        self._write_legacy_gsm8k_manifest()
        database.execute(
            """
            UPDATE deep_insights
            SET title=?, problem_statement=?, proposed_method=?, experimental_plan=?
            WHERE id=1
            """,
            (
                "Causal Scene Competence and Physical-Spatial Benchmark Validity",
                "Physical-spatial visual reasoning needs causal scene and intervention benchmarks, not math word problems.",
                json.dumps({"name": "Scene Intervention", "definition": "Audit support/contact intervention identifiability."}),
                json.dumps({"datasets": [{"name": "GSM8K"}]}),
            ),
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
        self.assertEqual(queued["status"], "failed")
        self.assertIn("physical_spatial_reasoning", queued["error_message"])

    def test_next_job_blocks_legacy_mbpp_manifest_for_molecular_equivariant_run(self):
        self._write_legacy_mbpp_manifest()
        database.execute(
            """
            UPDATE deep_insights
            SET title=?, problem_statement=?, proposed_method=?, experimental_plan=?
            WHERE id=1
            """,
            (
                "Equivariant message-passing control and conformation sampling",
                "GEOM-QM9 molecular conformation and 3D equivariant dynamics need molecular benchmarks, not code generation.",
                json.dumps({"name": "Equivariant Dynamics", "definition": "Train EGNN-style coordinate denoising on GEOM-QM9."}),
                json.dumps({"datasets": [{"name": "MBPP"}]}),
            ),
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
        self.assertEqual(queued["status"], "failed")
        self.assertIn("generic benchmark MBPP", queued["error_message"])
        self.assertIn("molecular_equivariant_dynamics", queued["error_message"])

    def test_next_job_allows_legacy_mbpp_manifest_for_formal_code_run(self):
        self._write_legacy_mbpp_manifest()
        database.execute(
            """
            UPDATE deep_insights
            SET title=?, proposed_method=?, experimental_plan=?
            WHERE id=1
            """,
            (
                "Formal code reasoning with verifier-guided Python repair",
                json.dumps({"name": "Verifier repair", "definition": "Use code reasoning and program repair."}),
                json.dumps({"datasets": [{"name": "MBPP"}]}),
            ),
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

        self.assertIsNotNone(job)
        self.assertEqual(job["id"], job_id)

    def test_next_job_allows_legacy_gsm8k_manifest_for_math_prm_run(self):
        self._write_legacy_gsm8k_manifest()
        database.execute(
            """
            UPDATE deep_insights
            SET title=?, proposed_method=?, experimental_plan=?
            WHERE id=1
            """,
            (
                "Process Reward Models as Bellman Factorizations for math reasoning",
                json.dumps({"name": "PRM Bellman", "definition": "Use process reward models over chain-of-thought."}),
                json.dumps({"datasets": [{"name": "GSM8K"}]}),
            ),
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

        self.assertIsNotNone(job)
        self.assertEqual(job["id"], job_id)

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

    def test_next_job_refuses_adapter_repairing_run_without_launching(self):
        database.execute(
            "UPDATE experiment_runs SET status='adapter_repairing', phase='adapter_repairing' WHERE id=1"
        )
        database.execute(
            "INSERT INTO auto_research_jobs (deep_insight_id, status, stage, experiment_run_id) VALUES (1, 'queued_gpu', 'gpu_scheduler', 1)"
        )
        job_id = gpu_scheduler.queue_run(
            insight_id=1, run_id=1, resource_grant_id=1,
            resource_class="gpu_small", priority=1, vram_required_gb=16,
        )
        self.assertIsNone(gpu_scheduler._next_job())
        queued = database.fetchone("SELECT status, error_message FROM gpu_jobs WHERE id=?", (job_id,))
        auto_job = database.fetchone("SELECT status, stage, last_error FROM auto_research_jobs WHERE deep_insight_id=1")
        self.assertEqual(queued["status"], "failed")
        self.assertIn("adapter repair", queued["error_message"])
        self.assertEqual(auto_job["status"], "failed")
        self.assertEqual(auto_job["stage"], "gpu_blocked")

    def test_run_job_rechecks_adapter_repair_before_attempt_admission(self):
        job = {
            "id": 21, "experiment_run_id": 1, "deep_insight_id": 1,
            "agenda_id": 1, "gpu_attempt_reservation_id": 9,
        }
        worker = {"id": "worker-1"}
        with (
            mock.patch.object(gpu_scheduler.db, "fetchone", return_value={
                "id": 1, "agenda_id": 1, "deep_insight_id": 1,
                "status": "adapter_repairing", "phase": "adapter_repairing",
                "error_message": "", "resource_grant_id": 1,
            }),
            mock.patch.object(gpu_scheduler, "_fail_blocked_queued_job") as failed,
            mock.patch.object(gpu_scheduler, "_release_worker_if_no_running_jobs") as released,
            mock.patch.object(gpu_scheduler, "_mark_job_active") as active,
            mock.patch.object(gpu_scheduler.db, "commit"),
        ):
            gpu_scheduler._run_job(job, worker)
        failed.assert_called_once()
        released.assert_called_once()
        active.assert_not_called()

    def test_recover_stale_local_running_job_requeues_after_restart(self):
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

        recovered = gpu_scheduler.recover_stale_local_running_jobs(workers)

        job = database.fetchone("SELECT status, assigned_worker, error_message FROM gpu_jobs WHERE id=?", (job_id,))
        auto_job = database.fetchone("SELECT status, stage, assigned_worker, last_note FROM auto_research_jobs WHERE deep_insight_id=1")
        self.assertEqual(recovered, 1)
        self.assertEqual(job["status"], "queued")
        self.assertIsNone(job["assigned_worker"])
        self.assertIn("stale local running", job["error_message"])
        self.assertEqual(auto_job["status"], "queued_gpu")
        self.assertIsNone(auto_job["assigned_worker"])

    def test_recover_skips_active_local_job_without_gpu_process(self):
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

        gpu_scheduler._mark_job_active(job_id)
        try:
            with mock.patch.object(gpu_scheduler, "_local_run_has_live_process", return_value=False):
                recovered = gpu_scheduler.recover_stale_local_running_jobs(workers)
        finally:
            gpu_scheduler._mark_job_inactive(job_id)

        job = database.fetchone("SELECT status, assigned_worker FROM gpu_jobs WHERE id=?", (job_id,))
        auto_job = database.fetchone("SELECT status, stage FROM auto_research_jobs WHERE deep_insight_id=1")
        self.assertEqual(recovered, 0)
        self.assertEqual(job["status"], "running")
        self.assertEqual(job["assigned_worker"], worker["id"])
        self.assertEqual(auto_job["status"], "running_gpu")

    def test_recover_completed_experiment_with_open_manuscript_requeues(self):
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
            UPDATE experiment_runs
            SET status='completed', hypothesis_verdict='confirmed'
            WHERE id=1
            """
        )
        database.execute(
            """
            INSERT INTO manuscript_runs (experiment_run_id, deep_insight_id, status, workdir)
            VALUES (1, 1, 'drafting', ?)
            """,
            (str(Path(self.tmpdir.name) / "paper_current"),),
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

        with mock.patch.object(gpu_scheduler, "_local_run_has_live_process", return_value=False):
            recovered = gpu_scheduler.recover_stale_local_running_jobs(workers)

        job = database.fetchone("SELECT status, assigned_worker, error_message FROM gpu_jobs WHERE id=?", (job_id,))
        auto_job = database.fetchone("SELECT status, stage, assigned_worker, last_note FROM auto_research_jobs WHERE deep_insight_id=1")
        self.assertEqual(recovered, 1)
        self.assertEqual(job["status"], "queued")
        self.assertIsNone(job["assigned_worker"])
        self.assertIn("stale local running", job["error_message"])
        self.assertEqual(auto_job["status"], "queued_gpu")
        self.assertEqual(auto_job["stage"], "gpu_scheduler")
        self.assertIsNone(auto_job["assigned_worker"])

    def test_periodic_recovery_runs_after_poll_interval(self):
        gpu_scheduler.GPU_STALE_RECOVERY_POLL_SECONDS = 30
        gpu_scheduler._last_recovery_check = 10.0

        with (
            mock.patch.object(gpu_scheduler.time, "time", return_value=20.0),
            mock.patch.object(gpu_scheduler, "recover_stale_ssh_running_jobs") as recover_ssh,
            mock.patch.object(gpu_scheduler, "recover_busy_workers_without_running_jobs") as recover_workers,
        ):
            recovered = gpu_scheduler._maybe_recover_stale_jobs()

        self.assertEqual(recovered, 0)
        recover_ssh.assert_not_called()
        recover_workers.assert_not_called()

        with (
            mock.patch.object(gpu_scheduler.time, "time", return_value=41.0),
            mock.patch.object(gpu_scheduler, "recover_stale_ssh_running_jobs", return_value=2) as recover_ssh,
            mock.patch.object(gpu_scheduler, "recover_busy_workers_without_running_jobs", return_value=1) as recover_workers,
        ):
            recovered = gpu_scheduler._maybe_recover_stale_jobs()

        self.assertEqual(recovered, 3)
        recover_ssh.assert_called_once()
        recover_workers.assert_called_once()

    def test_ssh_recovery_skips_job_active_in_this_process(self):
        database.execute(
            """
            INSERT INTO gpu_workers (id, hostname, gpu_index, gpu_model, total_mem_gb, status, metadata)
            VALUES ('ssh:gpu.example.com:gpu0', 'gpu.example.com', 0, 'L40S', 46, 'busy',
                    '{"backend": "ssh"}')
            """
        )
        job_id = gpu_scheduler.queue_run(
            insight_id=1,
            run_id=1,
            resource_class="gpu_large",
            priority=1,
            vram_required_gb=40,
        )
        database.execute(
            "UPDATE gpu_jobs SET status='running', assigned_worker='ssh:gpu.example.com:gpu0' WHERE id=?",
            (job_id,),
        )
        database.commit()

        gpu_scheduler._mark_job_active(job_id)
        try:
            with mock.patch.object(gpu_scheduler, "_ssh_run_has_live_process", return_value=False) as live_check:
                recovered = gpu_scheduler.recover_stale_ssh_running_jobs()
        finally:
            gpu_scheduler._mark_job_inactive(job_id)

        job = database.fetchone("SELECT status FROM gpu_jobs WHERE id=?", (job_id,))
        self.assertEqual(recovered, 0)
        self.assertEqual(job["status"], "running")
        live_check.assert_not_called()

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
        # The id now carries the port so two nodes behind one host:different-port
        # do not collide on ssh:{host}:gpu{n}.
        first = database.fetchone(
            "SELECT * FROM gpu_workers WHERE id=?", ("ssh:gpu.example.com:55860:gpu0",)
        )
        self.assertIsNotNone(first)
        self.assertIn('"backend": "ssh"', first["metadata"])
        self.assertIn('"ssh_host": "gpu.example.com"', first["metadata"])
        self.assertIn('"ssh_port": 55860', first["metadata"])

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
                SET status='completed', hypothesis_verdict='supported', effect_pct=12.5
                WHERE id=?
                """,
                (run_id,),
            )
            database.commit()
            return {"run_id": run_id, "verdict": "supported"}

        with (
            mock.patch.object(gpu_scheduler, "run_validation_loop", side_effect=_fake_validation_loop),
            mock.patch.object(gpu_scheduler, "process_completed_run"),
            mock.patch.object(gpu_scheduler, "collect_run_artifacts", return_value=[]),
            mock.patch.object(
                gpu_scheduler,
                "positive_decision_authorized",
                return_value=True,
            ),
            mock.patch.object(gpu_scheduler, "generate_submission_bundle", return_value={"error": "latex failed"}),
            mock.patch.object(gpu_scheduler, "log_metrics"),
            mock.patch.object(gpu_scheduler, "log_artifact"),
        ):
            gpu_scheduler._run_job(job, worker)

        run = database.fetchone("SELECT status, hypothesis_verdict FROM experiment_runs WHERE id=1")
        gpu_job = database.fetchone("SELECT status, error_message FROM gpu_jobs WHERE id=?", (job_id,))
        auto_job = database.fetchone("SELECT status, stage, last_error FROM auto_research_jobs WHERE deep_insight_id=1")

        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["hypothesis_verdict"], "supported")
        self.assertEqual(gpu_job["status"], "completed")
        self.assertIn("latex failed", gpu_job["error_message"])
        self.assertEqual(auto_job["status"], "queued")
        self.assertEqual(auto_job["stage"], "manuscript_retry_after_quality_gate")
        self.assertIn("latex failed", auto_job["last_error"])


    def test_run_job_blocks_manuscript_until_benchmark_manifest_is_complete(self):
        workers = gpu_scheduler.register_default_workers()
        worker = workers[0]
        workdir = Path(self.tmpdir.name) / "run1"
        results_dir = workdir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "benchmark_artifact_manifest.json").write_text(
            json.dumps(
                {
                    "full_benchmark_completed": False,
                    "readiness_blockers": ["required baselines missing: Extra Baseline"],
                }
            ),
            encoding="utf-8",
        )
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
                SET status='completed', hypothesis_verdict='supported', effect_pct=12.5
                WHERE id=?
                """,
                (run_id,),
            )
            database.commit()
            return {"run_id": run_id, "verdict": "supported"}

        with (
            mock.patch.object(gpu_scheduler, "run_validation_loop", side_effect=_fake_validation_loop),
            mock.patch.object(gpu_scheduler, "process_completed_run"),
            mock.patch.object(gpu_scheduler, "collect_run_artifacts", return_value=[]),
            mock.patch.object(gpu_scheduler, "generate_submission_bundle") as generate_bundle,
            mock.patch.object(gpu_scheduler, "log_metrics"),
            mock.patch.object(gpu_scheduler, "log_artifact"),
        ):
            gpu_scheduler._run_job(job, worker)

        generate_bundle.assert_not_called()
        gpu_job = database.fetchone("SELECT status, error_message FROM gpu_jobs WHERE id=?", (job_id,))
        auto_job = database.fetchone("SELECT status, stage, last_error FROM auto_research_jobs WHERE deep_insight_id=1")
        self.assertEqual(gpu_job["status"], "completed")
        self.assertIsNone(gpu_job["error_message"])
        self.assertEqual(auto_job["status"], "queued")
        self.assertEqual(auto_job["stage"], "benchmark_completion_required")
        self.assertIn("required baselines missing", auto_job["last_error"])

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

        self.assertEqual(gpu_job["status"], "failed")
        self.assertIn("validation execution failed", gpu_job["error_message"])
        self.assertEqual(auto_job["status"], "failed")
        self.assertEqual(auto_job["stage"], "gpu_failed")
        self.assertIn("validation execution failed", auto_job["last_error"])

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
                SET status='completed', hypothesis_verdict='supported', effect_pct=5.0
                WHERE id=?
                """,
                (run_id,),
            )
            database.commit()
            return {"run_id": run_id, "verdict": "supported", "full_benchmark_completed": True}

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


class LaunchBlockerProjectionTests(unittest.TestCase):
    """A run row that never carried the grant columns is a defect, not a policy.

    _capability_preflight_blocker resolves the grant's passed preflight from
    resource_grant_id and agenda_id. _next_job's projection omitted both, so
    the blocker read None, treated it as grant 0, and failed every queued job
    with "experiment run is not bound to a ResourceGrant" however well bound
    the run was. The legacy GPU path could not launch anything, and the message
    pointed at the data instead of the query.
    """

    BOUND_RUN = {
        "id": 140,
        "deep_insight_id": 105,
        "agenda_id": 11,
        "resource_grant_id": 35,
        "status": "scaffolding",
        "phase": "scaffold_ready",
        "error_message": None,
    }
    PASSED_PREFLIGHT = {
        "preflight_result_id": 25,
        "status": "passed",
        "adapter_id": "transformers_causal_lm_qa_v1",
        "dataset_revision": "a" * 40,
        "model_revision": "b" * 40,
    }

    def test_a_bound_run_with_a_passed_preflight_is_not_blocked(self):
        with (
            mock.patch.object(gpu_scheduler.db, "_use_pg", return_value=True),
            mock.patch.object(
                gpu_scheduler.db, "fetchone", return_value=self.PASSED_PREFLIGHT
            ),
        ):
            self.assertIsNone(
                gpu_scheduler._capability_preflight_blocker(dict(self.BOUND_RUN))
            )

    def test_an_incomplete_projection_names_the_missing_columns(self):
        run = {
            key: value
            for key, value in self.BOUND_RUN.items()
            if key not in {"resource_grant_id", "agenda_id"}
        }
        with mock.patch.object(gpu_scheduler.db, "_use_pg", return_value=True):
            blocker = gpu_scheduler._capability_preflight_blocker(run)
        self.assertIn("missing columns", blocker)
        self.assertIn("resource_grant_id", blocker)
        self.assertIn("agenda_id", blocker)
        self.assertNotIn("not bound to a ResourceGrant", blocker)

    def test_a_genuinely_unbound_run_is_still_refused(self):
        run = dict(self.BOUND_RUN, resource_grant_id=0)
        with mock.patch.object(gpu_scheduler.db, "_use_pg", return_value=True):
            self.assertEqual(
                gpu_scheduler._capability_preflight_blocker(run),
                "experiment run is not bound to a ResourceGrant",
            )


class FalselyUnboundRecoveryTests(unittest.TestCase):
    """Recover only where the recorded verdict contradicts the database."""

    ROW = {
        "job_id": 112,
        "agenda_id": 11,
        "deep_insight_id": 105,
        "run_id": 140,
        "phase": "scaffold_ready",
    }

    def test_requeues_the_run_without_regenerating_the_scaffold(self):
        statements = []

        def _execute(sql, params=None):
            statements.append((" ".join(sql.split()), params))

        with (
            mock.patch.object(gpu_scheduler.db, "fetchall", return_value=[self.ROW]),
            mock.patch.object(gpu_scheduler.db, "execute", side_effect=_execute),
            mock.patch.object(gpu_scheduler.db, "commit"),
            mock.patch(
                "orchestrator.meta_compute_runtime.settle_legacy_job",
                return_value="failed",
            ) as settle,
        ):
            self.assertEqual(gpu_scheduler.recover_falsely_unbound_gpu_jobs(), 1)

        settle.assert_called_once_with(112)
        run_update = next(s for s, _ in statements if "UPDATE experiment_runs" in s)
        self.assertIn("status='scaffolding'", run_update)
        self.assertIn("phase='scaffold_ready'", run_update)
        job_update, params = next(
            (s, p) for s, p in statements if "UPDATE auto_research_jobs" in s
        )
        self.assertIn("status='queued'", job_update)
        # The run id is preserved so the requeue reuses the admitted adapter
        # instead of paying for a fresh forge.
        self.assertEqual(params[0], 140)

    def test_the_predicate_requires_an_active_grant_and_passed_preflight(self):
        with (
            mock.patch.object(gpu_scheduler.db, "fetchall", return_value=[]) as fetch,
            mock.patch.object(gpu_scheduler.db, "execute") as execute,
        ):
            self.assertEqual(gpu_scheduler.recover_falsely_unbound_gpu_jobs(), 0)
        execute.assert_not_called()
        sql = " ".join(fetch.call_args.args[0].split())
        self.assertIn("g.status='active'", sql)
        self.assertIn("p.status='passed'", sql)
        self.assertIn("j.status='failed'", sql)
        self.assertIn("r.phase='scaffold_ready'", sql)
        self.assertEqual(fetch.call_args.args[1][0], gpu_scheduler._FALSE_UNBOUND_BLOCKER)


class ColabWorkerRegistrationTests(unittest.TestCase):
    """Preflight can only select Colab once something measured the device.

    preflight holds Colab VRAM at zero until a canary records real hardware,
    which is the correct default for a backend whose allocation is not
    guaranteed. Nothing could record that measurement, so the zero was
    permanent and colab_gpu was unselectable by construction.
    """

    def test_a_measured_device_is_written_as_an_idle_colab_worker(self):
        statements = []

        with (
            mock.patch.object(gpu_scheduler.db, "fetchone", return_value=None),
            mock.patch.object(
                gpu_scheduler.db,
                "execute",
                side_effect=lambda sql, params=None: statements.append(
                    (" ".join(sql.split()), params)
                ),
            ),
            mock.patch.object(
                gpu_scheduler, "_local_hostname", return_value="host"
            ),
        ):
            summary = gpu_scheduler.upsert_colab_worker(
                account_ref="colab-pro",
                gpu_model="Tesla T4",
                total_mem_gb=14.56,
                measured_at="2026-08-15",
                detail={"driver": "580.82.07"},
            )

        self.assertEqual(summary["total_mem_gb"], 14.56)
        self.assertEqual(summary["status"], "idle")
        self.assertEqual(summary["backend"], "colab")
        sql, params = statements[0]
        self.assertIn("INSERT INTO gpu_workers", sql)
        self.assertIn("'idle'", sql)
        self.assertIn("host:colab-colab-pro", params)

    def test_an_unmeasured_device_is_refused(self):
        for kwargs in (
            {"account_ref": "", "total_mem_gb": 14.56},
            {"account_ref": "colab-pro", "total_mem_gb": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    gpu_scheduler.upsert_colab_worker(
                        gpu_model="Tesla T4", measured_at="2026-08-15", **kwargs
                    )
