import unittest
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock

import main


class MainSingleInstanceTests(unittest.TestCase):
    def test_main_refuses_duplicate_process_before_side_effects(self):
        with (
            mock.patch.object(main, "_try_acquire_process_lock", return_value=False),
            mock.patch.object(main, "init_db") as init_db,
            mock.patch.object(main, "_serve_http") as serve_http,
        ):
            main.main()

        init_db.assert_not_called()
        serve_http.assert_not_called()

    def test_main_releases_lock_on_shutdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with (
                mock.patch.object(main, "_try_acquire_process_lock", return_value=True),
                mock.patch.object(main, "WORKSPACE_DIR", tmpdir_path / "workspace"),
                mock.patch.object(main, "PDF_CACHE_DIR", tmpdir_path / "pdf_cache"),
                mock.patch.object(main, "IDEA_WORKSPACE_DIR", tmpdir_path / "ideas"),
                mock.patch.object(main, "init_db"),
                mock.patch.object(main, "describe_backend", return_value={"target": "postgresql://test", "backend": "postgresql"}),
                mock.patch.object(main, "seed_taxonomy"),
                mock.patch.object(main, "backfill_result_taxonomy"),
                mock.patch.object(main, "backfill_entity_resolutions"),
                mock.patch.object(main, "AUTO_RESEARCH_ENABLED", False),
                mock.patch.object(main, "_serve_http", side_effect=RuntimeError("stop")),
                mock.patch.object(main, "_release_process_lock") as release_lock,
            ):
                with self.assertRaises(RuntimeError):
                    main.main()

        release_lock.assert_called_once()

    def test_compute_recovery_precedes_auto_research_worker(self):
        events: list[str] = []
        fake_gpu_scheduler = types.ModuleType("orchestrator.gpu_scheduler")
        fake_gpu_scheduler.start = lambda: (
            events.append("compute_reconciled") or {"status": "started"}
        )
        fake_auto_research = types.ModuleType("orchestrator.auto_research")
        fake_auto_research.start = lambda: (
            events.append("auto_research_started") or {"status": "started"}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "orchestrator.gpu_scheduler": fake_gpu_scheduler,
                        "orchestrator.auto_research": fake_auto_research,
                    },
                ),
                mock.patch.object(main, "_try_acquire_process_lock", return_value=True),
                mock.patch.object(main, "WORKSPACE_DIR", tmpdir_path / "workspace"),
                mock.patch.object(main, "PDF_CACHE_DIR", tmpdir_path / "pdf_cache"),
                mock.patch.object(main, "IDEA_WORKSPACE_DIR", tmpdir_path / "ideas"),
                mock.patch.object(main, "init_db"),
                mock.patch.object(
                    main,
                    "describe_backend",
                    return_value={"target": "postgresql://test", "backend": "postgresql"},
                ),
                mock.patch.object(main, "seed_taxonomy"),
                mock.patch.object(main, "backfill_result_taxonomy"),
                mock.patch.object(main, "backfill_entity_resolutions"),
                mock.patch.object(main, "AUTO_RESEARCH_ENABLED", True),
                mock.patch.object(main, "AUTO_PIPELINE_ENABLED", False),
                mock.patch.object(main, "_serve_http", side_effect=RuntimeError("stop")),
                mock.patch.object(main, "_release_process_lock"),
            ):
                with self.assertRaises(RuntimeError):
                    main.main()

        self.assertEqual(
            events,
            ["compute_reconciled", "auto_research_started"],
        )

    def test_auto_research_is_not_started_when_compute_recovery_fails(self):
        fake_gpu_scheduler = types.ModuleType("orchestrator.gpu_scheduler")
        fake_gpu_scheduler.start = lambda: {"status": "recovery_failed"}
        fake_auto_research = types.ModuleType("orchestrator.auto_research")
        fake_auto_research.start = mock.Mock(return_value={"status": "started"})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "orchestrator.gpu_scheduler": fake_gpu_scheduler,
                        "orchestrator.auto_research": fake_auto_research,
                    },
                ),
                mock.patch.object(main, "_try_acquire_process_lock", return_value=True),
                mock.patch.object(main, "WORKSPACE_DIR", tmpdir_path / "workspace"),
                mock.patch.object(main, "PDF_CACHE_DIR", tmpdir_path / "pdf_cache"),
                mock.patch.object(main, "IDEA_WORKSPACE_DIR", tmpdir_path / "ideas"),
                mock.patch.object(main, "init_db"),
                mock.patch.object(
                    main,
                    "describe_backend",
                    return_value={"target": "postgresql://test", "backend": "postgresql"},
                ),
                mock.patch.object(main, "seed_taxonomy"),
                mock.patch.object(main, "backfill_result_taxonomy"),
                mock.patch.object(main, "backfill_entity_resolutions"),
                mock.patch.object(main, "AUTO_RESEARCH_ENABLED", True),
                mock.patch.object(main, "AUTO_PIPELINE_ENABLED", False),
                mock.patch.object(main, "_serve_http"),
                mock.patch.object(main, "_release_process_lock"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Compute scheduler failed closed",
                ):
                    main.main()

        fake_auto_research.start.assert_not_called()

    def _run_main_with_paper_worker(self, start_fn):
        """Boot main() with AUTO_PIPELINE on and a stubbed ingestion worker."""
        fake_paper_worker = types.ModuleType("orchestrator.paper_worker")
        fake_paper_worker.start = start_fn

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with (
                mock.patch.dict(
                    sys.modules,
                    {"orchestrator.paper_worker": fake_paper_worker},
                ),
                mock.patch.object(main, "_try_acquire_process_lock", return_value=True),
                mock.patch.object(main, "WORKSPACE_DIR", tmpdir_path / "workspace"),
                mock.patch.object(main, "PDF_CACHE_DIR", tmpdir_path / "pdf_cache"),
                mock.patch.object(main, "IDEA_WORKSPACE_DIR", tmpdir_path / "ideas"),
                mock.patch.object(main, "init_db"),
                mock.patch.object(
                    main,
                    "describe_backend",
                    return_value={"target": "postgresql://test", "backend": "postgresql"},
                ),
                mock.patch.object(main, "seed_taxonomy"),
                mock.patch.object(main, "backfill_result_taxonomy"),
                mock.patch.object(main, "backfill_entity_resolutions"),
                mock.patch.object(main, "AUTO_RESEARCH_ENABLED", False),
                mock.patch.object(main, "AUTO_PIPELINE_ENABLED", True),
                mock.patch.object(main, "_serve_http") as serve_http,
                mock.patch.object(main, "_release_process_lock"),
                # Shadow print inside main's namespace only. Patching
                # builtins.print breaks the threading excepthook, and unrelated
                # daemon threads in the suite then fail this test.
                mock.patch.object(main, "print", create=True) as printed,
            ):
                main.main()

        lines = " ".join(str(call.args[0]) for call in printed.call_args_list if call.args)
        return serve_http, lines

    def test_legacy_background_ingestion_degrades_without_killing_web(self):
        """A fail-closed ingestion worker must not take the site down with it."""
        started: list[str] = []

        def start():
            started.append("called")
            return {
                "status": "disabled_resource_grant_required",
                "reason": "agenda_scoped_ingestion_job_required",
            }

        serve_http, lines = self._run_main_with_paper_worker(start)

        self.assertEqual(started, ["called"])
        # Fail-closed still holds: the worker is not treated as ready.
        self.assertNotIn("Paper ingestion worker ready.", lines)
        self.assertIn("Paper ingestion worker degraded", lines)
        self.assertIn("disabled_resource_grant_required", lines)
        self.assertIn("agenda_scoped_ingestion_job_required", lines)
        # And the web process still comes up.
        serve_http.assert_called_once()

    def test_background_ingestion_startup_exception_degrades(self):
        """An exception inside worker startup degrades instead of propagating."""

        def start():
            raise RuntimeError("grant lookup exploded")

        serve_http, lines = self._run_main_with_paper_worker(start)

        self.assertNotIn("Paper ingestion worker ready.", lines)
        self.assertIn("Paper ingestion worker degraded", lines)
        self.assertIn("grant lookup exploded", lines)
        serve_http.assert_called_once()
