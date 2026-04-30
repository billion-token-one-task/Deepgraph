import unittest
from pathlib import Path
import tempfile
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
