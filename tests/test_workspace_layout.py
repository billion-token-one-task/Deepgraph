import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents import workspace_layout
from db import database


def _load_backfill_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "backfill_idea_workspaces.py"
    spec = importlib.util.spec_from_file_location("backfill_idea_workspaces", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class WorkspaceLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test.db"
        self.old_db_path = database.DB_PATH
        self.old_database_url = database.DATABASE_URL
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(database._local, attr):
                try:
                    getattr(database._local, attr).close()
                except Exception:
                    pass
                setattr(database._local, attr, None)
        database.DATABASE_URL = ""
        database.DB_PATH = self.db_path
        database.init_db()
        self.workspace_root = Path(self.tmpdir.name) / "ideas"
        self.workspace_patch = mock.patch.object(workspace_layout, "IDEA_WORKSPACE_DIR", self.workspace_root)
        self.workspace_patch.start()

    def tearDown(self):
        self.workspace_patch.stop()
        for attr in ("pg_conn", "sqlite_conn", "conn"):
            if hasattr(database._local, attr):
                try:
                    getattr(database._local, attr).close()
                except Exception:
                    pass
                setattr(database._local, attr, None)
        database.DATABASE_URL = self.old_database_url
        database.DB_PATH = self.old_db_path
        self.tmpdir.cleanup()

    def test_get_idea_workspace_persists_roots_and_promotes_canonical_run(self):
        database.execute("INSERT INTO deep_insights (id, tier, title) VALUES (1, 2, 'Idea Workspace')")
        database.execute("INSERT INTO experiment_runs (id, deep_insight_id, status) VALUES (5, 1, 'testing')")
        database.commit()

        layout = workspace_layout.get_idea_workspace(1)
        self.assertTrue(Path(layout["workspace_root"]).exists())
        self.assertTrue(Path(layout["plan_root"]).exists())

        workspace_layout.promote_canonical_run(1, 5)

        insight = database.fetchone("SELECT * FROM deep_insights WHERE id=1")
        self.assertEqual(insight["canonical_run_id"], 5)
        self.assertTrue(Path(layout["experiment_current_root"]).exists())
        self.assertTrue((Path(layout["plan_root"]) / "latest_status.json").exists())

    def test_promote_canonical_run_falls_back_to_marker_when_current_dir_locked(self):
        database.execute("INSERT INTO deep_insights (id, tier, title) VALUES (1, 2, 'Idea Workspace')")
        database.execute("INSERT INTO experiment_runs (id, deep_insight_id, status) VALUES (5, 1, 'testing')")
        database.commit()
        layout = workspace_layout.ensure_run_workspace(1, 5)

        with mock.patch.object(workspace_layout.shutil, "rmtree", side_effect=OSError("locked")):
            workspace_layout.promote_canonical_run(1, 5)

        marker = Path(layout["suite_current_root"]) / "CURRENT_RUN.txt"
        self.assertTrue(marker.exists())
        self.assertEqual(marker.read_text(encoding="utf-8"), str(layout["run_root"]))

    def test_get_idea_workspace_tolerates_existing_current_symlink(self):
        database.execute("INSERT INTO deep_insights (id, tier, title) VALUES (1, 2, 'Idea Workspace')")
        database.commit()

        root = self.workspace_root / "idea_1"
        target = root / "legacy_target"
        target.mkdir(parents=True, exist_ok=True)
        current_link = root / "experiments" / "main" / "current"
        current_link.parent.mkdir(parents=True, exist_ok=True)
        current_link.symlink_to(target, target_is_directory=True)

        layout = workspace_layout.get_idea_workspace(1)

        self.assertTrue(Path(layout["experiment_current_root"]).is_symlink())
        self.assertEqual(Path(layout["experiment_current_root"]).resolve(), target.resolve())

    def test_backfill_script_maps_legacy_run_and_manuscript_dirs(self):
        legacy_run = Path(self.tmpdir.name) / "legacy_run"
        (legacy_run / "code").mkdir(parents=True, exist_ok=True)
        (legacy_run / "code" / "train.py").write_text("print('legacy')", encoding="utf-8")
        legacy_paper = Path(self.tmpdir.name) / "legacy_paper"
        legacy_paper.mkdir(parents=True, exist_ok=True)
        (legacy_paper / "main.tex").write_text("\\documentclass{article}", encoding="utf-8")

        database.execute("INSERT INTO deep_insights (id, tier, title) VALUES (1, 2, 'Migrated Idea')")
        database.execute(
            "INSERT INTO experiment_runs (id, deep_insight_id, status, workdir) VALUES (7, 1, 'completed', ?)",
            (str(legacy_run),),
        )
        database.execute(
            "INSERT INTO manuscript_runs (id, experiment_run_id, deep_insight_id, status, workdir) VALUES (3, 7, 1, 'bundle_ready', ?)",
            (str(legacy_paper),),
        )
        database.commit()

        module = _load_backfill_module()
        result = module.backfill_all(dry_run=False)

        self.assertEqual(len(result), 1)
        migrated_run = self.workspace_root / "idea_1" / "experiments" / "main" / "runs" / "run_7"
        migrated_paper = self.workspace_root / "idea_1" / "papers" / "current" / "main.tex"
        self.assertTrue((migrated_run / "code" / "train.py").exists())
        self.assertTrue(migrated_paper.exists())
        refreshed_run = database.fetchone("SELECT * FROM experiment_runs WHERE id=7")
        refreshed_idea = database.fetchone("SELECT * FROM deep_insights WHERE id=1")
        self.assertEqual(refreshed_run["workdir"], str(migrated_run))
        self.assertEqual(refreshed_idea["canonical_run_id"], 7)


if __name__ == "__main__":
    unittest.main()
