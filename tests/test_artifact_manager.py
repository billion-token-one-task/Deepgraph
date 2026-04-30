import json
import unittest
from pathlib import Path

from agents.artifact_manager import (
    artifact_path,
    ensure_artifact_dirs,
    list_artifacts,
    read_text_artifact,
    record_artifact,
)
from tests.temp_utils import temporary_workdir


class ArtifactManagerTests(unittest.TestCase):
    def test_ensure_artifact_dirs_creates_expected_directories(self):
        with temporary_workdir() as workdir:

            dirs = ensure_artifact_dirs(workdir)

            self.assertEqual(set(dirs), {"logs", "results", "figures", "tables", "manuscript", "reviews"})
            for path in dirs.values():
                self.assertTrue(path.is_dir())
                self.assertTrue(path.resolve().is_relative_to(workdir.resolve()))

    def test_record_artifact_stores_relative_paths(self):
        with temporary_workdir() as workdir:
            result_path = artifact_path(workdir, "artifacts/results/metrics.json")
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text('{"metric_value": 0.8}', encoding="utf-8")

            entry = record_artifact(workdir, 7, "metrics", result_path, {"metric": "accuracy"})
            manifest = json.loads((workdir / "artifact_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(entry["path"], "artifacts/results/metrics.json")
            self.assertEqual(manifest["run_id"], 7)
            self.assertEqual(manifest["artifacts"][0]["path"], "artifacts/results/metrics.json")
            self.assertNotIn(str(workdir), manifest["artifacts"][0]["path"])

    def test_list_artifacts_returns_empty_for_missing_manifest(self):
        with temporary_workdir() as workdir:
            self.assertEqual(list_artifacts(workdir), [])

    def test_checksum_changes_when_file_content_changes(self):
        with temporary_workdir() as workdir:
            path = artifact_path(workdir, "artifacts/results/metrics.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("first", encoding="utf-8")
            first = record_artifact(workdir, 1, "metrics", path)["sha256"]

            path.write_text("second", encoding="utf-8")
            second = record_artifact(workdir, 1, "metrics", path)["sha256"]

            self.assertNotEqual(first, second)

    def test_rejects_paths_outside_workdir(self):
        with temporary_workdir() as tmp:
            workdir = Path(tmp) / "work"
            workdir.mkdir()
            outside = Path(tmp) / "outside.txt"
            outside.write_text("secret", encoding="utf-8")

            with self.assertRaises(ValueError):
                record_artifact(workdir, 1, "leak", outside)

            with self.assertRaises(ValueError):
                artifact_path(workdir, "../outside.txt")

    def test_read_text_artifact_caps_length(self):
        with temporary_workdir() as workdir:
            path = artifact_path(workdir, "artifacts/logs/run.log")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("abcdef", encoding="utf-8")

            self.assertEqual(read_text_artifact(workdir, "artifacts/logs/run.log", max_chars=3), "abc")


if __name__ == "__main__":
    unittest.main()
