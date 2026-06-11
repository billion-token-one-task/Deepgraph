"""Tests for agents.direction_intake + scripts.agenda_inbox_watcher.

Covers the deterministic YAML -> ResearchAgenda mapping (good input, defaults,
compute-constraint mapping, echo content) and bad-input handling, plus the
inbox watcher's processed/failed file flow against a temp database.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DEEPGRAPH_DATABASE_URL", "")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


SAMPLE_DIRECTION_YAML = """
direction: "用扩散模型做小样本医学影像分割，关注跨中心泛化"
keywords: [medical imaging, diffusion, few-shot]
constraints:
  compute: "单卡以内"
  data: "仅公开数据集"
goal: experiment_plan
contact: "alice@example.com"
"""


class DirectionParsingTests(unittest.TestCase):
    def test_full_sample_maps_all_fields(self):
        from agents.direction_intake import parse_direction_yaml

        agenda, echo = parse_direction_yaml(SAMPLE_DIRECTION_YAML)
        self.assertEqual(agenda.description, "用扩散模型做小样本医学影像分割，关注跨中心泛化")
        self.assertEqual(agenda.focus, ["medical imaging", "diffusion", "few-shot"])
        self.assertEqual(agenda.required_output, {"goal": "experiment_plan"})
        self.assertEqual(agenda.submitter, "alice@example.com")
        # "单卡以内" -> conservative resource classes
        self.assertEqual(agenda.prefer.get("resource_class"), ["cpu", "gpu_small"])
        # Original submission preserved verbatim
        self.assertEqual(agenda.raw_config["constraints"]["compute"], "单卡以内")
        self.assertEqual(agenda.raw_config["constraints"]["data"], "仅公开数据集")
        self.assertTrue(agenda.name.startswith("direction-medical-imaging"))
        # Echo summarises what was understood
        self.assertEqual(echo["type"], "direction_intake_echo")
        self.assertEqual(echo["focus"], agenda.focus)
        self.assertEqual(echo["goal"], "experiment_plan")
        self.assertIn("单卡以内", echo["summary"])
        self.assertIn("alice@example.com", echo["summary"])

    def test_name_is_deterministic(self):
        from agents.direction_intake import parse_direction_yaml

        a1, _ = parse_direction_yaml(SAMPLE_DIRECTION_YAML)
        a2, _ = parse_direction_yaml(SAMPLE_DIRECTION_YAML)
        self.assertEqual(a1.name, a2.name)

    def test_goal_defaults_to_experiment_plan(self):
        from agents.direction_intake import parse_direction_yaml

        agenda, _ = parse_direction_yaml(
            "direction: few-shot segmentation with diffusion models\n"
            "contact: bob\n"
        )
        self.assertEqual(agenda.required_output, {"goal": "experiment_plan"})

    def test_focus_falls_back_to_direction_tokens(self):
        from agents.direction_intake import parse_direction_yaml

        agenda, _ = parse_direction_yaml(
            "direction: cross-center generalization for medical segmentation\n"
            "contact: bob\n"
        )
        self.assertIn("medical", agenda.focus)
        self.assertIn("segmentation", agenda.focus)

    def test_compute_mapping_rules(self):
        from agents.direction_intake import map_compute_constraint

        self.assertEqual(map_compute_constraint("单卡以内"), ["cpu", "gpu_small"])
        self.assertEqual(map_compute_constraint("Single GPU please"), ["cpu", "gpu_small"])
        self.assertEqual(map_compute_constraint("笔记本就能跑"), ["cpu"])
        self.assertEqual(map_compute_constraint("CPU only"), ["cpu"])
        self.assertIsNone(map_compute_constraint("8x H100 cluster"))
        self.assertIsNone(map_compute_constraint(""))

    # ---------- bad input ----------

    def test_missing_direction_rejected(self):
        from agents.direction_intake import DirectionParseError, parse_direction_yaml

        with self.assertRaises(DirectionParseError):
            parse_direction_yaml("keywords: [a, b]\ncontact: bob\n")

    def test_missing_contact_rejected(self):
        from agents.direction_intake import DirectionParseError, parse_direction_yaml

        with self.assertRaises(DirectionParseError):
            parse_direction_yaml("direction: some research direction\n")

    def test_invalid_goal_rejected(self):
        from agents.direction_intake import DirectionParseError, parse_direction_yaml

        with self.assertRaises(DirectionParseError):
            parse_direction_yaml(
                "direction: some research direction\ncontact: bob\ngoal: world_peace\n"
            )

    def test_non_mapping_yaml_rejected(self):
        from agents.direction_intake import DirectionParseError, parse_direction_yaml

        with self.assertRaises(DirectionParseError):
            parse_direction_yaml("- just\n- a list\n")

    def test_empty_text_rejected(self):
        from agents.direction_intake import DirectionParseError, parse_direction_yaml

        with self.assertRaises(DirectionParseError):
            parse_direction_yaml("   \n")

    def test_broken_yaml_rejected(self):
        from agents.direction_intake import DirectionParseError, parse_direction_yaml

        with self.assertRaises(DirectionParseError):
            parse_direction_yaml("direction: [unclosed\ncontact: bob\n")

    def test_chinese_only_direction_without_keywords_rejected(self):
        from agents.direction_intake import DirectionParseError, parse_direction_yaml

        # No keywords, no extractable ASCII terms, no compute constraint:
        # nothing to scope on -> reject with guidance instead of a blind agenda.
        with self.assertRaises(DirectionParseError):
            parse_direction_yaml("direction: 量子计算研究\ncontact: bob\n")

    def test_bad_token_budget_rejected(self):
        from agents.direction_intake import DirectionParseError, parse_direction_yaml

        with self.assertRaises(DirectionParseError):
            parse_direction_yaml(
                "direction: diffusion segmentation\ncontact: bob\ntoken_budget: lots\n"
            )


class InboxWatcherTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["DEEPGRAPH_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        from db import database as db

        self._original_db_path = db.DB_PATH
        for attr in ("sqlite_conn", "pg_conn", "conn"):
            if hasattr(db._local, attr):
                try:
                    getattr(db._local, attr).close()
                except Exception:
                    pass
                delattr(db._local, attr)
        db.DB_PATH = Path(os.environ["DEEPGRAPH_DB_PATH"])
        db.init_db()
        self.db = db
        self.inbox = Path(self._tmpdir.name) / "inbox"

    def tearDown(self):
        from db import database as db

        for attr in ("sqlite_conn", "pg_conn", "conn"):
            if hasattr(db._local, attr):
                try:
                    getattr(db._local, attr).close()
                except Exception:
                    pass
                delattr(db._local, attr)
        db.DB_PATH = self._original_db_path
        self._tmpdir.cleanup()
        os.environ.pop("DEEPGRAPH_DB_PATH", None)

    def test_good_file_is_processed_with_echo(self):
        import agenda_inbox_watcher as watcher

        self.inbox.mkdir(parents=True)
        (self.inbox / "alice.yaml").write_text(SAMPLE_DIRECTION_YAML, encoding="utf-8")

        results = watcher.scan_inbox(self.inbox)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "ok")
        agenda_id = results[0]["agenda_id"]

        # File moved to processed/ + echo written
        self.assertFalse((self.inbox / "alice.yaml").exists())
        processed = self.inbox / "processed" / "alice.yaml"
        self.assertTrue(processed.exists())
        echo = json.loads(
            (self.inbox / "processed" / "alice.yaml.echo.json").read_text(encoding="utf-8")
        )
        self.assertEqual(echo["agenda_id"], agenda_id)
        self.assertEqual(echo["focus"], ["medical imaging", "diffusion", "few-shot"])

        # Agenda actually persisted
        row = self.db.fetchone("SELECT * FROM research_agendas WHERE id=?", (agenda_id,))
        self.assertIsNotNone(row)
        self.assertEqual(row["submitter"], "alice@example.com")

    def test_bad_file_is_quarantined_with_error(self):
        import agenda_inbox_watcher as watcher

        self.inbox.mkdir(parents=True)
        (self.inbox / "broken.yaml").write_text("keywords: [a]\n", encoding="utf-8")

        results = watcher.scan_inbox(self.inbox)
        self.assertEqual(results[0]["status"], "failed")
        self.assertFalse((self.inbox / "broken.yaml").exists())
        failed = self.inbox / "failed" / "broken.yaml"
        self.assertTrue(failed.exists())
        error_text = (self.inbox / "failed" / "broken.yaml.error.txt").read_text(encoding="utf-8")
        self.assertIn("direction", error_text)
        # Nothing persisted
        row = self.db.fetchone("SELECT COUNT(*) AS c FROM research_agendas")
        self.assertEqual(row["c"], 0)

    def test_oversized_file_quarantined_without_parsing(self):
        import agenda_inbox_watcher as watcher

        self.inbox.mkdir(parents=True)
        big = "# padding\n" * (watcher.MAX_SUBMISSION_BYTES // 10 + 1)
        path = self.inbox / "huge.yaml"
        path.write_text(big, encoding="utf-8")
        self.assertGreater(path.stat().st_size, watcher.MAX_SUBMISSION_BYTES)

        results = watcher.scan_inbox(self.inbox)
        self.assertEqual(results[0]["status"], "failed")
        self.assertIn("too large", results[0]["error"])
        self.assertFalse(path.exists())
        failed = self.inbox / "failed" / "huge.yaml"
        self.assertTrue(failed.exists())
        error_text = (self.inbox / "failed" / "huge.yaml.error.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("too large", error_text)
        self.assertIn(str(watcher.MAX_SUBMISSION_BYTES), error_text)
        # Nothing persisted
        row = self.db.fetchone("SELECT COUNT(*) AS c FROM research_agendas")
        self.assertEqual(row["c"], 0)

    def test_non_yaml_files_ignored(self):
        import agenda_inbox_watcher as watcher

        self.inbox.mkdir(parents=True)
        (self.inbox / "notes.txt").write_text("hello", encoding="utf-8")
        results = watcher.scan_inbox(self.inbox)
        self.assertEqual(results, [])
        self.assertTrue((self.inbox / "notes.txt").exists())


if __name__ == "__main__":
    unittest.main()
