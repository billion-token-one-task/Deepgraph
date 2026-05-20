import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.paradigm_agent import store_deep_insight
from agents import discovery_supervisor
from agents.signal_harvester import get_tier2_signals, harvest_protocol_artifacts
from db import database
from orchestrator import discovery_scheduler


class TempDbTestCase(unittest.TestCase):
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

    def tearDown(self):
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


class Tier2SignalCompatibilityTests(TempDbTestCase):
    def test_get_tier2_signals_falls_back_without_paradigm_score(self):
        database.execute(
            "INSERT INTO deep_insights (tier, title, adversarial_score, mechanism_type) VALUES (1, ?, ?, ?)",
            ("Mechanism-first insight", 6.0, "protocol_artifact"),
        )
        database.commit()

        signals = get_tier2_signals()
        self.assertEqual(len(signals["high_potential_insights"]), 1)
        self.assertEqual(signals["high_potential_insights"][0]["title"], "Mechanism-first insight")

    def test_protocol_artifact_harvests_non_numeric_evidence(self):
        database.execute("INSERT INTO taxonomy_nodes (id, name, depth) VALUES ('ml.test', 'Test Node', 1)")
        database.execute("INSERT INTO papers (id, title) VALUES ('p1', 'Paper 1')")
        database.execute("INSERT INTO paper_taxonomy (paper_id, node_id) VALUES ('p1', 'ml.test')")
        database.execute(
            """
            INSERT INTO paper_insights (paper_id, limitations, open_questions)
            VALUES (?, ?, ?)
            """,
            (
                "p1",
                '["Benchmark protocol choices dominate the score."]',
                '["How sensitive is the metric choice?"]',
            ),
        )
        database.commit()

        count = harvest_protocol_artifacts(min_support=1)
        rows = database.fetchall("SELECT * FROM protocol_artifacts")
        self.assertEqual(count, 2)
        self.assertTrue(any(row["artifact_type"] == "benchmark_protocol" for row in rows))


class DiscoveryRankingTests(unittest.TestCase):
    def test_pairwise_ranking_prefers_richer_evidence_packet(self):
        candidates = [
            {
                "id": "a",
                "title": "Weak numeric idea",
                "signal_mix": ["plateau"],
                "mechanism_type": "plateau",
                "evidence_packet": {"non_numeric_evidence": ["one"], "structural_evidence": []},
                "resource_class": "cpu",
                "support_score": 0.2,
            },
            {
                "id": "b",
                "title": "Mechanism-first idea",
                "signal_mix": ["protocol_artifact", "mechanism_mismatch"],
                "mechanism_type": "protocol_artifact",
                "evidence_packet": {
                    "non_numeric_evidence": ["protocol flaw", "label mismatch"],
                    "structural_evidence": ["shared hidden variable"],
                },
                "resource_class": "cpu",
                "support_score": 1.5,
            },
        ]
        with mock.patch.object(discovery_supervisor, "collect_candidate_pool", return_value=candidates):
            ranked = discovery_supervisor.rank_candidates(limit=2)
        self.assertEqual(ranked[0]["id"], "b")


class InsightStorageValidationTests(unittest.TestCase):
    def test_store_deep_insight_skips_incomplete_tier1_payload(self):
        with mock.patch("agents.paradigm_agent.db.insert_returning_id") as insert_returning_id:
            rid = store_deep_insight(
                {
                    "tier": 1,
                    "status": "candidate",
                    "title": "Mechanism-first insight",
                    "field_a": "{}",
                    "field_b": "{}",
                    "formal_structure": "",
                    "transformation": "",
                }
            )

        self.assertEqual(rid, 0)
        insert_returning_id.assert_not_called()


class DiscoverySchedulerSkipTests(unittest.TestCase):
    def test_run_tier1_discovery_skips_zero_id_and_keeps_next_result(self):
        insights = [
            {"title": "Bad insight"},
            {"title": "Good insight", "adversarial_score": 7},
        ]

        with (
            mock.patch.object(discovery_scheduler, "_init_schema_v2"),
            mock.patch("agents.paradigm_agent.discover_paradigm_insights", return_value=insights),
            mock.patch("agents.paradigm_agent.store_deep_insight", side_effect=[0, 42]),
            mock.patch.object(discovery_scheduler, "log_event"),
        ):
            stored = discovery_scheduler.run_tier1_discovery(max_candidates=2)

        self.assertEqual(stored, [{"id": 42, "title": "Good insight", "adversarial_score": 7}])


if __name__ == "__main__":
    unittest.main()
