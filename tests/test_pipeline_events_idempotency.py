import tempfile
import unittest
from pathlib import Path
from unittest import mock

from db import database
from db import taxonomy
from orchestrator import discovery_scheduler


class TempDbCase(unittest.TestCase):
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


class IdempotencyAndEventTests(TempDbCase):
    def test_insert_paper_dedups_arxiv_versions(self):
        first = database.insert_paper(
            {
                "id": "2401.12345v1",
                "title": "Paper V1",
                "authors": ["A"],
                "abstract": "abs",
                "categories": ["cs.LG"],
                "published_date": "2024-01-01",
                "pdf_url": "https://example.com/v1.pdf",
            }
        )
        second = database.insert_paper(
            {
                "id": "2401.12345v2",
                "title": "Paper V2",
                "authors": ["A", "B"],
                "abstract": "updated",
                "categories": ["cs.LG"],
                "published_date": "2024-02-01",
                "pdf_url": "https://example.com/v2.pdf",
            }
        )
        rows = database.fetchall("SELECT id, arxiv_base_id, title FROM papers")
        self.assertEqual(len(rows), 1)
        self.assertEqual(first, second)
        self.assertEqual(rows[0]["arxiv_base_id"], "2401.12345")
        self.assertEqual(rows[0]["title"], "Paper V2")

    def test_insert_claim_and_result_are_idempotent(self):
        database.execute("INSERT INTO papers (id, title) VALUES (?, ?)", ("p1", "Paper 1"))
        database.execute("INSERT INTO taxonomy_nodes (id, name, depth) VALUES (?, ?, ?)", ("ml.test", "Test", 1))
        database.commit()

        claim = {
            "paper_id": "p1",
            "claim_text": "The method improves accuracy.",
            "claim_type": "finding",
            "method_name": "MyMethod",
            "dataset_name": "ToySet",
            "metric_name": "accuracy",
            "metric_value": 0.91,
            "evidence_location": "Table 1",
            "conditions": {"seed": 1},
        }
        claim_id_1 = database.insert_claim(dict(claim))
        claim_id_2 = database.insert_claim(dict(claim))
        self.assertEqual(claim_id_1, claim_id_2)
        self.assertEqual(database.fetchone("SELECT COUNT(*) AS c FROM claims")["c"], 1)

        result = {
            "paper_id": "p1",
            "node_id": "ml.test",
            "method_name": "MyMethod",
            "dataset_name": "ToySet",
            "metric_name": "accuracy",
            "metric_value": 0.91,
            "evidence_location": "Table 1",
        }
        result_id_1 = taxonomy.insert_result(dict(result))
        result_id_2 = taxonomy.insert_result(dict(result))
        self.assertEqual(result_id_1, result_id_2)
        self.assertEqual(database.fetchone("SELECT COUNT(*) AS c FROM results")["c"], 1)

    def test_pipeline_events_and_checkpoints_round_trip(self):
        event_id_1 = database.emit_pipeline_event(
            "paper_reasoned",
            {"paper_id": "p1"},
            entity_type="paper",
            entity_id="p1",
            dedupe_key="paper_reasoned:p1",
        )
        event_id_2 = database.emit_pipeline_event(
            "node_touched",
            {"paper_id": "p1", "node_id": "ml.test"},
            entity_type="taxonomy_node",
            entity_id="ml.test",
        )
        events = database.fetch_pipeline_events("consumer_a", limit=10)
        self.assertEqual([event["id"] for event in events], [event_id_1, event_id_2])
        database.ack_pipeline_events("consumer_a", event_id_2)
        self.assertEqual(database.fetch_pipeline_events("consumer_a", limit=10), [])

        database.execute("INSERT INTO papers (id, title) VALUES (?, ?)", ("p1", "Paper 1"))
        database.commit()
        database.record_paper_checkpoint("p1", "extracted", {"claims": 3})
        checkpoint = database.get_paper_checkpoint("p1", "extracted")
        self.assertEqual(checkpoint["payload"]["claims"], 3)

    @mock.patch("orchestrator.discovery_scheduler._refresh_node_outputs")
    def test_discovery_scheduler_consumes_node_events(self, refresh_node_outputs):
        refresh_node_outputs.return_value = {"node_id": "ml.test", "insights_created": []}
        database.emit_pipeline_event(
            "node_touched",
            {"paper_id": "p1", "node_id": "ml.test"},
            entity_type="taxonomy_node",
            entity_id="ml.test",
            dedupe_key="node_touched:p1:ml.test",
        )
        llm_scope = {
            "agenda_id": 1,
            "idea_id": 2,
            "resource_grant_id": 3,
            "stage": "ingestion",
        }
        stats = discovery_scheduler.consume_pipeline_events_once(
            limit=10,
            llm_scope=llm_scope,
        )
        self.assertEqual(stats["nodes_refreshed"], 1)
        refresh_node_outputs.assert_called_once_with(
            "ml.test",
            llm_scope=llm_scope,
        )
        self.assertEqual(database.fetch_pipeline_events(discovery_scheduler.DISCOVERY_CONSUMER, limit=10), [])

    def test_discovery_event_loop_rolls_back_after_refresh_failure(self):
        with (
            mock.patch.object(discovery_scheduler._stop_event, "is_set", side_effect=[False, True]),
            mock.patch.object(discovery_scheduler._stop_event, "wait"),
            mock.patch.object(discovery_scheduler, "consume_pipeline_events_once", side_effect=RuntimeError("pg aborted")),
            mock.patch.object(database, "rollback") as rollback,
        ):
            discovery_scheduler._event_loop()

        rollback.assert_called_once()

class PipelineTextPrefetchTests(TempDbCase):
    def test_prefetch_paper_texts_updates_text_ready_concurrently(self):
        from orchestrator import pipeline

        for idx in range(3):
            database.insert_paper(
                {
                    "id": f"2401.0000{idx}",
                    "title": f"Paper {idx}",
                    "authors": ["A"],
                    "abstract": "short abstract",
                    "categories": ["cs.LG"],
                    "published_date": "2024-01-01",
                    "pdf_url": f"https://example.com/{idx}.pdf",
                }
            )

        def fake_text(arxiv_id, pdf_url="", abstract=""):
            return ("main text for " + arxiv_id + " ") * 80, "appendix"

        with (
            mock.patch.object(pipeline, "PAPER_TEXT_PREFETCH_CONCURRENCY", 3),
            mock.patch.object(pipeline, "get_paper_text_parts", side_effect=fake_text) as get_text,
        ):
            results = pipeline._prefetch_paper_texts([f"2401.0000{idx}" for idx in range(3)])

        self.assertEqual(sum(1 for row in results if row.get("ok")), 3)
        self.assertEqual(get_text.call_count, 3)
        for idx in range(3):
            row = database.fetchone("SELECT full_text, appendix_text, processing_stage FROM papers WHERE id=?", (f"2401.0000{idx}",))
            self.assertIn(f"2401.0000{idx}", row["full_text"])
            self.assertEqual(row["appendix_text"], "appendix")
            self.assertEqual(row["processing_stage"], "text_ready")
            checkpoint = database.get_paper_checkpoint(f"2401.0000{idx}", "text_ready")
            self.assertTrue(checkpoint["payload"].get("prefetched"))


if __name__ == "__main__":
    unittest.main()
