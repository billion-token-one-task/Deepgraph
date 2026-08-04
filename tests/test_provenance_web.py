"""Coverage for the read-only provenance API, the stats cache route, and the
no-server-paths leak guard over public JSON responses."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from db import database
from web import app as web_app
from web import provenance_routes


def _reset_db(tmpdir: Path):
    for attr in ("pg_conn", "sqlite_conn", "conn"):
        if hasattr(database._local, attr):
            try:
                getattr(database._local, attr).close()
            except Exception:
                pass
            setattr(database._local, attr, None)
    database.DATABASE_URL = ""
    database.DB_PATH = tmpdir / "test.db"
    database.init_db()


def _create_meta_harness_tables():
    """Minimal sqlite versions of the migration-0001 tables the provenance
    blueprint reads. Column subsets match the SELECTs in provenance_routes."""
    statements = [
        """CREATE TABLE IF NOT EXISTS research_agendas (
            id INTEGER PRIMARY KEY, name TEXT, description TEXT, status TEXT,
            is_active INTEGER DEFAULT 0, focus_json TEXT,
            token_budget INTEGER DEFAULT 0, token_spent INTEGER DEFAULT 0,
            submitter TEXT, raw_config_json TEXT,
            created_at TEXT DEFAULT '2026-08-01 00:00:00',
            updated_at TEXT DEFAULT '2026-08-01 00:00:00')""",
        """CREATE TABLE IF NOT EXISTS evidence_state_transitions (
            id INTEGER PRIMARY KEY, agenda_id INTEGER, experiment_run_id INTEGER,
            from_state TEXT, to_state TEXT, actor TEXT, context_json TEXT,
            created_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS scientific_decision_records (
            id INTEGER PRIMARY KEY, agenda_id INTEGER, experiment_run_id INTEGER,
            verdict TEXT, verdict_hash TEXT, created_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS agenda_selections (
            id INTEGER PRIMARY KEY, agenda_id INTEGER, selected_insight_id INTEGER,
            score REAL, rationale TEXT, rejected_candidates_json TEXT,
            scoring_breakdown_json TEXT, status TEXT, created_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS idea_decision_packets (
            id INTEGER PRIMARY KEY, agenda_id INTEGER, idea_id INTEGER,
            decision TEXT, reason_codes_json TEXT, candidate_family TEXT,
            revisit_after TEXT, decided_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS resource_grants (
            id INTEGER PRIMARY KEY, agenda_id INTEGER, idea_id INTEGER,
            stage TEXT, token_cap INTEGER, max_gpu_hours REAL, status TEXT,
            grant_reason TEXT, created_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS compute_jobs_v1 (
            id INTEGER PRIMARY KEY, agenda_id INTEGER, idea_id INTEGER,
            stage TEXT, backend_kind TEXT, status TEXT, failure_reason TEXT,
            created_at TEXT, updated_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS frontier_packets (
            id INTEGER PRIMARY KEY, agenda_id INTEGER, research_problem_id INTEGER,
            gate_allowed INTEGER, gate_reason_codes_json TEXT, created_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS outcome_records (
            id INTEGER PRIMARY KEY, agenda_id INTEGER, idea_id INTEGER,
            experiment_run_id INTEGER, execution_result TEXT, effect REAL,
            baseline REAL, verdict TEXT, state_decision TEXT, recorded_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS legacy_scope_imports (
            id INTEGER PRIMARY KEY, agenda_id INTEGER, entity_type TEXT,
            entity_id INTEGER, actor TEXT, reason TEXT, idempotency_key TEXT,
            imported_at TEXT)""",
    ]
    for statement in statements:
        database.execute(statement)
    database.commit()


class ProvenanceApiTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.old_db_path = database.DB_PATH
        self.old_database_url = database.DATABASE_URL
        _reset_db(Path(self.tmpdir.name))
        _create_meta_harness_tables()
        self.client = web_app.app.test_client()

    def tearDown(self):
        database.DB_PATH = self.old_db_path
        database.DATABASE_URL = self.old_database_url
        self.tmpdir.cleanup()

    def test_agenda_list_is_public_but_excludes_submitter_and_raw_config(self):
        database.execute(
            """INSERT INTO research_agendas
               (id, name, description, status, is_active, focus_json,
                token_budget, token_spent, submitter, raw_config_json)
               VALUES (1, 'agenda-one', 'desc', 'active', 1, '["probing"]',
                       1000, 250, 'someone@example.com', '{"secret": "x"}')"""
        )
        payload = self.client.get("/api/v1/agendas").get_json()
        agendas = payload["agendas"]
        self.assertEqual(len(agendas), 1)
        self.assertEqual(agendas[0]["id"], 1)
        self.assertTrue(agendas[0]["is_active"])
        self.assertEqual(agendas[0]["budget_pct"], 25.0)
        self.assertNotIn("submitter", agendas[0])
        self.assertNotIn("raw_config_json", agendas[0])

    def test_evidence_states_requires_agenda_and_defaults_to_not_assessed(self):
        self.assertEqual(self.client.get("/api/v1/evidence_states").status_code, 400)

        # A run that merely exists (operationally complete or not) has no
        # entry: the UI renders that as "not assessed".
        database.execute(
            "INSERT INTO deep_insights (id, tier, title) VALUES (5, 2, 'Idea Five')"
        )
        database.execute(
            "INSERT INTO experiment_runs (id, deep_insight_id, status) VALUES (77, 5, 'completed')"
        )
        try:
            database.execute("ALTER TABLE experiment_runs ADD COLUMN agenda_id INTEGER")
        except Exception:
            pass
        database.execute("UPDATE experiment_runs SET agenda_id=1 WHERE id=77")
        payload = self.client.get("/api/v1/evidence_states?agenda_id=1").get_json()
        self.assertEqual(payload["runs"], {})
        self.assertEqual(payload["ideas"], {})

    def test_evidence_states_rolls_up_ladder_and_verdict(self):
        try:
            database.execute("ALTER TABLE experiment_runs ADD COLUMN agenda_id INTEGER")
        except Exception:
            pass
        database.execute(
            "INSERT INTO deep_insights (id, tier, title) VALUES (3, 2, 'Idea Three')"
        )
        database.execute(
            "INSERT INTO experiment_runs (id, deep_insight_id, status, agenda_id) VALUES (10, 3, 'completed', 1)"
        )
        for i, (frm, to) in enumerate([
            ("planned", "sanity_passed"),
            ("sanity_passed", "full_benchmark_complete"),
            ("full_benchmark_complete", "evidence_audited"),
            ("evidence_audited", "scientifically_decided"),
        ]):
            database.execute(
                """INSERT INTO evidence_state_transitions
                   (agenda_id, experiment_run_id, from_state, to_state, actor, created_at)
                   VALUES (1, 10, ?, ?, 'gate', ?)""",
                (frm, to, f"2026-08-01 00:0{i}:00"),
            )
        database.execute(
            """INSERT INTO scientific_decision_records
               (agenda_id, experiment_run_id, verdict, verdict_hash, created_at)
               VALUES (1, 10, 'supported', 'sha256:abc', '2026-08-01 00:05:00')"""
        )
        payload = self.client.get("/api/v1/evidence_states?agenda_id=1").get_json()
        self.assertEqual(payload["runs"]["10"]["state"], "scientifically_decided")
        self.assertEqual(payload["runs"]["10"]["verdict"], "supported")
        self.assertEqual(payload["ideas"]["3"]["state"], "scientifically_decided")
        self.assertEqual(payload["ideas"]["3"]["run_id"], 10)

    def test_timeline_merges_events_and_scrubs_server_paths(self):
        database.execute(
            """INSERT INTO compute_jobs_v1
               (agenda_id, idea_id, stage, backend_kind, status, failure_reason,
                created_at, updated_at)
               VALUES (1, 3, 'pilot', 'colab', 'failed',
                       'Traceback in /home/billion-token/Deepgraph/run.py line 5',
                       '2026-08-01 01:00:00', '2026-08-01 01:00:00')"""
        )
        database.execute(
            """INSERT INTO resource_grants
               (agenda_id, idea_id, stage, token_cap, max_gpu_hours, status,
                grant_reason, created_at)
               VALUES (1, 3, 'pilot', 250000, 2.0, 'active',
                       'artifacts under /home/billion-token/x', '2026-08-01 00:30:00')"""
        )
        database.execute(
            """INSERT INTO frontier_packets
               (agenda_id, research_problem_id, gate_allowed,
                gate_reason_codes_json, created_at)
               VALUES (1, 9, 0, '["obsolete_evidence"]', '2026-08-01 00:10:00')"""
        )
        database.execute(
            """INSERT INTO legacy_scope_imports
               (agenda_id, entity_type, entity_id, actor, reason,
                idempotency_key, imported_at)
               VALUES (1, 'deep_insight', 7, 'legacy_scope_backfill',
                       'artifacts were in /home/billion-token/x', 'k1',
                       '2026-08-01 00:05:00')"""
        )
        database.commit()
        payload = self.client.get("/api/v1/agendas/1/timeline").get_json()
        events = payload["events"]
        self.assertEqual(
            [e["kind"] for e in events],
            ["job", "authorization", "signal", "legacy_import"],
        )
        body = str(payload)
        self.assertNotIn("/home/", body)
        self.assertIn("<path>", body)
        signal = events[2]
        self.assertFalse(signal["gate_allowed"])
        self.assertEqual(signal["gate_reason_codes"], ["obsolete_evidence"])
        legacy = events[3]
        self.assertEqual(legacy["entity_id"], 7)
        self.assertEqual(legacy["actor"], "legacy_scope_backfill")

    def test_selection_rationale_exposes_rejected_candidates(self):
        database.execute(
            """INSERT INTO agenda_selections
               (agenda_id, selected_insight_id, score, rationale,
                rejected_candidates_json, scoring_breakdown_json, status, created_at)
               VALUES (1, 4, 0.81, 'highest contribution delta',
                       '[{"insight_id": 7, "score": 0.5, "reason": "obsolete"}]',
                       '{}', 'selected', '2026-08-01 00:00:00')"""
        )
        payload = self.client.get("/api/v1/agendas/1/selection").get_json()
        selection = payload["selections"][0]
        self.assertEqual(selection["selected_insight_id"], 4)
        self.assertIn("contribution delta", selection["rationale"])
        self.assertEqual(selection["rejected_candidates"][0]["insight_id"], 7)

    def test_provenance_survives_missing_tables(self):
        database.execute("DROP TABLE evidence_state_transitions")
        payload = self.client.get("/api/v1/evidence_states?agenda_id=1").get_json()
        self.assertEqual(payload["runs"], {})


class StatsCacheRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = web_app.app.test_client()

    def test_stats_returns_warming_before_prewarm_and_snapshot_after(self):
        with mock.patch.object(
            web_app, "_stats_cache", web_app.StatsCache(lambda: {"papers_total": 7})
        ) as cache:
            payload = self.client.get("/api/stats").get_json()
            self.assertEqual(payload, {"warming": True})

            cache.prewarm()
            payload = self.client.get("/api/stats").get_json()
            self.assertEqual(payload["papers_total"], 7)
            # Served from cache: no recompute per request.
            self.client.get("/api/stats")
            self.assertEqual(cache.compute_count, 1)


class LeakGuardTests(unittest.TestCase):
    """No public JSON response may carry server filesystem paths, raw log
    content, or the private keys stripped by the response scrubber."""

    FORBIDDEN_SUBSTRINGS = (
        "/home/", "log_tail", "workspace_root", "experiment_root",
        "plan_root", "paper_root", "binary_path", "research_workdir",
    )

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.old_db_path = database.DB_PATH
        self.old_database_url = database.DATABASE_URL
        _reset_db(Path(self.tmpdir.name))
        _create_meta_harness_tables()
        self.client = web_app.app.test_client()

    def tearDown(self):
        database.DB_PATH = self.old_db_path
        database.DATABASE_URL = self.old_database_url
        self.tmpdir.cleanup()

    def test_scrubber_strips_private_keys_recursively(self):
        scrubbed = web_app._scrub_private(
            {
                "ok": 1,
                "workdir": "/home/x",
                "nested": [{"log_tail": "boom", "keep": True}],
            }
        )
        self.assertEqual(scrubbed, {"ok": 1, "nested": [{"keep": True}]})

    def test_parameterless_get_routes_never_leak_paths_or_logs(self):
        skipped_prefixes = ("/static", "/api/events", "/papers")
        for rule in web_app.app.url_map.iter_rules():
            if "GET" not in rule.methods:
                continue
            if rule.arguments:
                continue
            url = str(rule)
            if url.startswith(skipped_prefixes) or url == "/":
                continue
            with self.subTest(url=url):
                response = self.client.get(url)
                body = response.get_data(as_text=True) or ""
                for token in self.FORBIDDEN_SUBSTRINGS:
                    self.assertNotIn(token, body, f"{url} leaked {token}")

    def test_automation_endpoint_carries_no_workdir_or_log_content(self):
        response = self.client.get("/api/automation")
        body = response.get_data(as_text=True) or ""
        for token in self.FORBIDDEN_SUBSTRINGS:
            self.assertNotIn(token, body)

    def test_api_failure_returns_generic_message_with_correlation_id(self):
        with web_app.app.test_request_context("/"):
            response, status = web_app._api_failure(
                "test_scope", RuntimeError("secret /home/billion-token detail")
            )
        self.assertEqual(status, 500)
        payload = response.get_json()
        self.assertNotIn("secret", str(payload))
        self.assertNotIn("/home/", str(payload))
        self.assertTrue(payload["correlation_id"])


if __name__ == "__main__":
    unittest.main()
