import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.paper_idea_agent import discover_paper_ideas
from agents.problem_first import discover_research_problems, problem_first_cycle, writeback_experiment_result
from agents.signal_harvester import harvest_protocol_artifacts
from db import database


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
        # meta-harness-v1 is PostgreSQL-first. These compatibility columns keep
        # the legacy SQLite unit fixture scoped without pretending to validate
        # the PostgreSQL migration.
        for table in (
            "research_problems",
            "deep_insights",
            "experimental_evidence_edges",
        ):
            database.execute(f"ALTER TABLE {table} ADD COLUMN agenda_id INTEGER")
        database.execute(
            """
            CREATE TABLE agenda_signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agenda_id INTEGER NOT NULL,
                run_id INTEGER,
                experimental_claim_id INTEGER,
                signal_table TEXT NOT NULL,
                signal_content_hash TEXT NOT NULL,
                verdict TEXT NOT NULL,
                effect_size REAL,
                p_value REAL,
                conditions_json TEXT NOT NULL DEFAULT '{}',
                idempotency_key TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (agenda_id, idempotency_key)
            )
            """
        )
        database.commit()

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


class ProblemFirstTests(TempDbTestCase):
    def setUp(self):
        super().setUp()
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
        harvest_protocol_artifacts(min_support=1)

    def test_discover_research_problems_promotes_problem_signal(self):
        problems = discover_research_problems(limit=5, agenda_id=1, persist=True)
        self.assertTrue(problems)
        problem = problems[0]
        self.assertGreater(problem["problem_quality_score"], 0)
        self.assertEqual(problem["source_signal_ref"]["table"], "protocol_artifacts")
        self.assertIn("p1", problem["paper_ids"])
        stored = database.fetchone("SELECT * FROM research_problems WHERE id=?", (problem["id"],))
        self.assertIsNotNone(stored)

    def test_problem_first_cycle_records_inconclusive_attempts(self):
        problem = discover_research_problems(limit=1, agenda_id=1, persist=True)[0]

        def _worker(_problem, approach):
            return {
                "verdict": "inconclusive",
                "conditions": {"reason": "dataset unavailable"},
                "effect_size": None,
                "run_id": None,
                "deep_insight_id": None,
                "source_signal_refs": approach.get("source_signal_refs"),
            }

        with mock.patch(
            "agents.problem_first.propose_approach",
            return_value={"summary": "try protocol-robust calibration", "source_signal_refs": {}},
        ):
            result = problem_first_cycle(agenda_id=1, max_attempts=1, worker=_worker)

        self.assertEqual(result["status"], "attempt_limit")
        stored_problem = database.fetchone(
            "SELECT attempts_count, status, ruled_out_approaches FROM research_problems WHERE id=?",
            (problem["id"],),
        )
        self.assertEqual(stored_problem["attempts_count"], 1)
        self.assertEqual(stored_problem["status"], "exploring")
        attempts = json.loads(stored_problem["ruled_out_approaches"])
        self.assertEqual(attempts[-1]["verdict"], "inconclusive")

    def test_positive_worker_result_cannot_solve_without_decided_run(self):
        problem = discover_research_problems(limit=1, agenda_id=1, persist=True)[0]

        def _worker(_problem, approach):
            return {
                "verdict": "confirmed",
                "run_id": None,
                "source_signal_refs": approach.get("source_signal_refs"),
            }

        with mock.patch(
            "agents.problem_first.propose_approach",
            return_value={"summary": "candidate", "source_signal_refs": {}},
        ):
            result = problem_first_cycle(
                agenda_id=1,
                max_attempts=1,
                worker=_worker,
            )

        self.assertEqual(result["status"], "awaiting_scientific_decision")
        stored = database.fetchone(
            "SELECT status FROM research_problems WHERE id=?",
            (problem["id"],),
        )
        self.assertNotEqual(stored["status"], "solved")

    def test_writeback_experiment_result_updates_posterior_and_problem_state(self):
        problem = discover_research_problems(limit=1, agenda_id=1, persist=True)[0]
        signal_refs = {
            "signals": [problem["source_signal_ref"]],
            "node_ids": problem["node_ids"],
            "paper_ids": problem["paper_ids"],
        }
        insight_id = database.insert_returning_id(
            """
            INSERT INTO deep_insights
              (agenda_id, tier, title, problem_statement, source_node_ids, source_paper_ids,
               source_signal_ids, source_signal_refs, research_problem_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                1,
                2,
                "Protocol-aware idea",
                problem["problem_statement"],
                json.dumps(problem["node_ids"]),
                json.dumps(problem["paper_ids"]),
                json.dumps([problem["source_signal_ref"]["content_hash"]]),
                json.dumps(signal_refs),
                problem["id"],
            ),
        )
        database.commit()

        summary = writeback_experiment_result(
            agenda_id=1,
            run_id=None,
            deep_insight_id=insight_id,
            verdict="refuted",
            effect_size=-0.15,
            conditions={"dataset": "toy"},
            source_signal_refs=signal_refs,
            experimental_claim_id=11,
        )

        self.assertEqual(len(summary["updated_signals"]), 1)
        row = database.fetchone(
            """
            SELECT verdict, signal_table, signal_content_hash
            FROM agenda_signal_outcomes
            WHERE agenda_id=?
            """,
            (1,),
        )
        self.assertEqual(row["verdict"], "refuted")
        self.assertEqual(row["signal_table"], "protocol_artifacts")
        self.assertEqual(
            row["signal_content_hash"],
            problem["source_signal_ref"]["content_hash"],
        )

        stored_problem = database.fetchone(
            "SELECT attempts_count, ruled_out_approaches FROM research_problems WHERE id=?",
            (problem["id"],),
        )
        self.assertEqual(stored_problem["attempts_count"], 1)
        ruled_out = json.loads(stored_problem["ruled_out_approaches"])
        self.assertTrue(ruled_out)

        edge_count = database.fetchone("SELECT COUNT(*) AS c FROM experimental_evidence_edges")
        self.assertGreaterEqual(edge_count["c"], 2)

    def test_discover_paper_ideas_prefers_problem_first_pool(self):
        problem = {
            "id": 7,
            "research_problem_id": 7,
            "title": "Protocol artifact problem",
            "problem_statement": "Benchmark protocol hides the true failure mode.",
            "formal_statement": "Benchmark protocol hides the true failure mode.",
            "current_failure_mode": "Metric-sensitive protocol artifact.",
            "desideratum": "A method that is robust to protocol shifts.",
            "central_question": "Can we remove protocol dependence?",
            "motivation": "Recent papers disagree because of protocol mismatch.",
            "result_that_would_change_belief": "Robustness under swapped protocol settings.",
            "mechanism_type": "protocol_artifact",
            "source_type": "protocol_artifact",
            "source_evidence": "Benchmark protocol choices dominate the score.",
            "non_numeric_evidence": ["Benchmark protocol choices dominate the score."],
            "difficulty": "medium",
            "impact_scope": "1 supporting papers across 1 taxonomy areas",
            "related_node_ids": ["ml.test"],
            "source_paper_ids": ["p1"],
            "source_signal_refs": {"signals": [], "node_ids": ["ml.test"], "paper_ids": ["p1"]},
            "ruled_out_approaches": [],
        }
        signals = {
            "contradiction_clusters": [],
            "performance_plateaus": [],
            "limitation_clusters": [],
            "high_potential_insights": [],
            "mechanism_mismatches": [],
            "protocol_artifacts": [{"id": 1, "summary": "protocol artifact"}],
            "negative_space_gaps": [],
            "hidden_variable_bridges": [],
            "claim_method_gaps": [],
        }
        llm_outputs = [
            (
                {
                    "method": {
                        "name": "ProtocolShield",
                        "type": "training_procedure",
                        "one_line": "Reduce protocol sensitivity.",
                        "definition": "min_theta L(theta)",
                        "key_properties": ["Controls protocol shift."],
                        "why_novel": "Unlike prior work, it explicitly regularizes protocol shifts over benchmark interfaces.",
                        "limitations": "May fail under extreme distribution shift.",
                        "mechanism_repair": "Decouples benchmark protocol from the learned objective.",
                        "falsification_hook": "No gain under changed protocol settings.",
                    }
                },
                123,
                {"provider": "p", "model": "m"},
            ),
            (
                {
                    "paper_title": "ProtocolShield: Auditing Protocol-Sensitive Benchmarks",
                    "baselines": [],
                    "datasets": [],
                    "metrics": {},
                    "ablations": [],
                    "expected_results": {"solid": "Improves robustness."},
                    "compute_budget": {},
                    "risks": [],
                    "paper_outline": {},
                    "problem_awareness": {},
                },
                87,
                {"provider": "p", "model": "m"},
            ),
        ]
        with (
            mock.patch("agents.paper_idea_agent.get_tier2_signals", return_value=signals),
            mock.patch("agents.paper_idea_agent.select_problem_first_candidates", return_value=[problem]),
            mock.patch(
                "agents.paper_idea_agent.call_llm_json_for_role",
                side_effect=llm_outputs,
            ) as call_llm_json,
            mock.patch(
                "agents.paper_idea_agent._proposal_candidate_and_grant",
                return_value=(71, {"id": 91, "token_cap": 1000}),
            ),
            mock.patch(
                "agents.paper_idea_agent.configured_role_prompt_version",
                return_value="proposer-v1",
            ),
            mock.patch("agents.paper_idea_agent.get_solution_signals", return_value=[]),
            mock.patch("agents.paper_idea_agent.graph_novelty_gate", return_value=None),
            mock.patch("agents.paper_idea_agent._find_existing_tier2_duplicate", return_value=None),
            mock.patch("agents.paper_idea_agent.get_evosci_input_issue", return_value=None),
            mock.patch("agents.paper_idea_agent.attach_graph_taste_to_insight", side_effect=lambda x: x),
            mock.patch("agents.paper_idea_agent.enrich_deep_insight", side_effect=lambda x: x),
            mock.patch("agents.paper_idea_agent.TIER2_EVOSCI_PREINSERT_REVIEW", False),
        ):
            ideas = discover_paper_ideas(
                max_problems=1,
                max_papers=1,
                agenda_id=1,
            )

        self.assertEqual(len(ideas), 1)
        self.assertEqual(ideas[0]["research_problem_id"], 7)
        self.assertEqual(ideas[0]["proposal_candidate_id"], 71)
        self.assertEqual(ideas[0]["resource_grant_id"], 91)
        self.assertEqual(call_llm_json.call_count, 2)
        self.assertIn("source_signal_refs", ideas[0])


class ProblemFirstMigrationTests(unittest.TestCase):
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

    def test_init_db_upgrades_legacy_signal_tables_before_schema_v2_indexes(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE node_entity_overlap (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_a_id TEXT NOT NULL,
                node_b_id TEXT NOT NULL,
                shared_entity_count INTEGER NOT NULL,
                shared_entity_ids TEXT,
                shared_entity_types TEXT,
                taxonomic_distance INTEGER DEFAULT 0,
                overlap_score REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
        conn.close()

        database.init_db()

        cols = {
            row["name"]
            for row in database.fetchall("PRAGMA table_info(node_entity_overlap)")
        }
        self.assertIn("content_hash", cols)
        self.assertIn("signal_role", cols)
        self.assertIn("empirical_posterior", cols)
        self.assertIsNotNone(
            database.fetchone(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_node_entity_overlap_content_hash'"
            )
        )


if __name__ == "__main__":
    unittest.main()
