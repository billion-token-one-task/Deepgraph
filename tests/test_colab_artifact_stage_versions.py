"""Stage-versioned Colab artifacts keep every registered byte sequence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from meta_harness.compute import ComputeBackendError
from orchestrator import colab_worker
from scripts import backfill_artifact_stage_versions as backfill
from scripts.meta_harness_migration import MIGRATION_KEYS, migration_plan


class _SQLiteRegistry:
    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE resource_grants (
                id INTEGER PRIMARY KEY,
                stage TEXT NOT NULL,
                preflight_result_id INTEGER
            );
            CREATE TABLE colab_work_requests_v1 (
                id INTEGER PRIMARY KEY,
                agenda_id INTEGER NOT NULL,
                idea_id INTEGER NOT NULL,
                experiment_run_id INTEGER NOT NULL,
                resource_grant_id INTEGER NOT NULL,
                stage TEXT NOT NULL,
                artifact_output_dir TEXT NOT NULL,
                status TEXT NOT NULL,
                completed_at TIMESTAMP
            );
            CREATE TABLE experiment_runs (
                id INTEGER PRIMARY KEY,
                agenda_id INTEGER NOT NULL,
                deep_insight_id INTEGER NOT NULL,
                resource_grant_id INTEGER NOT NULL,
                workdir TEXT NOT NULL,
                scientific_evidence_state TEXT,
                status TEXT,
                phase TEXT,
                error_message TEXT,
                baseline_metric_name TEXT,
                baseline_metric_value REAL,
                best_metric_value REAL,
                effect_size REAL,
                effect_pct REAL,
                hypothesis_verdict TEXT,
                completed_at TIMESTAMP
            );
            CREATE TABLE experiment_artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agenda_id INTEGER NOT NULL,
                run_id INTEGER NOT NULL,
                artifact_type TEXT NOT NULL,
                path TEXT NOT NULL,
                artifact_stage TEXT,
                artifact_version INTEGER NOT NULL DEFAULT 1,
                content_sha256 TEXT,
                metric_key TEXT,
                metric_value REAL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX idx_experiment_artifacts_stage_content
                ON experiment_artifacts(
                    run_id, artifact_type, artifact_stage, content_sha256
                )
                WHERE artifact_stage IS NOT NULL
                  AND content_sha256 IS NOT NULL;
            """
        )

    def execute(self, sql: str, params=()):
        return self.connection.execute(sql, params)

    def fetchone(self, sql: str, params=()):
        row = self.connection.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def fetchall(self, sql: str, params=()):
        return [dict(row) for row in self.connection.execute(sql, params).fetchall()]

    def commit(self) -> None:
        self.connection.commit()


class ArtifactStageVersionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = _SQLiteRegistry()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workdir = Path(self.temp.name)
        self.results = self.workdir / "results"
        self.results.mkdir()
        self.source = self.results / "final_results.json"
        self.run = {"id": 246, "agenda_id": 14, "workdir": str(self.workdir)}

    def _register(self, content: bytes, *, stage: str, grant_id: int) -> int:
        self.source.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        with mock.patch.object(colab_worker, "db", self.registry):
            return colab_worker._register_artifact_version(
                run=self.run,
                resource_grant_id=grant_id,
                artifact_stage=stage,
                artifact_type="final_results",
                source_path=self.source.resolve(),
                content=content,
                content_sha256=digest,
                metric_key="accuracy",
                metric_value=0.63 if stage == "full_benchmark" else 0.41,
            )

    def _rows(self):
        return self.registry.fetchall(
            """
            SELECT id, artifact_type, path, artifact_stage, artifact_version,
                   content_sha256, metadata
            FROM experiment_artifacts
            ORDER BY id
            """
        )

    def test_pilot_and_full_benchmark_append_distinct_immutable_versions(self):
        pilot = b'{"stage":"pilot","score":0.41}'
        full = b'{"stage":"full_benchmark","score":0.63}'

        pilot_id = self._register(pilot, stage="pilot", grant_id=61)
        full_id = self._register(full, stage="full_benchmark", grant_id=72)

        rows = self._rows()
        self.assertEqual([row["id"] for row in rows], [pilot_id, full_id])
        self.assertEqual(
            [row["artifact_stage"] for row in rows], ["pilot", "full_benchmark"]
        )
        self.assertEqual([row["artifact_version"] for row in rows], [1, 2])
        self.assertNotEqual(rows[0]["path"], rows[1]["path"])
        self.assertEqual(Path(rows[0]["path"]).read_bytes(), pilot)
        self.assertEqual(Path(rows[1]["path"]).read_bytes(), full)
        self.assertEqual(
            rows[0]["content_sha256"], hashlib.sha256(pilot).hexdigest()
        )
        self.assertEqual(
            rows[1]["content_sha256"], hashlib.sha256(full).hexdigest()
        )
        full_metadata = json.loads(rows[1]["metadata"])
        self.assertTrue(full_metadata["immutable_snapshot"])
        self.assertEqual(full_metadata["source_path"], str(self.source.resolve()))
        self.assertEqual(full_metadata["supersedes_artifact_id"], pilot_id)

        # Reconciliation may replay a succeeded request. Identical stage bytes
        # resolve to the existing row rather than manufacturing a third version.
        self.assertEqual(
            self._register(full, stage="full_benchmark", grant_id=72), full_id
        )
        self.assertEqual(len(self._rows()), 2)

    def test_legacy_pilot_row_is_not_rewritten_when_full_version_is_added(self):
        pilot_hash = hashlib.sha256(b"pilot bytes no longer on source path").hexdigest()
        old_metadata = json.dumps({"sha256": pilot_hash, "source": "legacy"})
        cursor = self.registry.execute(
            """
            INSERT INTO experiment_artifacts
                (agenda_id, run_id, artifact_type, path, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (14, 246, "final_results", str(self.source), old_metadata),
        )
        legacy_id = int(cursor.lastrowid)
        full = b'{"stage":"full_benchmark","score":0.63}'

        current_id = self._register(full, stage="full_benchmark", grant_id=72)
        rows = self._rows()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], legacy_id)
        self.assertEqual(rows[0]["path"], str(self.source))
        self.assertEqual(rows[0]["metadata"], old_metadata)
        self.assertIsNone(rows[0]["artifact_stage"])
        self.assertEqual(rows[1]["id"], current_id)
        self.assertEqual(rows[1]["artifact_stage"], "full_benchmark")
        self.assertEqual(rows[1]["artifact_version"], 2)
        self.assertEqual(Path(rows[1]["path"]).read_bytes(), full)

    def test_existing_snapshot_with_wrong_bytes_fails_closed(self):
        content = b"verified full benchmark"
        row_id = self._register(content, stage="full_benchmark", grant_id=72)
        snapshot = Path(self._rows()[0]["path"])
        snapshot.chmod(0o644)
        snapshot.write_bytes(b"tampered")

        with self.assertRaisesRegex(
            ComputeBackendError, "artifact_snapshot_hash_mismatch"
        ):
            self._register(content, stage="full_benchmark", grant_id=72)
        self.assertEqual(self._rows()[0]["id"], row_id)

    def test_terminal_handoff_derives_artifact_stage_from_its_grant(self):
        self.source.write_text("{}", encoding="utf-8")
        self.registry.execute(
            "INSERT INTO resource_grants (id, stage) VALUES (?, ?)",
            (72, "full_benchmark"),
        )
        self.registry.execute(
            """
            INSERT INTO experiment_runs
                (id, agenda_id, deep_insight_id, resource_grant_id, workdir,
                 scientific_evidence_state, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (246, 14, 91, 72, str(self.workdir), "full_benchmark_complete", "running"),
        )
        payload = {
            "artifacts": {"final_results": {"path": "final_results.json"}},
            "artifact_hashes": {},
            "hypothesis_verdict": "supported",
        }
        verification = SimpleNamespace(
            metric_name="accuracy",
            baseline_value=0.32,
            candidate_value=0.63,
            direction="higher",
        )
        request = {
            "experiment_run_id": 246,
            "agenda_id": 14,
            "idea_id": 91,
            "resource_grant_id": 72,
            "stage": "full_benchmark",
            "artifact_output_dir": str(self.results),
        }

        with (
            mock.patch.object(colab_worker, "db", self.registry),
            mock.patch.object(colab_worker, "validate_final_results", return_value=payload),
            mock.patch.object(
                colab_worker, "verify_metric_from_artifacts", return_value=verification
            ),
        ):
            colab_worker._record_terminal_run_success(
                request, SimpleNamespace(status="succeeded")
            )

        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["artifact_stage"], "full_benchmark")
        self.assertEqual(Path(rows[0]["path"]).read_text(encoding="utf-8"), "{}")
        run = self.registry.fetchone(
            "SELECT status, phase, hypothesis_verdict FROM experiment_runs WHERE id=?",
            (246,),
        )
        self.assertEqual(
            run,
            {
                "status": "completed",
                "phase": "colab_result_verified",
                "hypothesis_verdict": "supported",
            },
        )

    def test_durable_output_directory_cannot_escape_the_run_workdir(self):
        outside = self.workdir.parent / "other-run-results"
        with self.assertRaisesRegex(
            ComputeBackendError, "artifact_output_dir_outside_run_workdir"
        ):
            colab_worker._request_artifact_output_dir(
                self.run, {"artifact_output_dir": str(outside)}
            )


class ArtifactStageMigrationTests(unittest.TestCase):
    def test_migration_is_registered_and_additive(self):
        self.assertIn("0007_artifact_stage_versions", MIGRATION_KEYS)
        self.assertLess(
            MIGRATION_KEYS.index("0007_artifact_stage_versions"),
            MIGRATION_KEYS.index("0008_manuscript_gate_records"),
        )
        plan = migration_plan("0007_artifact_stage_versions")
        self.assertEqual(plan["destructive_tokens"], [])
        self.assertEqual(plan["statement_count"], 5)

    def test_fresh_schemas_expose_legacy_columns_plus_stage_version_fields(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("schema_v2.sql", "schema_postgres.sql"):
            schema = (root / "db" / name).read_text(encoding="utf-8")
            table = schema.split("CREATE TABLE IF NOT EXISTS experiment_artifacts", 1)[1]
            table = table.split(");", 1)[0]
            with self.subTest(schema=name):
                for legacy in ("artifact_type", "path", "metric_key", "metadata"):
                    self.assertIn(legacy, table)
                for added in (
                    "artifact_stage",
                    "artifact_version",
                    "content_sha256",
                ):
                    self.assertIn(added, table)


class ArtifactStageBackfillTests(unittest.TestCase):
    def test_operator_help_runs_outside_the_repository_working_directory(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "backfill_artifact_stage_versions.py"
        )
        with tempfile.TemporaryDirectory() as outside:
            completed = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=outside,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--run-id", completed.stdout)

    def test_dry_run_then_apply_appends_full_and_holdout_and_marks_lost_pilot(self):
        registry = _SQLiteRegistry()
        with tempfile.TemporaryDirectory() as temp:
            workdir = Path(temp)
            full_dir = workdir / "results"
            holdout_dir = workdir / "results_holdout"
            full_dir.mkdir()
            holdout_dir.mkdir()
            full_path = full_dir / "final_results.json"
            holdout_path = holdout_dir / "final_results.json"
            full = b"full benchmark bytes"
            holdout = b"holdout bytes"
            full_path.write_bytes(full)
            holdout_path.write_bytes(holdout)
            registry.execute(
                """
                INSERT INTO experiment_runs
                    (id, agenda_id, deep_insight_id, resource_grant_id, workdir,
                     scientific_evidence_state, status)
                VALUES (246, 14, 91, 72, ?, 'scientifically_decided', 'completed')
                """,
                (str(workdir),),
            )
            registry.execute(
                "INSERT INTO resource_grants (id, stage) VALUES (72, 'full_benchmark')"
            )
            registry.execute(
                "INSERT INTO resource_grants (id, stage) VALUES (73, 'evidence_audit')"
            )
            registry.execute(
                """
                INSERT INTO colab_work_requests_v1
                    (id, agenda_id, idea_id, experiment_run_id, resource_grant_id,
                     stage, artifact_output_dir, status, completed_at)
                VALUES (140, 14, 91, 246, 72, 'full_benchmark', ?, 'succeeded', 1)
                """,
                (str(full_dir),),
            )
            registry.execute(
                """
                INSERT INTO colab_work_requests_v1
                    (id, agenda_id, idea_id, experiment_run_id, resource_grant_id,
                     stage, artifact_output_dir, status, completed_at)
                VALUES (141, 14, 91, 246, 73, 'evidence_audit', ?, 'succeeded', 2)
                """,
                (str(holdout_dir),),
            )
            pilot_sha = hashlib.sha256(b"unavailable pilot bytes").hexdigest()
            cursor = registry.execute(
                """
                INSERT INTO experiment_artifacts
                    (agenda_id, run_id, artifact_type, path, metadata)
                VALUES (14, 246, 'final_results', ?, ?)
                """,
                (str(full_path), json.dumps({"sha256": pilot_sha})),
            )
            pilot_id = int(cursor.lastrowid)

            def verified(_run, request):
                source = Path(str(request["artifact_output_dir"])) / "final_results.json"
                content = source.read_bytes()
                return (
                    {},
                    SimpleNamespace(metric_name="accuracy", candidate_value=0.63),
                    [
                        {
                            "artifact_type": "final_results",
                            "source_path": source.resolve(),
                            "content": content,
                            "content_sha256": hashlib.sha256(content).hexdigest(),
                        }
                    ],
                )

            with (
                mock.patch.object(backfill, "db", registry),
                mock.patch.object(colab_worker, "db", registry),
                mock.patch.object(backfill, "_verified_request_artifacts", side_effect=verified),
            ):
                plan = backfill.run(run_id=246)
                self.assertEqual(plan["mode"], "dry_run")
                self.assertFalse((workdir / ".artifact-history-v1").exists())
                self.assertEqual(len(registry.fetchall("SELECT id FROM experiment_artifacts")), 1)
                self.assertEqual(
                    plan["legacy_unresolved"]["final_results"][0]["artifact_id"],
                    pilot_id,
                )

                applied = backfill.run(run_id=246, apply=True)

            self.assertEqual(applied["mode"], "apply")
            rows = registry.fetchall(
                """
                SELECT id, path, artifact_stage, artifact_version, metadata
                FROM experiment_artifacts ORDER BY id
                """
            )
            self.assertEqual(len(rows), 3)
            self.assertIsNone(rows[0]["artifact_stage"])
            self.assertEqual(
                [row["artifact_stage"] for row in rows[1:]],
                ["full_benchmark", "evidence_audit"],
            )
            self.assertEqual([row["artifact_version"] for row in rows], [1, 2, 3])
            self.assertEqual(Path(rows[1]["path"]).read_bytes(), full)
            self.assertEqual(Path(rows[2]["path"]).read_bytes(), holdout)
            for row in rows[1:]:
                issue = json.loads(row["metadata"])["legacy_unresolved"][0]
                self.assertEqual(issue["artifact_id"], pilot_id)
                self.assertEqual(issue["status"], "legacy_unresolved")


if __name__ == "__main__":
    unittest.main()
