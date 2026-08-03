"""The deployer must refuse to install anything the manifest does not pin."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from unittest import mock

from scripts import deploy_manifest_batch as deployer


ROOT = Path(__file__).resolve().parents[1]


def _manifest(**overrides) -> dict:
    manifest = {
        "manifest_key": "test-manifest",
        "source_commit": "0" * 40,
        "backup_root": "/tmp/does-not-exist",
        "batches": [
            {
                "batch": "1-selfheal",
                "systemd_reload_required": True,
                "restart_required": [],
                "health_check": "check",
                "files": [
                    {
                        "source": "scripts/deepgraph_selfheal.py",
                        "target": "/usr/local/bin/deepgraph-selfheal.py",
                        "owner": "root:root",
                        "mode": "0755",
                        "source_sha256": "a" * 64,
                    }
                ],
            },
            {
                "batch": "2-schema",
                "files": [
                    {
                        "source": "db/migrations/0002_topic_gate_and_frontier_authority.sql",
                        "target": "applied-via-scripts/meta_harness_migration.py",
                        "kind": "migration",
                        "source_sha256": "b" * 64,
                    }
                ],
            },
        ],
    }
    manifest.update(overrides)
    return manifest


class BatchSelectionTests(unittest.TestCase):
    def test_unknown_batch_is_refused(self):
        with self.assertRaises(SystemExit):
            deployer._batch(_manifest(), "9-nonexistent")

    def test_migration_entries_are_never_copied_as_files(self):
        schema = deployer._batch(_manifest(), "2-schema")

        self.assertEqual(deployer._deployable(schema), [])


class SafetyTests(unittest.TestCase):
    def test_apply_aborts_when_the_source_drifted_from_the_manifest(self):
        manifest = _manifest()
        batch = deployer._batch(manifest, "1-selfheal")

        with mock.patch.object(deployer.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout=b"")
            with self.assertRaisesRegex(SystemExit, "regenerate the manifest"):
                deployer.apply(manifest, batch)

    def test_apply_aborts_when_the_deployed_file_does_not_match(self):
        source = ROOT / "scripts/deepgraph_selfheal.py"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = _manifest()
        manifest["batches"][0]["files"][0]["source_sha256"] = digest
        batch = deployer._batch(manifest, "1-selfheal")

        # Reading the target back returns different bytes than were installed.
        with mock.patch.object(deployer, "_read_target", return_value=b"tampered"), \
                mock.patch.object(deployer.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout=b"")
            with self.assertRaisesRegex(SystemExit, "post-deploy SHA256 mismatch"):
                deployer.apply(manifest, batch)

    def test_verify_reports_absent_and_drifted_targets(self):
        manifest = _manifest()
        batch = deployer._batch(manifest, "1-selfheal")

        with mock.patch.object(deployer, "_read_target", return_value=None):
            self.assertEqual(deployer.verify(manifest, batch), 1)
        with mock.patch.object(deployer, "_read_target", return_value=b"other"):
            self.assertEqual(deployer.verify(manifest, batch), 1)

    def test_rollback_requires_a_recorded_ledger(self):
        manifest = _manifest()
        batch = deployer._batch(manifest, "1-selfheal")

        with mock.patch.object(deployer, "_read_target", return_value=None):
            with self.assertRaisesRegex(SystemExit, "no deployment ledger"):
                deployer.rollback(manifest, batch)

    def test_rollback_verifies_the_restored_digest(self):
        manifest = _manifest()
        batch = deployer._batch(manifest, "1-selfheal")
        ledger = json.dumps(
            {
                "files": [
                    {
                        "target": "/usr/local/bin/deepgraph-selfheal.py",
                        "backup": "/tmp/backup.bak",
                        "previous_sha256": "c" * 64,
                    }
                ]
            }
        ).encode("utf-8")

        with mock.patch.object(
            deployer, "_read_target", side_effect=[ledger, b"not-the-backup"]
        ), mock.patch.object(deployer.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0)
            with self.assertRaisesRegex(SystemExit, "rollback SHA256 mismatch"):
                deployer.rollback(manifest, batch)


class RealManifestTests(unittest.TestCase):
    def test_the_committed_manifest_pins_every_source_it_names(self):
        manifest = json.loads(
            (ROOT / "deploy/manifest/recovery_2026-08-03.json").read_text(
                encoding="utf-8"
            )
        )
        for batch in manifest["batches"]:
            for entry in batch["files"]:
                source = ROOT / entry["source"]
                with self.subTest(source=entry["source"]):
                    self.assertTrue(source.is_file())
                    self.assertEqual(
                        hashlib.sha256(source.read_bytes()).hexdigest(),
                        entry["source_sha256"],
                    )


if __name__ == "__main__":
    unittest.main()
