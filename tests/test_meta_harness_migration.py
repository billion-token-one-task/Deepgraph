"""Unit coverage for guards that protect the one-time local migration path."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import meta_harness_migration as migration


class LiveLocalMigrationGuardTests(unittest.TestCase):
    def test_live_local_url_requires_exact_loopback_endpoint_and_database(self):
        migration._validate_live_local_url(  # noqa: SLF001
            "postgresql://operator:secret@127.0.0.1:5433/deepgraph"
        )
        for url in (
            "postgresql://operator:secret@localhost:5433/deepgraph",
            "postgresql://operator:secret@127.0.0.1:5432/deepgraph",
            "postgresql://operator:secret@127.0.0.1:5433/deepgraph_restore",
        ):
            with self.subTest(url=url), self.assertRaises(SystemExit):
                migration._validate_live_local_url(url)  # noqa: SLF001

    def test_backup_must_be_directly_under_home_and_match_recorded_digest(self):
        with tempfile.NamedTemporaryFile(
            dir=migration.LIVE_LOCAL_BACKUP_DIRECTORY,  # noqa: SLF001
            prefix="test-meta-harness-backup-",
            suffix=".dump",
            delete=False,
        ) as handle:
            handle.write(b"custom backup fixture")
            backup_path = Path(handle.name)
        self.addCleanup(backup_path.unlink, missing_ok=True)
        digest = hashlib.sha256(b"custom backup fixture").hexdigest()

        verified = migration._verified_backup(str(backup_path), digest)  # noqa: SLF001

        self.assertEqual(verified["backup_file"], str(backup_path))
        self.assertEqual(verified["backup_sha256"], digest)
        with self.assertRaises(SystemExit):
            migration._verified_backup(str(backup_path), "0" * 64)  # noqa: SLF001

    @patch("scripts.meta_harness_migration.subprocess.run")
    def test_service_check_fails_closed_unless_inactive(self, run: Mock):
        run.return_value = Mock(returncode=0, stdout="active\n")
        with self.assertRaises(SystemExit):
            migration._require_service_stopped()  # noqa: SLF001

        run.return_value = Mock(returncode=0, stdout="inactive\n")
        migration._require_service_stopped()  # noqa: SLF001

    def test_live_local_opt_in_is_required_before_database_access(self):
        with patch.dict(os.environ, {"DEEPGRAPH_ALLOW_LIVE_LOCAL_MIGRATION": ""}, clear=False):
            with self.assertRaises(SystemExit):
                migration.apply_to_live_local(
                    "postgresql://operator:secret@127.0.0.1:5433/deepgraph",
                    source_commit="a" * 40,
                    backup_file="/home/ec2-user/not-used.dump",
                    backup_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
