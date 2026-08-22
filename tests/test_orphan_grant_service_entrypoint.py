"""The orphan-grant operator must run from the activated immutable release."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "repair_orphaned_grants.py"
UNIT = ROOT / "ops" / "v1" / "deepgraph-orphan-grants.service"


class OrphanGrantServiceEntrypointTests(unittest.TestCase):
    def test_absolute_script_entrypoint_imports_outside_repository_cwd(self):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--help"],
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--min-age-minutes", result.stdout)

    def test_unit_follows_the_release_activated_for_web(self):
        unit = UNIT.read_text(encoding="utf-8")

        self.assertNotIn(
            "WorkingDirectory=/home/billion-token/Deepgraph", unit
        )
        self.assertIn(
            "systemctl show --property=WorkingDirectory --value "
            "deepgraph-web.service",
            unit,
        )
        self.assertIn(
            "/home/billion-token/releases/deepgraph-v1-*", unit
        )
        self.assertIn(
            '"$$release_dir/scripts/repair_orphaned_grants.py" --apply', unit
        )
        self.assertIn('test -r "$$release_dir/.release-commit"', unit)


if __name__ == "__main__":
    unittest.main()
