import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ConfigTomlTests(unittest.TestCase):
    def _run_config_probe(self, toml_text: str, extra_env: dict[str, str] | None = None) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            toml_path = Path(tmp) / "deepgraph.toml"
            toml_path.write_text(textwrap.dedent(toml_text).strip() + "\n", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONPATH": str(ROOT),
                    "DEEPGRAPH_CONFIG_TOML": str(toml_path),
                    "DEEPGRAPH_APP_NAME": "",
                    "DEEPGRAPH_WEB_PORT": "",
                    "DEEPGRAPH_WORKSPACE_DIR": "",
                    "DEEPGRAPH_RUNTIME_PYTHON": "",
                }
            )
            if extra_env:
                env.update(extra_env)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import config; "
                        "print(config.APP_NAME); "
                        "print(config.WEB_PORT); "
                        "print(config.WORKSPACE_DIR); "
                        "print(bool(config.RUNTIME_PYTHON))"
                    ),
                ],
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            return proc.stdout.strip()

    def test_toml_defaults_are_loaded(self):
        output = self._run_config_probe(
            """
            [app]
            name = "TomlGraph"

            [web]
            port = 8181

            [paths]
            workspace_dir = "{project_root}/workspace_from_toml"

            [runtime]
            python = ""
            """
        )

        lines = output.splitlines()
        self.assertEqual(lines[0], "TomlGraph")
        self.assertEqual(lines[1], "8181")
        self.assertTrue(lines[2].endswith("workspace_from_toml"))
        self.assertEqual(lines[3], "True")

    def test_environment_overrides_toml(self):
        output = self._run_config_probe(
            """
            [app]
            name = "TomlGraph"

            [web]
            port = 8181
            """,
            {"DEEPGRAPH_APP_NAME": "EnvGraph", "DEEPGRAPH_WEB_PORT": "9090"},
        )

        lines = output.splitlines()
        self.assertEqual(lines[0], "EnvGraph")
        self.assertEqual(lines[1], "9090")


if __name__ == "__main__":
    unittest.main()
