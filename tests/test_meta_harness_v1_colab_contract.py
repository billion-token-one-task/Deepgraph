"""Pure Colab adapter validation; no CLI, network, OAuth, or GPU calls."""

from __future__ import annotations

import unittest

from meta_harness.backends.colab_cli import ColabCLIConfig, ColabCLIError


class ColabContractTests(unittest.TestCase):
    def test_isolated_code_and_artifact_roots_are_required(self):
        config = ColabCLIConfig(
            binary="colab-cli",
            allowed_code_root="",
            allowed_artifact_root="",
        )
        with self.assertRaises(ColabCLIError):
            config.validate()

    def test_dependency_install_cannot_be_enabled(self):
        config = ColabCLIConfig(
            binary="colab-cli",
            allowed_code_root="/tmp/meta-harness-candidates",
            allowed_artifact_root="/tmp/meta-harness-artifacts",
            allow_dependency_install=True,
        )
        with self.assertRaises(ColabCLIError):
            config.validate()


if __name__ == "__main__":
    unittest.main()
