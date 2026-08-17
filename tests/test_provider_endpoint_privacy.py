"""Declared provider endpoints must not hard-code a private gateway.

deepgraph.toml is tracked in a public repository. An API key is not the only
site-specific value in a provider block: the endpoint of a private gateway
identifies the operator's infrastructure just as directly.
"""

import re
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Endpoints that are safe to publish because they are public commercial APIs.
PUBLIC_HOSTS = {
    "api.example.com",
    "api.openai.com",
    "api.deepseek.com",
    "api.novita.ai",
    "api.minimaxi.com",
    "api2.tabcode.cc",
    "127.0.0.1",
    "localhost",
}


class ProviderEndpointPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.config = tomllib.loads((REPO / "deepgraph.toml").read_text())

    def test_declared_providers_do_not_publish_a_private_host(self):
        for entry in self.config.get("llm", {}).get("providers", []) or []:
            literal = str(entry.get("base_url") or "").strip()
            if not literal:
                self.assertTrue(
                    str(entry.get("base_url_env") or "").strip(),
                    f"provider {entry.get('name')} declares neither base_url nor base_url_env",
                )
                continue
            host = re.sub(r"^[a-z]+://", "", literal).split("/")[0].split(":")[0]
            self.assertIn(
                host,
                PUBLIC_HOSTS,
                f"provider {entry.get('name')} hard-codes {host}; use base_url_env",
            )

    def test_no_literal_api_key_in_tracked_config(self):
        for entry in self.config.get("llm", {}).get("providers", []) or []:
            self.assertFalse(
                str(entry.get("api_key") or "").strip(),
                f"provider {entry.get('name')} declares a literal api_key",
            )


if __name__ == "__main__":
    unittest.main()
