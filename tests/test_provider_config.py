"""The provider page must never become a credential or exfiltration surface."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from web import provider_config


ALLOWED = ["api.example.com", "api2.tabcode.cc"]


def _entry(**overrides) -> dict:
    entry = {
        "name": "gpt5",
        "base_url": "https://api.example.com/v1/",
        "model": "gpt-5.4",
        "model_family": "gpt",
        "protocol": "chat_completions",
        "api_key_env": "DEEPGRAPH_LLM_GPT5_API_KEY",
        "rpm": 0,
        "enabled": True,
    }
    entry.update(overrides)
    return entry


class ValidationTests(unittest.TestCase):
    def test_a_valid_entry_is_normalized(self):
        entry = provider_config.validate_entry(_entry(), allowed_hosts=ALLOWED)

        self.assertEqual(entry["base_url"], "https://api.example.com/v1")
        self.assertEqual(entry["model_family"], "gpt")
        self.assertNotIn("api_key", entry)

    def test_a_credential_field_is_refused(self):
        for field in ("api_key", "apikey", "key", "token", "secret"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    provider_config.ProviderConfigError, "never carry a credential"
                ):
                    provider_config.validate_entry(
                        _entry(**{field: "sk-live"}), allowed_hosts=ALLOWED
                    )

    def test_base_url_must_be_https_and_allowlisted(self):
        with self.assertRaisesRegex(provider_config.ProviderConfigError, "https"):
            provider_config.validate_entry(
                _entry(base_url="http://api.example.com/v1"), allowed_hosts=ALLOWED
            )
        with self.assertRaisesRegex(
            provider_config.ProviderConfigError, "not in the operator allowlist"
        ):
            provider_config.validate_entry(
                _entry(base_url="https://attacker.example.net/v1"),
                allowed_hosts=ALLOWED,
            )

    def test_no_allowlist_means_no_route_can_be_added(self):
        with self.assertRaisesRegex(
            provider_config.ProviderConfigError, "no provider host allowlist"
        ):
            provider_config.validate_entry(_entry(), allowed_hosts=[])

    def test_key_variable_must_follow_the_reserved_pattern(self):
        for bad in ("AWS_SECRET_ACCESS_KEY", "DEEPGRAPH_DATABASE_URL", "PATH", ""):
            with self.subTest(name=bad):
                with self.assertRaisesRegex(
                    provider_config.ProviderConfigError, "api_key_env"
                ):
                    provider_config.validate_entry(
                        _entry(api_key_env=bad), allowed_hosts=ALLOWED
                    )

    def test_model_family_is_mandatory_because_independence_depends_on_it(self):
        with self.assertRaisesRegex(
            provider_config.ProviderConfigError, "independence"
        ):
            provider_config.validate_entry(
                _entry(model_family=""), allowed_hosts=ALLOWED
            )

    def test_name_protocol_and_rpm_are_constrained(self):
        with self.assertRaises(provider_config.ProviderConfigError):
            provider_config.validate_entry(_entry(name="Bad Name"), allowed_hosts=ALLOWED)
        with self.assertRaises(provider_config.ProviderConfigError):
            provider_config.validate_entry(_entry(protocol="telnet"), allowed_hosts=ALLOWED)
        with self.assertRaises(provider_config.ProviderConfigError):
            provider_config.validate_entry(_entry(rpm=-1), allowed_hosts=ALLOWED)

    def test_unknown_fields_are_refused(self):
        with self.assertRaisesRegex(
            provider_config.ProviderConfigError, "unsupported fields"
        ):
            provider_config.validate_entry(
                _entry(headers={"x": "y"}), allowed_hosts=ALLOWED
            )


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="provider-store-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "config" / "llm_providers.json"

    def test_round_trip_and_permissions(self):
        provider_config.upsert(
            self.path, _entry(), allowed_hosts=ALLOWED, actor="operator:alice"
        )
        store = provider_config.load_store(self.path)

        self.assertEqual([p["name"] for p in store["providers"]], ["gpt5"])
        self.assertEqual(store["updated_by"], "operator:alice")
        self.assertEqual(oct(self.path.stat().st_mode & 0o777), "0o640")

    def test_upsert_replaces_by_name_and_keeps_order(self):
        provider_config.upsert(self.path, _entry(), allowed_hosts=ALLOWED, actor="a")
        provider_config.upsert(
            self.path, _entry(name="aglm", model_family="glm"), allowed_hosts=ALLOWED, actor="a"
        )
        provider_config.upsert(
            self.path, _entry(model="gpt-5.5"), allowed_hosts=ALLOWED, actor="a"
        )
        store = provider_config.load_store(self.path)

        self.assertEqual([p["name"] for p in store["providers"]], ["aglm", "gpt5"])
        self.assertEqual(store["providers"][1]["model"], "gpt-5.5")

    def test_remove_reports_whether_anything_changed(self):
        provider_config.upsert(self.path, _entry(), allowed_hosts=ALLOWED, actor="a")

        self.assertTrue(provider_config.remove(self.path, "gpt5", actor="a"))
        self.assertFalse(provider_config.remove(self.path, "gpt5", actor="a"))

    def test_an_actor_is_required_for_every_change(self):
        with self.assertRaises(provider_config.ProviderConfigError):
            provider_config.save_store(self.path, [], actor=" ")

    def test_a_corrupt_store_reads_as_empty_rather_than_crashing(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{not json", encoding="utf-8")

        self.assertEqual(provider_config.load_store(self.path)["providers"], [])


class ReadinessTests(unittest.TestCase):
    def test_readiness_never_exposes_the_key_itself(self):
        with mock.patch.dict(os.environ, {"DEEPGRAPH_LLM_GPT5_API_KEY": "sk-secret"}):
            described = provider_config.readiness([_entry()])

        self.assertTrue(described[0]["key_present"])
        self.assertTrue(described[0]["ready"])
        blob = json.dumps(described)
        self.assertNotIn("sk-secret", blob)
        self.assertNotIn("secret", blob.replace("DEEPGRAPH_LLM_GPT5_API_KEY", ""))

    def test_a_missing_key_is_reported_as_not_ready(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEEPGRAPH_LLM_GPT5_API_KEY", None)
            described = provider_config.readiness([_entry()])

        self.assertFalse(described[0]["key_present"])
        self.assertFalse(described[0]["ready"])

    def test_a_disabled_route_is_never_ready(self):
        with mock.patch.dict(os.environ, {"DEEPGRAPH_LLM_GPT5_API_KEY": "k"}):
            described = provider_config.readiness([_entry(enabled=False)])

        self.assertTrue(described[0]["key_present"])
        self.assertFalse(described[0]["ready"])


class IndependenceTests(unittest.TestCase):
    def test_one_family_is_not_independent(self):
        with mock.patch.dict(
            os.environ,
            {
                "DEEPGRAPH_LLM_GPT5_API_KEY": "k",
                "DEEPGRAPH_LLM_GLM_API_KEY": "k",
            },
        ):
            report = provider_config.independence_report(
                [
                    _entry(),
                    _entry(
                        name="glm",
                        model="glm-5.2",
                        model_family="gpt",
                        api_key_env="DEEPGRAPH_LLM_GLM_API_KEY",
                    ),
                ]
            )

        self.assertEqual(report["ready_routes"], 2)
        self.assertFalse(report["independent_evaluator_possible"])

    def test_two_families_make_an_independent_evaluator_possible(self):
        with mock.patch.dict(
            os.environ,
            {
                "DEEPGRAPH_LLM_GPT5_API_KEY": "k",
                "DEEPGRAPH_LLM_GLM_API_KEY": "k",
            },
        ):
            report = provider_config.independence_report(
                [
                    _entry(),
                    _entry(
                        name="glm",
                        model="glm-5.2",
                        model_family="glm",
                        api_key_env="DEEPGRAPH_LLM_GLM_API_KEY",
                    ),
                ]
            )

        self.assertTrue(report["independent_evaluator_possible"])
        self.assertEqual(report["distinct_model_families"], ["glm", "gpt"])

    def test_an_unready_route_does_not_count_towards_independence(self):
        with mock.patch.dict(os.environ, {"DEEPGRAPH_LLM_GPT5_API_KEY": "k"}):
            os.environ.pop("DEEPGRAPH_LLM_GLM_API_KEY", None)
            report = provider_config.independence_report(
                [
                    _entry(),
                    _entry(
                        name="glm",
                        model_family="glm",
                        api_key_env="DEEPGRAPH_LLM_GLM_API_KEY",
                    ),
                ]
            )

        self.assertEqual(report["ready_routes"], 1)
        self.assertFalse(report["independent_evaluator_possible"])


class EffectivePoolTests(unittest.TestCase):
    """The page must report what the process would actually route to."""

    def test_runtime_entries_are_ready_without_an_env_variable(self):
        described = provider_config.readiness(
            [{"name": "secondary", "model": "glm-5.2", "model_family": "glm",
              "source": "runtime", "api_key_env": "", "enabled": True}]
        )

        self.assertTrue(described[0]["ready"])
        self.assertTrue(described[0]["key_present"])

    def test_independence_counts_legacy_slots_not_only_the_store(self):
        # The store knows nothing about the env-configured "secondary" slot, so
        # judging independence from the store alone said "not possible" while a
        # second family was in fact live.
        pool = [
            {"name": "secondary", "model": "glm-5.2", "model_family": "glm",
             "source": "runtime", "api_key_env": "", "enabled": True},
            {"name": "sora2_gemini", "model": "gemini-3.6-flash-high",
             "model_family": "gemini", "source": "runtime", "api_key_env": "",
             "enabled": True},
        ]
        report = provider_config.independence_report(pool)

        self.assertEqual(report["ready_routes"], 2)
        self.assertTrue(report["independent_evaluator_possible"])
        self.assertEqual(report["distinct_model_families"], ["gemini", "glm"])

    def test_effective_pool_never_returns_a_credential(self):
        fake = [{"name": "secondary", "model": "glm-5.2", "model_family": "glm",
                 "api_key": "sk-live-secret", "base_url": "https://x/v1",
                 "protocol": "chat_completions", "rpm": 0}]
        with mock.patch.dict("sys.modules"):
            import agents.llm_client as client
            with mock.patch.object(client, "_providers", fake), \
                    mock.patch.object(client, "_init_providers", lambda: None):
                pool = provider_config.effective_pool()

        self.assertEqual(pool[0]["name"], "secondary")
        self.assertNotIn("sk-live-secret", json.dumps(pool))
        self.assertNotIn("api_key", json.dumps(pool).replace("api_key_env", ""))

    def test_a_broken_provider_pool_degrades_to_empty_not_an_exception(self):
        import agents.llm_client as client

        with mock.patch.object(
            client, "_init_providers", side_effect=RuntimeError("no providers")
        ):
            self.assertEqual(provider_config.effective_pool(), [])


if __name__ == "__main__":
    unittest.main()
